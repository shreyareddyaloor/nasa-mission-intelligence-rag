#!/usr/bin/env python3
"""
End-to-end evaluation runner for NASA Mission Intelligence RAG System.
Loads evaluation_dataset.txt, runs each question through the full pipeline,
and outputs per-question scores plus aggregate metrics.

Usage:
    python evaluate.py --openai-key $OPENAI_API_KEY
    python evaluate.py --openai-key $OPENAI_API_KEY --chroma-dir ./chroma_db_openai --k 3
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from llm_client import generate_response
from rag_client import initialize_rag_system, retrieve_documents, format_context
from ragas_evaluator import evaluate_response_quality

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_questions(dataset_path: str) -> List[str]:
    """Load questions from evaluation_dataset.txt"""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    questions = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(questions)} questions from {dataset_path}")
    return questions


def run_evaluation(questions: List[str], collection, openai_key: str,
                   model: str, k: int) -> List[Dict]:
    """Run full RAG pipeline for each question and evaluate"""
    results = []

    for i, question in enumerate(questions):
        logger.info(f"\n[{i+1}/{len(questions)}] Question: {question}")

        try:
            # Step 1: Retrieve relevant documents
            raw_results = retrieve_documents(collection, question, n_results=k,
                                             openai_key=openai_key)
            documents = raw_results["documents"][0] if raw_results else []
            metadatas = raw_results["metadatas"][0] if raw_results else []

            # Step 2: Format context
            context = format_context(documents, metadatas)

            # Step 3: Generate answer
            answer = generate_response(
                openai_key=openai_key,
                user_message=question,
                context=context,
                conversation_history=[],
                model=model
            )

            logger.info(f"Answer: {answer[:200]}...")

            # Step 4: Evaluate quality
            scores = evaluate_response_quality(
                question=question,
                answer=answer,
                contexts=documents
            )

            logger.info(f"Scores: {scores}")

            results.append({
                "question": question,
                "answer": answer,
                "contexts_used": len(documents),
                "scores": scores
            })

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            results.append({
                "question": question,
                "answer": "ERROR",
                "contexts_used": 0,
                "scores": {"error": str(e)}
            })

    return results


def print_summary(results: List[Dict]):
    """Print per-question scores and aggregate metrics"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS — PER QUESTION")
    print("="*70)

    metric_names = ["faithfulness", "answer_relevancy", "bleu_score", "rouge_score"]
    all_scores = {m: [] for m in metric_names}

    for i, result in enumerate(results):
        print(f"\nQ{i+1}: {result['question']}")
        print(f"  Contexts used: {result['contexts_used']}")
        scores = result["scores"]
        if "error" in scores:
            print(f"  ERROR: {scores['error']}")
        else:
            for metric in metric_names:
                score = scores.get(metric, 0.0)
                print(f"  {metric:20s}: {score:.4f}")
                all_scores[metric].append(score)

    print("\n" + "="*70)
    print("AGGREGATE METRICS (MEAN)")
    print("="*70)
    for metric in metric_names:
        vals = all_scores[metric]
        if vals:
            mean = sum(vals) / len(vals)
            min_v = min(vals)
            max_v = max(vals)
            print(f"  {metric:20s}: mean={mean:.4f}  min={min_v:.4f}  max={max_v:.4f}")

    print("="*70)
    return all_scores


def main():
    parser = argparse.ArgumentParser(description='End-to-end RAG Evaluation Runner')
    parser.add_argument('--openai-key', default=os.environ.get("OPENAI_API_KEY"), help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--dataset', default='./evaluation_dataset.txt', help='Path to evaluation questions')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--k', type=int, default=3, help='Number of documents to retrieve per question (top-k)')
    parser.add_argument('--output', default='./evaluation_results.json', help='Output JSON file for results')

    args = parser.parse_args()

    if not args.openai_key:
        logger.error("No OpenAI API key. Set OPENAI_API_KEY or pass --openai-key")
        return

    # Load questions
    questions = load_questions(args.dataset)

    # Connect to ChromaDB
    collection, success, error = initialize_rag_system(args.chroma_dir, args.collection_name)
    if not success:
        logger.error(f"Failed to connect to ChromaDB: {error}")
        return
    logger.info(f"Connected to collection '{args.collection_name}' with {collection.count()} documents")

    # Run evaluation
    results = run_evaluation(questions, collection, args.openai_key, args.model, args.k)

    # Print summary
    print_summary(results)

    # Save results to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()