import os
import asyncio
from typing import Dict, List

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available. Install with: pip install ragas"}

    # Guard against empty inputs
    if not question or not answer:
        return {"error": "Question and answer must not be empty"}

    if not contexts:
        contexts = [""]

    try:
        # TODO: Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url="https://openai.vocareum.com/v1"
            )
        )

        # TODO: Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url="https://openai.vocareum.com/v1"
            )
        )

        # Create a single-turn sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # TODO: Define an instance for each metric to evaluate
        metrics = {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            ),
            "bleu_score": BleuScore(),
            "rouge_score": RougeScore(),
        }

        # TODO: Evaluate the response using the metrics
        results = {}
        for name, metric in metrics.items():
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        score = loop.run_until_complete(metric.single_turn_ascore(sample))
                    else:
                        score = asyncio.run(metric.single_turn_ascore(sample))
                except RuntimeError:
                    score = asyncio.run(metric.single_turn_ascore(sample))

                results[name] = round(float(score), 4)

            except Exception as e:
                results[name] = 0.0

        # TODO: Return the evaluation results
        return results

    except Exception as e:
        return {"error": str(e)}