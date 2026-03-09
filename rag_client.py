import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
from openai import OpenAI


def get_query_embedding(query: str, openai_key: str = None) -> List[float]:
    """Generate OpenAI embedding for a query string (1536 dims)"""
    api_key = openai_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for directories that look like ChromaDB stores
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and ('chroma' in d.name.lower() or 'db' in d.name.lower())
        and not d.name.startswith('.')
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}::{collection.name}"
                try:
                    count = collection.count()
                except Exception:
                    count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({count} docs) — {chroma_dir.name}",
                }

        except Exception as e:
            # Still add it so user sees it in UI, but mark as error
            backends[str(chroma_dir)] = {
                "directory": str(chroma_dir),
                "collection_name": "unknown",
                "display_name": f"{chroma_dir.name} — Error: {str(e)[:40]}",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple:
    """Initialize the RAG system with specified backend"""
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None,
                       openai_key: str = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Build optional metadata filter
    where_filter = None
    if mission_filter and mission_filter.strip().lower() not in ["all", "none", "", "any"]:
        where_filter = {"mission": mission_filter}

    # Generate OpenAI embedding (1536 dims) to match collection
    query_embedding = get_query_embedding(query, openai_key)

    # Execute semantic search using query_embeddings (not query_texts)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Deduplicate by chunk id and sort by score (lower distance = better match)
    if results and results.get("ids") and results["ids"][0]:
        seen_ids = set()
        deduped_ids, deduped_docs, deduped_meta, deduped_dist = [], [], [], []

        combined = zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )

        # Sort by distance ascending (closest = most relevant first)
        sorted_combined = sorted(combined, key=lambda x: x[3])

        for doc_id, doc, meta, dist in sorted_combined:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduped_ids.append(doc_id)
                deduped_docs.append(doc)
                deduped_meta.append(meta)
                deduped_dist.append(dist)

        results["ids"][0] = deduped_ids
        results["documents"][0] = deduped_docs
        results["metadatas"][0] = deduped_meta
        results["distances"][0] = deduped_dist

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into a clean context string for the LLM"""
    if not documents:
        return ""

    context_parts = ["=== Retrieved Context from NASA Documents ===\n"]

    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        # Extract and clean metadata fields
        mission = metadata.get('mission', 'unknown').replace('_', ' ').title()
        source = metadata.get('source', 'unknown')
        category = metadata.get('document_category', 'unknown').replace('_', ' ').title()

        # Format source header
        header = f"[Source {i + 1}] Mission: {mission} | File: {source} | Category: {category}"
        context_parts.append(header)
        context_parts.append("-" * len(header))

        # Truncate very long chunks to avoid overwhelming the LLM context window
        if len(doc) > 800:
            doc = doc[:800] + "... [truncated]"

        context_parts.append(doc)
        context_parts.append("")  # blank line between sources

    return "\n".join(context_parts)