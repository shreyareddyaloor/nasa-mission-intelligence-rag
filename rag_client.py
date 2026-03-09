import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # TODO: Create list of directories that match specific criteria
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and ('chroma' in d.name.lower() or 'db' in d.name.lower())
        and not d.name.startswith('.')
    ]

    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:

        # TODO: Wrap connection attempt in try-except block
        try:
            # TODO: Initialize database client
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # TODO: Retrieve list of available collections
            collections = client.list_collections()

            # TODO: Loop through each collection found
            for collection in collections:

                # TODO: Create unique identifier key
                key = f"{chroma_dir.name}::{collection.name}"

                # TODO: Get document count with fallback
                try:
                    count = collection.count()
                except Exception:
                    count = 0

                # TODO: Build information dictionary
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({count} docs) — {chroma_dir.name}",
                }

        # TODO: Handle connection errors gracefully
        except Exception as e:
            backends[str(chroma_dir)] = {
                "directory": str(chroma_dir),
                "collection_name": "unknown",
                "display_name": f"{chroma_dir.name} — Error: {str(e)[:40]}",
            }

    # TODO: Return complete backends dictionary
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""

    # TODO: Create a chromadb persistent client
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        # TODO: Return the collection with the collection_name
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None
    where_filter = None

    # TODO: Check if filter parameter exists and is not set to "all"
    if mission_filter and mission_filter.strip().lower() not in ["all", "none", "", "any"]:
        # TODO: Create filter dictionary
        where_filter = {"mission": mission_filter}

    # TODO: Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # TODO: Return query results
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # TODO: Initialize list with header text
    context_parts = ["=== Retrieved Context from NASA Documents ===\n"]

    # TODO: Loop through paired documents and metadata
    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):

        # TODO: Extract mission information with fallback
        mission = metadata.get('mission', 'unknown').replace('_', ' ').title()
        # TODO: Extract source information with fallback
        source = metadata.get('source', 'unknown')
        # TODO: Extract category information with fallback
        category = metadata.get('document_category', 'unknown').replace('_', ' ').title()

        # TODO: Create formatted source header
        header = f"[Source {i + 1}] Mission: {mission} | File: {source} | Category: {category}"
        # TODO: Add source header to context parts
        context_parts.append(header)
        context_parts.append("-" * len(header))

        # TODO: Check document length and truncate if necessary
        if len(doc) > 800:
            doc = doc[:800] + "... [truncated]"

        # TODO: Add document content to context parts
        context_parts.append(doc)
        context_parts.append("")

    # TODO: Join all context parts and return
    return "\n".join(context_parts)