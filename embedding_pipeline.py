#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only
"""

import os
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        # TODO: Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url="https://openai.vocareum.com/v1"
        )

        # TODO: Store configuration parameters
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

        # TODO: Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # TODO: Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Collection '{collection_name}' ready. "
                    f"Current document count: {self.collection.count()}")

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into chunks with metadata"""

        # TODO: Handle short texts that don't need chunking
        if len(text) <= self.chunk_size:
            chunk_metadata = {**metadata, "chunk_index": 0, "total_chunks": 1,
                              "chunk_start": 0, "chunk_end": len(text)}
            return [(text.strip(), chunk_metadata)]

        chunks = []
        start = 0
        chunk_index = 0

        # TODO: Implement chunking logic with overlap
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # TODO: Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size * 0.5:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1

            chunk_stripped = chunk.strip()
            if chunk_stripped:
                # TODO: Create metadata for each chunk
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": min(end, len(text)),
                    "total_chunks": 0
                }
                chunks.append((chunk_stripped, chunk_metadata))
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        total = len(chunks)
        for _, meta in chunks:
            meta["total_chunks"] = total

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID already exists in the collection"""
        # TODO: Query collection for document ID
        try:
            result = self.collection.get(ids=[doc_id])
            # TODO: Return True if exists, False otherwise
            return len(result['ids']) > 0
        except Exception:
            return False

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing document in the collection"""
        try:
            embedding = self.get_embedding(text)
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """Delete all documents from a specific source"""
        try:
            all_docs = self.collection.get()
            ids_to_delete = [
                all_docs['ids'][i]
                for i, metadata in enumerate(all_docs['metadatas'])
                if source_pattern in metadata.get('source', '')
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching: {source_pattern}")
                return len(ids_to_delete)
            logger.info(f"No documents found matching: {source_pattern}")
            return 0
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """Get all document IDs for a specific file"""
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()
            return [
                all_docs['ids'][i]
                for i, metadata in enumerate(all_docs['metadatas'])
                if metadata.get('source') == source and metadata.get('mission') == mission
            ]
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        # TODO: Call OpenAI embeddings API
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            # TODO: Return embedding vector
            return response.data[0].embedding
        except Exception as e:
            # TODO: Add error handling
            logger.error(f"Error getting embedding: {e}")
            raise

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Generate stable document ID based on file path and chunk position"""
        # TODO: Use mission, source, and chunk_index
        mission = metadata.get('mission', 'unknown')
        source = metadata.get('source', 'unknown')
        chunk_index = metadata.get('chunk_index', 0)
        # TODO: Create consistent ID format: mission_source_chunk_0001
        return f"{mission}_{source}_chunk_{chunk_index:04d}"

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """Process plain text files with enhanced metadata extraction"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename"""
        fn = filename.lower()
        if 'pao' in fn:
            return 'public_affairs_officer'
        elif '_cm_' in fn or fn.endswith('cm.txt'):
            return 'command_module'
        elif 'tec' in fn:
            return 'technical'
        elif 'flight_plan' in fn:
            return 'flight_plan'
        elif 'mission_audio' in fn:
            return 'mission_audio'
        elif 'ntrs' in fn:
            return 'nasa_archive'
        elif '19900066485' in fn:
            return 'technical_report'
        elif '19710015566' in fn:
            return 'mission_report'
        elif 'full_text' in fn:
            return 'complete_document'
        return 'general_document'

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """Scan data directories for text files only"""
        base_path = Path(base_path)
        files_to_process = []
        data_dirs = ['apollo11', 'apollo13', 'challenger']

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        filtered = [
            f for f in files_to_process
            if not f.name.startswith('.')
            and 'summary' not in f.name.lower()
            and f.suffix.lower() == '.txt'
        ]

        logger.info(f"Total text files to process: {len(filtered)}")
        return filtered

    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]],
                                    file_path: Path, batch_size: int = 50,
                                    update_mode: str = 'skip') -> Dict[str, int]:
        """Add documents to ChromaDB collection in batches with update handling"""
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        # TODO: Handle different update modes (skip, update, replace)
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"Replace mode: deleted {len(existing_ids)} existing docs")

        # TODO: Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            for text, metadata in batch:
                doc_id = self.generate_document_id(file_path, metadata)
                exists = self.check_document_exists(doc_id)

                if exists:
                    if update_mode == 'update':
                        success = self.update_document(doc_id, text, metadata)
                        if success:
                            stats['updated'] += 1
                    else:
                        stats['skipped'] += 1
                    continue

                # TODO: Get embedding and add to collection
                try:
                    embedding = self.get_embedding(text)
                    self.collection.add(
                        ids=[doc_id],
                        documents=[text],
                        metadatas=[metadata],
                        embeddings=[embedding]
                    )
                    stats['added'] += 1
                    time.sleep(0.05)
                except Exception as e:
                    logger.error(f"Error adding document {doc_id}: {e}")

            logger.info(f"Batch {i // batch_size + 1} done. "
                        f"Added: {stats['added']}, Skipped: {stats['skipped']}")

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip') -> Dict[str, Any]:
        """Process all text files and add to ChromaDB"""
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        # TODO: Get files to process
        files = self.scan_text_files_only(base_path)

        # TODO: Loop through each file
        for file_path in files:
            try:
                mission = self.extract_mission_from_path(file_path)
                logger.info(f"Processing: {file_path.name} [{mission}]")

                # TODO: Process file and add to collection
                documents = self.process_text_file(file_path)
                if not documents:
                    continue

                file_stats = self.add_documents_to_collection(
                    documents, file_path, batch_size=50, update_mode=update_mode
                )

                # TODO: Update statistics
                stats['files_processed'] += 1
                stats['total_chunks'] += len(documents)
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']

                if mission not in stats['missions']:
                    stats['missions'][mission] = {
                        'files': 0, 'chunks': 0,
                        'added': 0, 'updated': 0, 'skipped': 0
                    }
                stats['missions'][mission]['files'] += 1
                stats['missions'][mission]['chunks'] += len(documents)
                stats['missions'][mission]['added'] += file_stats['added']
                stats['missions'][mission]['updated'] += file_stats['updated']
                stats['missions'][mission]['skipped'] += file_stats['skipped']

            # TODO: Handle errors gracefully
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        # TODO: Return collection name, document count, metadata
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection.count(),
            'embedding_model': self.embedding_model
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the collection for testing"""
        # TODO: Perform test query and return results
        embedding = self.get_embedding(query_text)
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            all_docs = self.collection.get()
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }

            for metadata in all_docs['metadatas']:
                for key, field in [
                    ('missions', 'mission'),
                    ('data_types', 'data_type'),
                    ('document_categories', 'document_category'),
                    ('file_types', 'file_type')
                ]:
                    val = metadata.get(field, 'unknown')
                    stats[key][val] = stats[key].get(val, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='./data_text', help='Path to data directories')
    parser.add_argument('--openai-key', default=os.environ.get("OPENAI_API_KEY"), help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                        help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents matching a source pattern')

    args = parser.parse_args()

    if not args.openai_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY or pass --openai-key")
        return

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    if args.delete_source:
        deleted = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted} documents matching: {args.delete_source}")
        return

    if args.stats_only:
        logger.info("=== Collection Statistics ===")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    logger.info(f"Starting processing with update mode: {args.update_mode}")
    start_time = time.time()

    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed:        {stats['files_processed']}")
    logger.info(f"Total chunks created:   {stats['total_chunks']}")
    logger.info(f"Documents added:        {stats['documents_added']}")
    logger.info(f"Documents updated:      {stats['documents_updated']}")
    logger.info(f"Documents skipped:      {stats['documents_skipped']}")
    logger.info(f"Errors:                 {stats['errors']}")
    logger.info(f"Processing time:        {elapsed:.2f} seconds")

    logger.info("\nMission breakdown:")
    for mission, ms in stats['missions'].items():
        logger.info(f"  {mission}: {ms['files']} files, {ms['chunks']} chunks "
                    f"(added={ms['added']}, updated={ms['updated']}, skipped={ms['skipped']})")

    info = pipeline.get_collection_info()
    logger.info(f"\nCollection '{info['collection_name']}' now has {info['document_count']} total documents")

    if args.test_query:
        logger.info(f"\nTest query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query, n_results=3)
        if results and 'documents' in results:
            for i, doc in enumerate(results['documents'][0]):
                logger.info(f"Result {i + 1}: {doc[:200]}...")


if __name__ == "__main__":
    main()