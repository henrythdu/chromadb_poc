#!/usr/bin/env python3
"""Test script for metadata filtering functionality."""

import logging
import os
import sys

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.chroma_store import ChromaStore
from src.retrieval.hybrid_search import HybridSearchRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Test metadata filtering."""
    # Load environment variables
    load_dotenv()

    # Get API keys
    chroma_api_key = os.getenv("CHROMA_CLOUD_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")

    # Validate required keys
    missing = []
    if not chroma_api_key:
        missing.append("CHROMA_CLOUD_API_KEY")
    if not chroma_tenant:
        missing.append("CHROMA_TENANT")
    if not chroma_database:
        missing.append("CHROMA_DATABASE")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Initialize ChromaStore
    logger.info("Testing ChromaStore metadata filtering...")
    store = ChromaStore(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
        collection_name="arxiv_papers_v1",
    )

    # Test connection
    if not store.test_connection():
        logger.error("Failed to connect to ChromaDB Cloud")
        sys.exit(1)

    # Get all arxiv_ids to find one to test with
    logger.info("Fetching all arxiv_ids...")
    all_ids = store.get_all_arxiv_ids()

    if not all_ids:
        logger.error("No papers found in collection")
        sys.exit(1)

    logger.info(f"Found {len(all_ids)} papers in collection")
    test_arxiv_id = all_ids[0]
    logger.info(f"Testing with arxiv_id: {test_arxiv_id}")

    # Test 1: get_chunks_by_arxiv_id
    print("\n" + "=" * 60)
    print("Test 1: get_chunks_by_arxiv_id()")
    print("=" * 60)

    chunks = store.get_chunks_by_arxiv_id(test_arxiv_id)

    if chunks:
        print(f"\nRetrieved {len(chunks['ids'])} chunks")
        print(f"Chunk IDs: {chunks['ids'][:5]}...")

        if chunks.get('metadatas'):
            print(f"\nFirst chunk metadata:")
            for key, value in chunks['metadatas'][0].items():
                print(f"  {key}: {value}")

        if chunks.get('documents'):
            print(f"\nFirst chunk text preview:")
            print(f"  {chunks['documents'][0][:200]}...")
    else:
        logger.error("Failed to retrieve chunks")
        sys.exit(1)

    # Test 2: HybridSearchRetriever with metadata filter
    print("\n" + "=" * 60)
    print("Test 2: retrieve_with_filter()")
    print("=" * 60)

    retriever = HybridSearchRetriever(
        chroma_api_key=chroma_api_key,
        chroma_tenant=chroma_tenant,
        chroma_database=chroma_database,
        collection_name="arxiv_papers_v1",
        top_k=10,
    )

    # Query with metadata filter
    results = retriever.retrieve_with_filter(
        query="machine learning",
        where={"arxiv_id": test_arxiv_id},
        use_fusion=False,  # BM25 not built
    )

    print(f"\nRetrieved {len(results)} chunks with query 'machine learning' and filter")

    if results:
        for i, result in enumerate(results[:3], 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Chunk: {result['metadata'].get('chunk_index', '?')}")
            print(f"  Section: {result['metadata'].get('section', 'N/A')}")
            print(f"  Text preview: {result['text'][:150]}...")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
