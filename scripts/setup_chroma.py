#!/usr/bin/env python3
"""Setup ChromaDB Cloud connection and create initial collection.

This script tests the ChromaDB Cloud connection, creates the collection
for ArXiv papers, and verifies everything works.
"""

import logging
import sys

from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.chroma_store import ChromaStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/setup_chroma.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for ChromaDB setup."""
    logger.info("=" * 80)
    logger.info("ChromaDB Cloud Setup")
    logger.info("=" * 80)

    # Display configuration (without sensitive data)
    logger.info(f"Tenant: {settings.chroma_tenant}")
    logger.info(f"Database: {settings.chroma_database}")
    logger.info(f"API Key: {'*' * 20}{settings.chroma_cloud_api_key[-8:]}")

    collection_name = "arxiv_papers_v1"

    try:
        # Initialize ChromaStore
        logger.info(f"\nInitializing ChromaDB client...")
        store = ChromaStore(
            api_key=settings.chroma_cloud_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            collection_name=collection_name,
        )
        logger.info("✓ ChromaDB client initialized successfully")

        # Test connection
        logger.info(f"\nTesting connection...")
        if store.test_connection():
            logger.info("✓ Connection successful")
        else:
            logger.error("✗ Connection failed")
            sys.exit(1)

        # Get or create collection
        logger.info(f"\nAccessing collection '{collection_name}'...")
        collection = store._get_or_create_collection()
        logger.info(f"✓ Collection '{collection_name}' is ready")

        # Check collection count
        count = store.count()
        logger.info(f"✓ Collection contains {count} documents")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Setup Complete!")
        logger.info("=" * 80)
        logger.info(f"  Tenant:    {settings.chroma_tenant}")
        logger.info(f"  Database:  {settings.chroma_database}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Documents: {count}")
        logger.info("=" * 80)
        logger.info("\nChromaDB is ready for ingestion!")
        logger.info("You can now run the parse_papers.py script to process PDFs.")

    except Exception as e:
        logger.error(f"\n✗ Setup failed: {e}")
        logger.error("Please check:")
        logger.error("  1. Your .env file has correct ChromaDB credentials")
        logger.error("  2. ChromaDB Cloud is accessible")
        logger.error("  3. Your tenant/database exists in ChromaDB Cloud")
        sys.exit(1)


if __name__ == "__main__":
    main()
