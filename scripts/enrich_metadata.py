"""Enrich metadata for already-indexed papers using ArXiv API.

This script fetches rich metadata (title, authors, abstract, categories, etc.)
from the ArXiv API and updates existing documents in ChromaDB without re-parsing
or re-embedding the PDFs.

Usage:
    python scripts/enrich_metadata.py
"""
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path before importing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.arxiv_metadata import ArxivMetadataFetcher
from src.ingestion.chroma_store import ChromaStore


def main():
    """Enrich metadata for all papers in ChromaDB."""
    logger.info("=" * 60)
    logger.info("STARTING METADATA ENRICHMENT")
    logger.info("=" * 60)

    # Initialize ChromaDB connection
    logger.info("Connecting to ChromaDB Cloud...")
    chroma = ChromaStore(
        api_key=settings.chroma_cloud_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
        collection_name="arxiv_papers_v1",
    )

    # Test connection
    if not chroma.test_connection():
        logger.error("Failed to connect to ChromaDB Cloud")
        return

    # Get all arxiv_ids
    logger.info("Fetching all arxiv_ids from ChromaDB...")
    arxiv_ids = chroma.get_all_arxiv_ids()
    total_papers = len(arxiv_ids)
    logger.info(f"Found {total_papers} unique papers to enrich")

    if total_papers == 0:
        logger.warning("No papers found in collection. Nothing to enrich.")
        return

    # Initialize metadata fetcher (3 second delay per ArXiv API request)
    fetcher = ArxivMetadataFetcher(delay_seconds=3.0)

    # Enrich each paper
    success_count = 0
    failed_count = 0
    start_time = time.time()

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        logger.info(f"[{i}/{total_papers}] Fetching metadata for {arxiv_id}...")

        # Fetch metadata from ArXiv API
        new_metadata = fetcher.fetch_metadata(arxiv_id)

        # Check if API returned actual data
        if new_metadata.get("title") == f"Paper {arxiv_id}":
            logger.warning(f"  ✗ No API data found for {arxiv_id}")
            failed_count += 1
            continue

        # Update all chunks for this paper
        updated = chroma.update_paper_metadata(arxiv_id, new_metadata)

        if updated > 0:
            success_count += 1
            logger.info(
                f"  ✓ {arxiv_id}: {updated} chunks updated | "
                f"title: '{new_metadata.get('title', 'N/A')[:50]}...' | "
                f"authors: {len(new_metadata.get('authors', []))}"
            )
        else:
            failed_count += 1
            logger.error(f"  ✗ Failed to update {arxiv_id}")

    elapsed_time = time.time() - start_time

    # Report results
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total papers: {total_papers}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    if success_count > 0:
        logger.info(f"Avg time per paper: {elapsed_time/success_count:.1f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
