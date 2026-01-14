"""Ingest ArXiv papers from ml_pdfs using parallel processing."""
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

from src.config import settings  # noqa: E402
from src.ingestion.indexer import IngestionIndexer  # noqa: E402


def get_paper_metadata(pdf_path: Path) -> dict:
    """Extract arxiv_id from filename and create minimal metadata.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Metadata dictionary with arxiv_id
    """
    # Filename format: 2601.03764v1.pdf -> arxiv_id: 2601.03764v1
    arxiv_id = pdf_path.stem  # removes .pdf

    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": [],  # Not available from filename alone
        "published_date": "",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "summary": "",
    }


def main():
    """Ingest papers from ml_pdfs directory using parallel processing."""
    pdf_dir = Path("ml_pdfs")

    if not pdf_dir.exists():
        logger.error(f"Directory not found: {pdf_dir}")
        return

    # Get all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return

    total_papers = len(pdf_files)
    logger.info(f"Found {total_papers} PDFs to process")

    # Create paper list with metadata
    papers = []
    for pdf_path in pdf_files:
        metadata = get_paper_metadata(pdf_path)
        papers.append({
            "file_path": str(pdf_path),
            **metadata
        })

    # Initialize indexer (using Docling - free, local parser)
    logger.info("Initializing IngestionIndexer with Docling...")
    indexer = IngestionIndexer(
        chroma_api_key=settings.chroma_cloud_api_key,
        chroma_tenant=settings.chroma_tenant,
        chroma_database=settings.chroma_database,
        collection_name="arxiv_papers_v1",
    )

    # Test ChromaDB connection first
    logger.info("Testing ChromaDB connection...")
    if not indexer.chroma.test_connection():
        logger.error("Failed to connect to ChromaDB Cloud")
        return
    logger.info("ChromaDB connection successful")

    # Index papers in parallel
    # Using 5 workers to balance speed and API rate limits
    max_workers = 5
    logger.info(f"Starting PARALLEL ingestion with {max_workers} workers...")
    logger.info("=" * 60)

    start_time = time.time()

    results = indexer.index_batch_parallel(papers, max_workers=max_workers)

    elapsed_time = time.time() - start_time

    # Report results
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Papers processed: {total_papers}")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Total chunks indexed: {results['total_chunks']}")
    logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    if results['success'] > 0:
        logger.info(f"Avg time per paper: {elapsed_time/results['success']:.1f} seconds")
    logger.info("=" * 60)

    # Verify ChromaDB count
    count = indexer.chroma.count()
    logger.info(f"Total documents in ChromaDB collection 'arxiv_papers_v1': {count}")


if __name__ == "__main__":
    main()
