"""Ingest first 10 ArXiv papers from ml_pdfs as a test."""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.indexer import IngestionIndexer


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
    """Ingest first 10 papers from ml_pdfs directory."""
    pdf_dir = Path("ml_pdfs")

    if not pdf_dir.exists():
        logger.error(f"Directory not found: {pdf_dir}")
        return

    # Get first 10 PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[:10]

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDFs to process")

    # Create paper list with metadata
    papers = []
    for pdf_path in pdf_files:
        metadata = get_paper_metadata(pdf_path)
        papers.append({
            "file_path": str(pdf_path),
            **metadata
        })
        logger.info(f"Added: {metadata['arxiv_id']}")

    # Initialize indexer
    logger.info("Initializing IngestionIndexer...")
    indexer = IngestionIndexer(
        llamaparse_api_key=settings.llamaparse_api_key,
        chroma_api_key=settings.chroma_cloud_api_key,
        chroma_tenant=settings.chroma_tenant,
        chroma_database=settings.chroma_database,
        collection_name="test_papers",
    )

    # Test ChromaDB connection first
    logger.info("Testing ChromaDB connection...")
    if not indexer.chroma.test_connection():
        logger.error("Failed to connect to ChromaDB Cloud")
        return
    logger.info("ChromaDB connection successful")

    # Index papers
    logger.info("Starting batch ingestion...")
    results = indexer.index_batch(papers)

    # Report results
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Success: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Total chunks indexed: {results['total_chunks']}")
    logger.info("=" * 60)

    # Verify ChromaDB count
    count = indexer.chroma.count()
    logger.info(f"Total documents in ChromaDB collection 'test_papers': {count}")


if __name__ == "__main__":
    main()
