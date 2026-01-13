#!/usr/bin/env python3
"""Batch process downloaded papers with LlamaParse.

This script processes all downloaded PDFs from the ArXiv download,
converts them to Markdown using LlamaParse, and saves the results.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.parser import LlamaParserWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/parse_papers.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path) -> list[dict]:
    """Load paper metadata from JSON file.

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        List of paper metadata dictionaries
    """
    with open(metadata_path) as f:
        return json.load(f)


def save_progress(progress_path: Path, parsed_count: int, total_count: int):
    """Save parsing progress to file.

    Args:
        progress_path: Path to save progress file
        parsed_count: Number of papers successfully parsed
        total_count: Total number of papers to parse
    """
    with open(progress_path, "w") as f:
        json.dump(
            {
                "parsed_count": parsed_count,
                "total_count": total_count,
                "progress_percent": round((parsed_count / total_count) * 100, 2),
            },
            f,
            indent=2,
        )


def main():
    """Main entry point for batch processing."""
    # Paths
    project_root = Path(__file__).parent.parent
    metadata_file = project_root / "data" / "downloaded_papers_500.json"
    pdf_dir = project_root / "ml_pdfs"
    output_dir = project_root / "data" / "processed"
    progress_file = project_root / "data" / "parse_progress.json"
    log_dir = project_root / "logs"

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting batch PDF processing with LlamaParse")
    logger.info("=" * 80)

    # Load metadata
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        sys.exit(1)

    papers = load_metadata(metadata_file)
    logger.info(f"Loaded metadata for {len(papers)} papers")

    # Initialize parser
    parser = LlamaParserWrapper(api_key=settings.llamaparse_api_key, result_type="markdown")
    logger.info("LlamaParse parser initialized")

    # Process each paper
    success_count = 0
    error_count = 0
    skip_count = 0

    for idx, paper in enumerate(papers, 1):
        arxiv_id = paper.get("arxiv_id", "")
        pdf_filename = f"{arxiv_id}.pdf"
        pdf_path = pdf_dir / pdf_filename
        output_path = output_dir / f"{arxiv_id}.md"

        logger.info(f"[{idx}/{len(papers)}] Processing {arxiv_id}")

        # Skip if already processed
        if output_path.exists():
            logger.info(f"  ✓ Already processed, skipping")
            skip_count += 1
            continue

        # Check if PDF exists
        if not pdf_path.exists():
            logger.warning(f"  ✗ PDF not found: {pdf_path}")
            error_count += 1
            continue

        # Parse the PDF
        result = parser.parse_pdf(pdf_path, max_retries=3)

        if result["error"]:
            logger.error(f"  ✗ Failed to parse: {result['error']}")
            error_count += 1
            # Save error info
            error_path = output_dir / f"{arxiv_id}.error"
            with open(error_path, "w") as f:
                json.dump(
                    {
                        "arxiv_id": arxiv_id,
                        "error": result["error"],
                        "pdf_path": str(pdf_path),
                    },
                    f,
                )
            continue

        # Save parsed markdown
        with open(output_path, "w") as f:
            f.write(result["markdown"])

        logger.info(
            f"  ✓ Parsed successfully: {result['pages']} pages, "
            f"{len(result['markdown'])} chars"
        )
        success_count += 1

        # Save progress every 10 papers
        if idx % 10 == 0:
            save_progress(progress_file, success_count, len(papers))
            logger.info(f"Progress saved: {idx}/{len(papers)} papers processed")

    # Final progress save
    save_progress(progress_file, success_count, len(papers))

    # Summary
    logger.info("=" * 80)
    logger.info("Batch processing complete!")
    logger.info(f"  Total papers:      {len(papers)}")
    logger.info(f"  Successfully parsed: {success_count}")
    logger.info(f"  Skipped (existing): {skip_count}")
    logger.info(f"  Errors:            {error_count}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info("=" * 80)

    # Save summary
    summary_path = output_dir.parent / "parse_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_papers": len(papers),
                "success_count": success_count,
                "skip_count": skip_count,
                "error_count": error_count,
                "output_directory": str(output_dir),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
