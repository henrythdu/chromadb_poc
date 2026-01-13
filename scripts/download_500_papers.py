#!/usr/bin/env python3
"""Download 500 ArXiv research papers from cs.LG category.

This script downloads papers from ArXiv and saves them along with metadata.
"""

import json
import logging
import sys
from pathlib import Path

# Add src directory to path AND ensure we can import from src
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ingestion.downloader import ArxivDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/download_500_papers.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main function to download 500 ArXiv papers."""
    logger.info("=" * 80)
    logger.info("Starting download of 500 ArXiv papers from cs.LG category")
    logger.info("=" * 80)

    # Configuration
    # Note: This script is specifically for downloading 500 papers for the POC.
    # The config.toml max_papers (200) is for the general pipeline.
    max_papers = 500
    query = "cat:cs.LG"  # Machine Learning category
    download_dir = Path("./ml_pdfs")
    metadata_file = Path("data/downloaded_papers_500.json")

    # Initialize downloader
    logger.info(f"Initializing downloader with max_results={max_papers}")
    logger.info(f"Download directory: {download_dir.absolute()}")
    logger.info(f"Metadata file: {metadata_file.absolute()}")

    downloader = ArxivDownloader(download_dir=download_dir, max_results=max_papers)

    # Download papers
    try:
        logger.info(f"Searching and downloading papers with query: {query}")
        papers = downloader.download_papers(query=query, max_results=max_papers)

        # Save metadata to JSON
        logger.info(f"Saving metadata for {len(papers)} papers to {metadata_file}")
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total papers downloaded: {len(papers)}")
        logger.info(f"Download directory: {download_dir.absolute()}")
        logger.info(f"Metadata file: {metadata_file.absolute()}")

        # Count successful downloads
        successful = sum(1 for p in papers if p.get("local_pdf_path"))
        logger.info(f"Successful downloads: {successful}/{len(papers)}")

        if successful < len(papers):
            logger.warning(f"Some downloads may have failed: {len(papers) - successful} papers")

    except Exception as e:
        logger.error(f"Fatal error during download: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
