"""ArXiv paper downloader module."""

import logging
from pathlib import Path
from typing import Any

import arxiv

logger = logging.getLogger(__name__)


class ArxivDownloader:
    """Downloader for ArXiv research papers.

    This class handles downloading papers from ArXiv, extracting metadata,
    and saving PDFs to a local directory.
    """

    def __init__(self, download_dir: str | Path, max_results: int = 10) -> None:
        """Initialize the ArXiv downloader.

        Args:
            download_dir: Directory to save downloaded PDFs
            max_results: Maximum number of papers to download per query
        """
        self.download_dir = Path(download_dir)
        self.max_results = max_results
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ArxivDownloader with dir={self.download_dir}, max_results={max_results}")

    def download_papers(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Download papers from ArXiv based on a search query.

        Args:
            query: Search query for ArXiv
            max_results: Maximum number of papers to download (overrides instance default)

        Returns:
            List of paper dictionaries containing metadata and local PDF paths
        """
        if max_results is None:
            max_results = self.max_results

        logger.info(f"Searching ArXiv for query='{query}', max_results={max_results}")

        papers = []
        failed_count = 0
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        client = arxiv.Client()
        for result in client.results(search):
            try:
                logger.debug(f"Processing paper: {result.title}")

                # Extract metadata
                paper = self._extract_metadata(result)

                # Download PDF
                pdf_path = self._download_pdf(result, paper)
                paper["local_pdf_path"] = str(pdf_path)

                papers.append(paper)
                logger.info(f"Downloaded paper {paper['arxiv_id']}: {paper['title']}")

            except Exception as e:
                # Get arxiv_id for error reporting
                arxiv_id = result.get_short_id()
                logger.error(f"Failed to process paper {arxiv_id}: {e}")
                failed_count += 1
                # Continue with next paper
                continue

        logger.info(f"Successfully downloaded {len(papers)} papers, {failed_count} failed")
        return papers

    def _extract_metadata(self, result: arxiv.Result) -> dict[str, Any]:
        """Extract metadata from an ArXiv result object.

        Args:
            result: arxiv.Result object containing paper information

        Returns:
            Dictionary containing extracted metadata fields
        """
        # Use the official library method for robustness
        arxiv_id = result.get_short_id()

        # Extract authors
        authors = [author.name for author in result.authors]

        metadata = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": authors,
            "published_date": result.published.strftime("%Y-%m-%dT%H:%M:%SZ") if result.published else "",
            "pdf_url": result.pdf_url,
            "summary": result.summary,
        }

        logger.debug(f"Extracted metadata for {arxiv_id}: {metadata['title']}")
        return metadata

    def _download_pdf(self, result: arxiv.Result, paper: dict[str, Any]) -> Path:
        """Download PDF from ArXiv to local directory.

        Args:
            result: arxiv.Result object
            paper: Paper metadata dictionary

        Returns:
            Path to the downloaded PDF file
        """
        # Use arxiv_id as filename to avoid special character issues
        filename = f"{paper['arxiv_id']}.pdf"
        pdf_path = self.download_dir / filename

        logger.debug(f"Downloading PDF to {pdf_path}")

        try:
            # Download the PDF
            result.download_pdf(dirpath=str(self.download_dir), filename=filename)
            logger.info(f"Successfully downloaded PDF to {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to download PDF for {paper['arxiv_id']}: {e}", exc_info=True)
            raise

        return pdf_path
