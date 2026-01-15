"""Fetch paper metadata from ArXiv API."""

import logging
import time
from typing import Any, Dict

import arxiv

logger = logging.getLogger(__name__)


class ArxivMetadataFetcher:
    """Fetch metadata from ArXiv API for papers."""

    def __init__(self, delay_seconds: float = 3.0):
        """Initialize fetcher with rate limit delay.

        Args:
            delay_seconds: Delay between API requests (ArXiv recommends 3 seconds)
        """
        self.client = arxiv.Client()
        self.delay_seconds = delay_seconds
        self._last_request_time = 0

    def fetch_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch metadata for a single paper by arxiv_id.

        Args:
            arxiv_id: ArXiv ID (e.g., "2601.03764v1")

        Returns:
            Metadata dictionary with keys: title, authors, published_date, summary
        """
        # Rate limiting
        self._rate_limit()

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(self.client.results(search), None)

            if result is None:
                logger.warning(f"No metadata found for {arxiv_id}")
                return self._empty_metadata(arxiv_id)

            # Handle categories - newer arxiv library returns strings directly
            categories = []
            for cat in result.categories:
                if hasattr(cat, "term"):
                    categories.append(cat.term)
                else:
                    # Category is already a string
                    categories.append(str(cat))

            return {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": ", ".join(
                    [author.name for author in result.authors]
                ),  # Convert to string
                "published_date": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "summary": result.summary.replace("\n", " "),  # Remove newlines
                "categories": ", ".join(categories),  # Convert to string
                "doi": result.doi if result.doi else "",
            }

        except Exception as e:
            logger.error(f"Error fetching metadata for {arxiv_id}: {e}")
            return self._empty_metadata(arxiv_id)

    def fetch_batch(self, arxiv_ids: list[str]) -> Dict[str, Any]:
        """Fetch metadata for multiple papers.

        Args:
            arxiv_ids: List of ArXiv IDs

        Returns:
            Dict mapping arxiv_id to metadata
        """
        metadata_map = {}

        for arxiv_id in arxiv_ids:
            metadata_map[arxiv_id] = self.fetch_metadata(arxiv_id)

        return metadata_map

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()

    def _empty_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Return empty metadata fallback."""
        return {
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "authors": "",  # Empty string instead of list
            "published_date": "",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "summary": "",
            "categories": "",  # Empty string instead of list
            "doi": "",
        }
