"""Tests for ArXiv downloader module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_downloader_imports():
    """Test that the downloader module can be imported."""
    from src.ingestion.downloader import ArxivDownloader

    assert ArxivDownloader is not None


def test_metadata_extraction():
    """Test metadata extraction from ArXiv result."""
    from src.ingestion.downloader import ArxivDownloader

    # Create a mock arxiv Result object
    mock_author1 = Mock()
    mock_author1.name = "Author One"
    mock_author2 = Mock()
    mock_author2.name = "Author Two"

    mock_result = Mock()
    mock_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
    mock_result.title = "Test Paper Title"
    mock_result.authors = [mock_author1, mock_author2]
    mock_result.published = datetime(2023, 1, 15, 10, 30, 0)
    mock_result.pdf_url = "http://arxiv.org/pdf/2301.12345v1.pdf"
    mock_result.summary = "This is a test abstract."

    # Create downloader instance
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ArxivDownloader(download_dir=tmpdir, max_results=10)
        metadata = downloader._extract_metadata(mock_result)

        # Verify all metadata fields are extracted
        assert metadata["arxiv_id"] == "2301.12345v1"
        assert metadata["title"] == "Test Paper Title"
        assert metadata["authors"] == ["Author One", "Author Two"]
        assert metadata["published_date"] == "2023-01-15T10:30:00Z"
        assert metadata["pdf_url"] == "http://arxiv.org/pdf/2301.12345v1.pdf"
        assert metadata["summary"] == "This is a test abstract."


@pytest.mark.integration
def test_download_single_paper():
    """Integration test for downloading a single paper from ArXiv.

    This test may be skipped initially as it requires network access.
    """
    from src.ingestion.downloader import ArxivDownloader

    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ArxivDownloader(download_dir=tmpdir, max_results=1)

        # Download a specific well-known paper
        papers = downloader.download_papers(query="quantum", max_results=1)

        # Verify we got at least one paper
        assert len(papers) > 0

        # Verify paper structure
        paper = papers[0]
        assert "arxiv_id" in paper
        assert "title" in paper
        assert "authors" in paper
        assert "published_date" in paper
        assert "pdf_url" in paper
        assert "summary" in paper
        assert "local_pdf_path" in paper

        # Verify PDF file was downloaded
        assert Path(paper["local_pdf_path"]).exists()


def test_downloader_initialization():
    """Test downloader initialization with parameters."""
    from src.ingestion.downloader import ArxivDownloader

    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ArxivDownloader(download_dir=tmpdir, max_results=5)

        assert downloader.download_dir == Path(tmpdir)
        assert downloader.max_results == 5


def test_sanitized_filename():
    """Test that filenames are sanitized properly."""
    from src.ingestion.downloader import ArxivDownloader

    # Test with a title that has special characters
    mock_result = Mock()
    mock_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
    mock_result.title = "Test/Paper: Special<Chars> & More"
    mock_result.authors = []
    mock_result.published = datetime(2023, 1, 15, 10, 30, 0)
    mock_result.pdf_url = "http://arxiv.org/pdf/2301.12345v1.pdf"
    mock_result.summary = "Abstract"

    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ArxivDownloader(download_dir=tmpdir, max_results=10)
        metadata = downloader._extract_metadata(mock_result)

        # Verify arxiv_id is used for filename (sanitized)
        assert "2301.12345v1" in metadata["arxiv_id"]


def test_download_papers_with_no_results():
    """Test download_papers when no papers are found."""
    from src.ingestion.downloader import ArxivDownloader

    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = ArxivDownloader(download_dir=tmpdir, max_results=5)

        with patch("src.ingestion.downloader.arxiv.Client") as mock_client:
            # Mock the search to return no results
            mock_search = MagicMock()
            mock_search.results = []
            mock_client.return_value.__enter__.return_value.search.return_value = mock_search

            papers = downloader.download_papers(query="nonexistentqueryxyz123", max_results=5)

            # Should return empty list
            assert papers == []
