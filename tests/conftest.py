"""Pytest configuration and shared fixtures."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Provide mock API keys for testing."""
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_llamaparse_key")
    monkeypatch.setenv("CHROMA_HOST", "http://test-chroma.com")
    monkeypatch.setenv("CHROMA_API_KEY", "test_chroma_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    return {
        "llamaparse": "test_llamaparse_key",
        "chroma": {"host": "http://test-chroma.com", "key": "test_chroma_key"},
        "openrouter": "test_openrouter_key",
        "cohere": "test_cohere_key",
    }


@pytest.fixture
def sample_arxiv_metadata():
    """Sample ArXiv paper metadata for testing."""
    return {
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "published_date": "2023-01-17",
        "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
    }


@pytest.fixture
def sample_chunk():
    """Sample document chunk for testing."""
    return {
        "document_id": "2301.07041",
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "page_number": 1,
        "section": "Introduction",
        "chunk_index": 0,
        "text": "The dominant sequence transduction models...",
    }
