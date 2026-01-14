"""Test hybrid search implementation."""
from unittest.mock import MagicMock, patch

import pytest


def test_hybrid_search_imports():
    """Test that hybrid_search module can be imported."""
    from src.retrieval.hybrid_search import HybridSearchRetriever
    assert HybridSearchRetriever is not None


@patch("chromadb.CloudClient")
def test_hybrid_search_initialization(mock_cloud_client):
    """Test HybridSearchRetriever initialization."""
    from src.retrieval.hybrid_search import HybridSearchRetriever

    # Mock the CloudClient
    mock_client_instance = MagicMock()
    mock_cloud_client.return_value = mock_client_instance

    # Mock collection
    mock_collection = MagicMock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection

    retriever = HybridSearchRetriever(
        chroma_api_key="test_key",
        chroma_tenant="test-tenant",
        chroma_database="test-db",
        top_k=50,
    )

    assert retriever.top_k == 50
    assert retriever.collection_name == "arxiv_papers_v1"


@pytest.mark.integration
def test_hybrid_search_retrieves():
    """Test that hybrid search retrieves relevant chunks."""
    from src.config import settings
    from src.retrieval.hybrid_search import HybridSearchRetriever

    retriever = HybridSearchRetriever(
        chroma_api_key=settings.chroma_cloud_api_key,
        chroma_tenant=settings.chroma_tenant,
        chroma_database=settings.chroma_database,
        collection_name="test_papers",
        top_k=50,
    )

    results = retriever.retrieve("What is attention in transformers?")

    assert len(results) <= 50
    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]
