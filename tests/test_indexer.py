"""Test ingestion indexer pipeline."""
from unittest.mock import MagicMock, patch


def test_indexer_imports():
    """Test that indexer can be imported."""
    from src.ingestion.indexer import IngestionIndexer
    assert IngestionIndexer is not None


@patch("src.ingestion.chroma_store.chromadb.CloudClient")
def test_indexer_initialization(mock_cloud_client):
    """Test IngestionIndexer initialization."""
    from src.ingestion.indexer import IngestionIndexer

    # Mock the ChromaDB client to avoid actual connection
    mock_client_instance = MagicMock()
    mock_cloud_client.return_value = mock_client_instance

    indexer = IngestionIndexer(
        llamaparse_api_key="test_key",
        chroma_api_key="test_chroma_key",
        chroma_tenant="test-tenant",
        chroma_database="test-db",
    )

    assert indexer is not None
    assert indexer.chunker.chunk_size == 800
    assert indexer.chunker.chunk_overlap == 100
