"""Tests for Chroma Store module."""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_chroma_store_imports():
    """Test that the chroma_store module can be imported."""
    from src.ingestion.chroma_store import ChromaStore

    assert ChromaStore is not None


def test_chroma_initialization():
    """Test ChromaStore initializes correctly with parameters."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    # Skip if chromadb is not installed
    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    # Mock the Chroma client to avoid actual connection
    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Create store with test parameters
        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="test_collection",
        )

        # Verify client was initialized with correct parameters
        mock_client.assert_called_once_with(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
        )

        # Verify store attributes
        assert store.collection_name == "test_collection"
        assert store.api_key == "test_key"
        assert store.tenant == "test-tenant"
        assert store.database == "test_database"
        assert store.client == mock_instance


def test_get_or_create_collection():
    """Test get-or-create pattern for collections."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock collection
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="test_collection",
        )

        # Get collection
        collection = store._get_or_create_collection()

        # Verify collection was retrieved
        mock_instance.get_or_create_collection.assert_called_once_with(
            name="test_collection"
        )
        assert collection == mock_collection


def test_add_documents():
    """Test adding documents to the collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock collection
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="test_collection",
        )

        # Test data
        documents = ["doc1", "doc2", "doc3"]
        metadatas = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        ids = ["id1", "id2", "id3"]

        # Add documents
        store.add_documents(documents=documents, metadatas=metadatas, ids=ids)

        # Verify add was called
        mock_collection.add.assert_called_once_with(
            documents=documents, metadatas=metadatas, ids=ids
        )


def test_count():
    """Test getting document count from collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_instance.get_or_create_collection.return_value = mock_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="test_collection",
        )

        # Get count
        count = store.count()

        # Verify count was retrieved
        mock_collection.count.assert_called_once()
        assert count == 42


def test_connection():
    """Test connection method."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock heartbeat to return True
        mock_instance.heartbeat.return_value = True

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="test_collection",
        )

        # Test connection
        result = store.test_connection()

        # Verify heartbeat was called
        mock_instance.heartbeat.assert_called_once()
        assert result is True


@pytest.mark.integration
def test_chroma_connection(mock_api_keys):
    """Integration test for real Chroma Cloud connection.

    This test requires valid CHROMA_CLOUD_API_KEY, CHROMA_TENANT,
    and CHROMA_DATABASE environment variables.
    Marked as integration test - can be skipped with: pytest -m "not integration"
    """
    from src.ingestion.chroma_store import ChromaStore

    # Skip if no real credentials (mock_api_keys is just a placeholder)
    api_key = os.getenv("CHROMA_CLOUD_API_KEY", "")
    if not api_key or "test" in api_key.lower():
        pytest.skip("Skipping integration test - no real credentials")

    # Create store with real credentials from environment
    store = ChromaStore(
        collection_name="test_collection",
    )

    # Test connection
    assert store.test_connection() is True

    # Test collection operations
    collection = store._get_or_create_collection()
    assert collection is not None

    # Test adding documents
    documents = ["Test document 1", "Test document 2"]
    metadatas = [{"test": "data1"}, {"test": "data2"}]
    ids = ["test_id_1", "test_id_2"]

    store.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    # Verify count (should be at least 2)
    count = store.count()
    assert count >= 2
