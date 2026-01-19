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

        # Verify store attributes (credentials NOT stored for security)
        assert store.collection_name == "test_collection"
        assert store.client == mock_instance
        # Note: api_key, tenant, database are deliberately NOT stored as instance attributes


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

        # Verify upsert was called (changed from add to upsert for resumable ingestion)
        mock_collection.upsert.assert_called_once_with(
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


def test_get_collection_dynamic():
    """Test getting a collection by name dynamically."""
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

        # Mock two different collections
        mock_arxiv_collection = MagicMock()
        mock_contracts_collection = MagicMock()

        def mock_get_or_create(name):
            if name == "arxiv_papers":
                return mock_arxiv_collection
            elif name == "contracts":
                return mock_contracts_collection
            return MagicMock()

        mock_instance.get_or_create_collection = mock_get_or_create

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="arxiv_papers",
        )

        # Get arxiv_papers collection (default)
        arxiv_coll = store.get_collection("arxiv_papers")
        assert arxiv_coll == mock_arxiv_collection

        # Get contracts collection (dynamic)
        contracts_coll = store.get_collection("contracts")
        assert contracts_coll == mock_contracts_collection


def test_get_collection_default():
    """Test that get_collection with no args returns default collection."""
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

        mock_default_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_default_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="default_collection",
        )

        # Get default collection without argument
        default_coll = store.get_collection()
        assert default_coll == mock_default_collection
        assert default_coll == store._get_or_create_collection()  # Same as default


def test_get_collection_invalid_input_raises_error():
    """Test that get_collection raises ValueError for invalid names."""
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

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="default_collection",
        )

        # Test empty string raises ValueError
        with pytest.raises(ValueError, match="Collection name must be a non-empty string"):
            store.get_collection("")

        # Test non-string type raises ValueError
        with pytest.raises(ValueError, match="Collection name must be a non-empty string"):
            store.get_collection(123)


def test_add_documents_to_specific_collection():
    """Test adding documents to a specific collection by name."""
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

        # Mock collections
        mock_default_collection = MagicMock()
        mock_contracts_collection = MagicMock()

        collections_called = []

        def mock_get_or_create(name):
            collections_called.append(name)
            if name == "papers":
                return mock_default_collection
            elif name == "contracts":
                return mock_contracts_collection
            return MagicMock()

        mock_instance.get_or_create_collection = mock_get_or_create

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="papers",
        )

        # Add to default collection (no collection_name specified)
        store.add_documents(
            documents=["doc1"],
            metadatas=[{"key": "value1"}],
            ids=["id1"]
        )

        # Add to contracts collection (specific collection)
        store.add_documents(
            documents=["doc2"],
            metadatas=[{"key": "value2"}],
            ids=["id2"],
            collection_name="contracts"
        )

        # Verify both collections were accessed
        assert "papers" in collections_called
        assert "contracts" in collections_called

        # Verify upsert was called on both collections
        assert mock_default_collection.upsert.called
        assert mock_contracts_collection.upsert.called


def test_add_documents_backward_compatible():
    """Test that existing add_documents usage (without collection_name) still works."""
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

        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="papers",
        )

        # Call without collection_name parameter (existing usage)
        store.add_documents(
            documents=["doc1"],
            metadatas=[{"key": "value"}],
            ids=["id1"]
        )

        # Verify upsert was called
        mock_collection.upsert.assert_called_once_with(
            documents=["doc1"],
            metadatas=[{"key": "value"}],
            ids=["id1"]
        )
