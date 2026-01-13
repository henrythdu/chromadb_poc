"""Test parallel ingestion functionality."""
from unittest.mock import MagicMock, patch

from .test_indexer import test_indexer_imports  # noqa: F401


@patch("src.ingestion.chroma_store.chromadb.CloudClient")
@patch("src.ingestion.parser.LlamaParserWrapper")
def test_parallel_ingestion_handles_errors(mock_parser_class, mock_cloud_client):
    """Test that parallel ingestion continues even when some papers fail."""
    from src.ingestion.indexer import IngestionIndexer

    # Mock ChromaDB client
    mock_client_instance = MagicMock()
    mock_cloud_client.return_value = mock_client_instance

    # Create indexer
    indexer = IngestionIndexer(
        llamaparse_api_key="test_key",
        chroma_api_key="test_chroma_key",
        chroma_tenant="test-tenant",
        chroma_database="test-db",
    )

    # Create mock papers (some with file_path, some without)
    papers = [
        {
            "arxiv_id": "2601.0001",
            "file_path": "/fake/path1.pdf",
        },
        {
            "arxiv_id": "2601.0002",
            # No file_path - should be skipped
        },
        {
            "arxiv_id": "2601.0003",
            "file_path": "/fake/path3.pdf",
        },
    ]

    # Mock the index_paper method to simulate success/failure
    call_count = [0]

    def mock_index_paper(pdf_path, metadata):
        call_count[0] += 1
        arxiv_id = metadata["arxiv_id"]

        # Simulate one paper failing
        if arxiv_id == "2601.0001":
            return {
                "status": "error",
                "error": "Parse error",
                "chunks_indexed": 0,
                "arxiv_id": arxiv_id,
            }

        return {
            "status": "success",
            "chunks_indexed": 10,
            "arxiv_id": arxiv_id,
        }

    indexer.index_paper = mock_index_paper

    # Run parallel ingestion
    results = indexer.index_batch_parallel(papers, max_workers=2)

    # Verify results
    assert results["success"] == 1  # Only 2601.0003 succeeded
    assert results["failed"] == 2  # 2601.0001 failed + 2601.0002 had no file_path
    assert results["total_chunks"] == 10


@patch("src.ingestion.chroma_store.chromadb.CloudClient")
def test_parallel_imports(mock_cloud_client):
    """Test that index_batch_parallel method exists."""
    from src.ingestion.indexer import IngestionIndexer

    mock_client_instance = MagicMock()
    mock_cloud_client.return_value = mock_client_instance

    indexer = IngestionIndexer(
        llamaparse_api_key="test_key",
        chroma_api_key="test_chroma_key",
        chroma_tenant="test-tenant",
        chroma_database="test-db",
    )

    # Verify method exists
    assert hasattr(indexer, 'index_batch_parallel')
    assert callable(indexer.index_batch_parallel)
