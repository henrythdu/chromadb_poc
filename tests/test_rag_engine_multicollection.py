"""Tests for multi-collection RAG query engine."""

from unittest.mock import MagicMock, patch

import pytest


def test_query_with_collection_papers():
    """Test querying arxiv_papers_v1 collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
        return

    from src.retrieval.engine import RAGQueryEngine

    if chromadb is None:
        pytest.skip("chromadb not installed")

    with patch("src.retrieval.hybrid_search.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock heartbeat
        mock_instance.heartbeat.return_value = True

        # Mock collections
        mock_papers_collection = MagicMock()
        mock_papers_collection.query.return_value = {
            "documents": [["Test document"]],
            "metadatas": [[{"arxiv_id": "2301.07041"}]],
            "distances": [[0.1]]
        }
        mock_instance.get_or_create_collection.return_value = mock_papers_collection

        # Mock LLM response
        with patch("src.retrieval.engine.OpenRouterLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.answer_question.return_value = "Test answer"
            mock_llm_class.return_value = mock_llm

            # Mock reranker
            with patch("src.retrieval.engine.CohereReranker") as mock_reranker_class:
                mock_reranker = MagicMock()
                mock_reranker.rerank_results.return_value = [
                    {"text": "Test", "metadata": {"arxiv_id": "2301.07041"}}
                ]
                mock_reranker_class.return_value = mock_reranker

                engine = RAGQueryEngine(
                    chroma_api_key="test_key",
                    chroma_tenant="test_tenant",
                    chroma_database="test_db",
                )

                result = engine.query_with_collection(
                    question="test query",
                    collection_name="arxiv_papers_v1",
                    use_rerank=False,
                )

                # Verify we got a result
                assert "answer" in result
                assert result["answer"] == "Test answer"


def test_query_with_collection_contracts():
    """Test querying contracts collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
        return

    from src.retrieval.engine import RAGQueryEngine

    if chromadb is None:
        pytest.skip("chromadb not installed")

    with patch("src.retrieval.hybrid_search.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        mock_instance.heartbeat.return_value = True

        mock_contracts_collection = MagicMock()
        mock_contracts_collection.query.return_value = {
            "documents": [["Test contract clause"]],
            "metadatas": [[{"document_id": "abc123", "company_name": "Test Corp"}]],
            "distances": [[0.1]]
        }
        mock_instance.get_or_create_collection.return_value = mock_contracts_collection

        with patch("src.retrieval.engine.OpenRouterLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.answer_question.return_value = "Test contract answer"
            mock_llm_class.return_value = mock_llm

            with patch("src.retrieval.engine.CohereReranker") as mock_reranker_class:
                mock_reranker = MagicMock()
                mock_reranker.rerank_results.return_value = [
                    {"text": "Test", "metadata": {"document_id": "abc123"}}
                ]
                mock_reranker_class.return_value = mock_reranker

                engine = RAGQueryEngine(
                    chroma_api_key="test_key",
                    chroma_tenant="test_tenant",
                    chroma_database="test_db",
                )

                result = engine.query_with_collection(
                    question="test query",
                    collection_name="contracts",
                    use_rerank=False,
                )

                assert "answer" in result
                assert result["answer"] == "Test contract answer"


def test_query_with_collection_invalid_collection():
    """Test that invalid collection name raises ValueError."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
        return

    from src.retrieval.engine import RAGQueryEngine

    if chromadb is None:
        pytest.skip("chromadb not installed")

    with patch("src.retrieval.hybrid_search.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.heartbeat.return_value = True

        engine = RAGQueryEngine(
            chroma_api_key="test_key",
            chroma_tenant="test_tenant",
            chroma_database="test_db",
        )

        with pytest.raises(ValueError, match="Invalid collection"):
            engine.query_with_collection(
                question="test query",
                collection_name="invalid_collection_name",
            )
