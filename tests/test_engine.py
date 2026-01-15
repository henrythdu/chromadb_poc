"""Test query engine orchestration."""

from unittest.mock import patch


def test_engine_imports():
    """Test that engine module can be imported."""
    from src.retrieval.engine import RAGQueryEngine

    assert RAGQueryEngine is not None


@patch("src.retrieval.engine.HybridSearchRetriever")
@patch("src.retrieval.engine.CohereReranker")
@patch("src.retrieval.engine.OpenRouterLLM")
def test_engine_initialization(mock_llm, mock_reranker, mock_retriever):
    """Test RAGQueryEngine initialization."""
    from src.retrieval.engine import RAGQueryEngine

    engine = RAGQueryEngine(
        chroma_api_key="test_key",
        chroma_tenant="test-tenant",
        chroma_database="test-db",
        cohere_api_key="test_key",
        openrouter_api_key="test_key",
    )

    assert engine is not None
    assert engine.top_k_retrieve == 50
    assert engine.top_k_rerank == 5
