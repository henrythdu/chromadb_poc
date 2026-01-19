"""Tests for LRU-cached RAG engine."""

import time

from src.retrieval.cached_engine import CachedRAGEngine


def test_cache_hit_returns_same_result(monkeypatch):
    """Test that cache returns the same result for identical queries."""
    # Mock required environment variables
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    # Create cached engine with small cache
    engine = CachedRAGEngine(max_cache_size=10, cache_ttl_seconds=60)

    # Mock the base engine to return predictable results
    call_count = [0]

    def mock_query(question, use_rerank=True):
        call_count[0] += 1
        return {
            "answer": f"Answer to: {question}",
            "citations": ["Test citation"],
            "sources": [],
        }

    engine.base_engine.query = mock_query

    # First call - cache miss
    result1 = engine.query("What is ML?")

    # Second call - cache hit
    result2 = engine.query("What is ML?")

    assert result1 == result2
    assert call_count[0] == 1, "Base engine should only be called once"

    # Different query - cache miss
    engine.query("What is AI?")

    assert call_count[0] == 2, "Base engine should be called for different query"


def test_cache_expiry(monkeypatch):
    """Test that cache entries expire after TTL."""
    # Mock required environment variables
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    # Create cached engine with short TTL
    engine = CachedRAGEngine(max_cache_size=10, cache_ttl_seconds=1)

    call_count = [0]

    def mock_query(question, use_rerank=True):
        call_count[0] += 1
        return {"answer": f"Answer {call_count[0]}", "citations": [], "sources": []}

    engine.base_engine.query = mock_query

    # First call
    result1 = engine.query("Test question")
    assert call_count[0] == 1
    assert result1["answer"] == "Answer 1"

    # Second call immediately - cache hit
    result2 = engine.query("Test question")
    assert call_count[0] == 1, "Should use cache"
    assert result2["answer"] == "Answer 1"

    # Wait for cache to expire
    time.sleep(1.1)

    # Third call after expiry - cache miss
    result3 = engine.query("Test question")
    assert call_count[0] == 2, "Should call base engine again"
    assert result3["answer"] == "Answer 2"


def test_cache_lru_eviction(monkeypatch):
    """Test that LRU eviction works when cache is full."""
    # Mock required environment variables
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    # Create cached engine with tiny cache
    engine = CachedRAGEngine(max_cache_size=2, cache_ttl_seconds=60)

    call_count = [0]

    def mock_query(question, use_rerank=True):
        call_count[0] += 1
        return {"answer": f"Answer to: {question}", "citations": [], "sources": []}

    engine.base_engine.query = mock_query

    # Fill cache
    engine.query("Query 1")
    engine.query("Query 2")
    assert call_count[0] == 2

    # Add third item - should evict oldest (Query 1)
    engine.query("Query 3")
    assert call_count[0] == 3

    # Query 1 should be cache miss now
    engine.query("Query 1")
    assert call_count[0] == 4, "Query 1 should have been evicted"


def test_cache_stats(monkeypatch):
    """Test that cache statistics are tracked correctly."""
    # Mock required environment variables
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    engine = CachedRAGEngine(max_cache_size=10, cache_ttl_seconds=60)

    def mock_query(question, use_rerank=True):
        return {"answer": "Answer", "citations": [], "sources": []}

    engine.base_engine.query = mock_query

    # Initial stats
    stats = engine.get_cache_stats()
    assert stats["cache_size"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0

    # Cache miss
    engine.query("Test")
    stats = engine.get_cache_stats()
    assert stats["cache_size"] == 1
    assert stats["cache_misses"] == 1

    # Cache hit
    engine.query("Test")
    stats = engine.get_cache_stats()
    assert stats["cache_hits"] == 1


def test_clear_cache(monkeypatch):
    """Test that clear_cache removes all entries."""
    # Mock required environment variables
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    engine = CachedRAGEngine(max_cache_size=10, cache_ttl_seconds=60)

    def mock_query(question, use_rerank=True):
        return {"answer": "Answer", "citations": [], "sources": []}

    engine.base_engine.query = mock_query

    # Add some entries
    engine.query("Query 1")
    engine.query("Query 2")
    engine.query("Query 1")  # Cache hit

    stats = engine.get_cache_stats()
    assert stats["cache_size"] == 2
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2

    # Clear cache
    engine.clear_cache()

    stats = engine.get_cache_stats()
    assert stats["cache_size"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0


def test_cache_key_normalization(monkeypatch):
    """Test that cache key is normalized (case-insensitive, trimmed)."""
    # Mock required environment variables
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    engine = CachedRAGEngine(max_cache_size=10, cache_ttl_seconds=60)

    call_count = [0]

    def mock_query(question, use_rerank=True):
        call_count[0] += 1
        return {"answer": "Answer", "citations": [], "sources": []}

    engine.base_engine.query = mock_query

    # All these should be treated as the same query
    engine.query("What is ML?")
    engine.query("what is ml?")
    engine.query("  What is ML?  ")
    engine.query("WHAT IS ML?")

    assert call_count[0] == 1, "All variations should hit the same cache entry"
