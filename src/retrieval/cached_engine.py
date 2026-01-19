"""LRU-cached RAG query engine for performance optimization."""

import hashlib
import logging
import threading
import time
from typing import Any

from .engine import RAGQueryEngine

logger = logging.getLogger(__name__)


class CachedRAGEngine:
    """RAG query engine with LRU result caching."""

    def __init__(
        self,
        base_engine: RAGQueryEngine | None = None,
        max_cache_size: int = 100,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize cached RAG engine.

        Args:
            base_engine: Base RAG engine (creates new if None)
            max_cache_size: Maximum number of cached results
            cache_ttl_seconds: Cache TTL in seconds (default 1 hour)
        """
        self.base_engine = base_engine or RAGQueryEngine()
        self.max_cache_size = max_cache_size
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cache structure: {key: (timestamp, result)}
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0

        # Lock for thread-safe cache operations
        self._lock = threading.Lock()

    def _cache_key(self, query: str) -> str:
        """Generate cache key from query.

        Args:
            query: User query string

        Returns:
            SHA256 hash of normalized query
        """
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def query(
        self,
        question: str,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        """Execute query with caching.

        Args:
            question: User question
            use_rerank: Whether to use reranker

        Returns:
            Result dict with answer and citations
        """
        key = self._cache_key(question)
        current_time = time.time()

        # Check cache with lock
        with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                age_seconds = current_time - timestamp

                if age_seconds < self.cache_ttl_seconds:
                    self._cache_hits += 1
                    cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
                    logger.info(
                        f"Cache HIT: '{question[:50]}...' "
                        f"(age: {age_seconds:.0f}s, hit rate: {cache_hit_rate:.1%})"
                    )
                    return result
                else:
                    # Cache expired, remove it
                    logger.debug(f"Cache EXPIRED: '{question[:50]}...'")
                    del self._cache[key]

            # Cache miss - increment counter
            self._cache_misses += 1

        # Run query outside of lock (I/O operation)
        result = self.base_engine.query(question, use_rerank)

        # Store in cache with lock
        with self._lock:
            self._cache[key] = (current_time, result)

            # LRU eviction if over limit
            if len(self._cache) > self.max_cache_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
                logger.debug(f"Cache EVICTED: {len(self._cache)}/{self.max_cache_size} entries")

        return result

    def clear_cache(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        logger.info("Cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache size, hits, misses, hit rate
        """
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.cache_ttl_seconds,
            }
