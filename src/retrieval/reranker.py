"""Cohere Rerank v3 integration for result filtering."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CohereReranker:
    """Cohere Rerank v3 for filtering retrieved chunks."""

    def __init__(
        self,
        api_key: str,
        top_n: int = 5,
        model: str = "rerank-v3",
    ):
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            top_n: Number of top results to return
            model: Rerank model to use
        """
        self.api_key = api_key
        self.top_n = top_n
        self.model = model

        try:
            from cohere import Client as CohereClient

            self.client = CohereClient(api_key=api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Cohere client: {e}")
            self.client = None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document texts
            top_n: Override default top_n

        Returns:
            Reranked results with scores
        """
        if self.client is None:
            logger.warning("Cohere client not initialized, returning original order")
            return [
                {"index": i, "text": doc, "relevance_score": 0.5}
                for i, doc in enumerate(documents[: self.top_n])
            ]

        top_n = top_n or self.top_n

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "text": documents[result.index],
                    "relevance_score": result.relevance_score,
                })

            logger.info(f"Reranked {len(documents)} → {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return top_n in original order
            return [
                {"index": i, "text": doc, "relevance_score": 0.5}
                for i, doc in enumerate(documents[:top_n])
            ]

    def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank retrieval results.

        Args:
            query: Search query
            results: List of results with 'text' field
            top_n: Override default top_n

        Returns:
            Reranked results
        """
        documents = [r["text"] for r in results]

        reranked = self.rerank(query, documents, top_n)

        # Map back to original results with metadata
        output = []
        for r in reranked:
            original = results[r["index"]]
            output.append({**original, "rerank_score": r["relevance_score"]})

        return output
