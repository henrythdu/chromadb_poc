"""Hybrid search combining vector similarity and BM25 keyword matching."""

import logging
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class HybridSearchRetriever:
    """Hybrid search using vector + BM25 with Reciprocal Rank Fusion."""

    # RRF constant (typically 50-60 for diversity)
    RRF_K = 60

    def __init__(
        self,
        chroma_api_key: str,
        chroma_tenant: str,
        chroma_database: str,
        collection_name: str = "arxiv_papers_v1",
        top_k: int = 50,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        """Initialize hybrid search retriever.

        Args:
            chroma_api_key: Chroma API key
            chroma_tenant: Chroma Cloud tenant ID
            chroma_database: Chroma Cloud database name
            collection_name: Collection to search
            top_k: Number of results to retrieve
            embedding_model: Embedding model name (for future use with local embeddings)
        """
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Initialize Chroma client with correct parameters
        self.client = chromadb.CloudClient(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database,
        )

        # Get collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # BM25 retriever (will be built from documents later)
        self.bm25_retriever = None
        self.bm25_corpus = []

    def build_bm25_index(self, documents: list[str]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document texts for BM25 indexing
        """
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents for BM25
            tokenized_corpus = [doc.split() for doc in documents]

            # Initialize BM25
            self.bm25_retriever = BM25Okapi(tokenized_corpus)
            self.bm25_corpus = documents

            logger.info(f"Built BM25 index with {len(documents)} documents")
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search will be unavailable")
            self.bm25_retriever = None

    def retrieve(
        self,
        query: str,
        use_fusion: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks using hybrid search.

        Args:
            query: Search query
            use_fusion: Whether to use RRF fusion of vector+BM25

        Returns:
            List of retrieved chunks with text, metadata, score
        """
        if use_fusion and self.bm25_retriever is not None:
            return self._hybrid_retrieve(query)
        else:
            return self._vector_retrieve(query)

    def _vector_retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve using vector search only.

        Uses Chroma's native query method which automatically uses the
        tenant's default embedding function (matching ingestion).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k,
        )

        # Format results
        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = {}
                if results["metadatas"] and results["metadatas"][0]:
                    metadata = results["metadatas"][0][i] or {}

                distance = 0.0
                if results["distances"] and results["distances"][0]:
                    distance = results["distances"][0][i] or 0.0

                # Convert distance to similarity score (higher is better)
                score = 1.0 - distance if distance is not None else 0.0

                formatted_results.append(
                    {
                        "text": doc,
                        "metadata": metadata,
                        "score": score,
                    }
                )

        return formatted_results

    def _bm25_retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve using BM25 keyword search.

        Args:
            query: Search query

        Returns:
            List of retrieved chunks with text, metadata, score
        """
        if self.bm25_retriever is None:
            logger.warning(
                "BM25 retriever not initialized, falling back to vector search"
            )
            return self._vector_retrieve(query)

        # Tokenize query
        tokenized_query = query.split()

        # Get BM25 scores
        import numpy as np

        scores = self.bm25_retriever.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        # Get all documents to retrieve metadata
        # Note: BM25 only gives us document indices, we need to fetch from Chroma
        # For now, return scores with indices
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                # Normalize score to 0-1 range (approximately)
                # BM25 scores can vary widely, so we use sigmoid-like normalization
                normalized_score = 1.0 / (1.0 + np.exp(-scores[idx] / 10))
                results.append(
                    {
                        "text": self.bm25_corpus[idx],
                        "metadata": {"bm25_index": int(idx)},
                        "score": float(normalized_score),
                        "bm25_index": int(idx),
                    }
                )

        return results

    def _hybrid_retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve using hybrid vector + BM25 with RRF fusion.

        Uses Reciprocal Rank Fusion (RRF) to combine results from
        vector search and BM25 keyword search.

        RRF formula: score = 1/(k + rank) where k is typically 60

        Args:
            query: Search query

        Returns:
            List of retrieved chunks with text, metadata, combined_score
        """
        # Get results from both retrievers
        vector_results = self._vector_retrieve(query)
        bm25_results = self._bm25_retrieve(query)

        # Calculate RRF scores
        rrf_scores = {}

        # Add vector search scores
        for rank, result in enumerate(vector_results, start=1):
            # Use a unique identifier for the document
            doc_id = result.get("metadata", {}).get("id", id(result["text"]))
            rrf_score = 1.0 / (self.RRF_K + rank)
            rrf_scores[doc_id] = {
                "rrf_score": rrf_score,
                "result": result,
                "vector_rank": rank,
                "bm25_rank": None,
            }

        # Add BM25 scores
        for rank, result in enumerate(bm25_results, start=1):
            # Try to find matching document by text
            doc_id = result.get("metadata", {}).get("id", id(result["text"]))
            rrf_score = 1.0 / (self.RRF_K + rank)

            if doc_id in rrf_scores:
                # Combine scores
                rrf_scores[doc_id]["rrf_score"] += rrf_score
                rrf_scores[doc_id]["bm25_rank"] = rank
            else:
                # Only in BM25 results
                rrf_scores[doc_id] = {
                    "rrf_score": rrf_score,
                    "result": result,
                    "vector_rank": None,
                    "bm25_rank": rank,
                }

        # Sort by combined RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Return top-k results
        final_results = []
        for item in sorted_results[: self.top_k]:
            result = item["result"].copy()
            result["rrf_score"] = item["rrf_score"]
            result["vector_rank"] = item["vector_rank"]
            result["bm25_rank"] = item["bm25_rank"]
            final_results.append(result)

        logger.info(
            f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"→ {len(final_results)} fused results"
        )

        return final_results

    def get_top_k(self, k: int | None = None) -> int:
        """Get or set top_k value."""
        if k is not None:
            self.top_k = k
        return self.top_k

    def has_bm25(self) -> bool:
        """Check if BM25 index is built."""
        return self.bm25_retriever is not None
