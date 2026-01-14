"""Hybrid search combining vector similarity and BM25 keyword matching."""
import logging
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class HybridSearchRetriever:
    """Hybrid search using vector + BM25 with Reciprocal Rank Fusion."""

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
        if use_fusion:
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

                formatted_results.append({
                    "text": doc,
                    "metadata": metadata,
                    "score": score,
                })

        return formatted_results

    def _hybrid_retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve using hybrid vector + BM25 with RRF fusion.

        For now, use vector search only.
        BM25 requires building index from all documents first.
        """
        return self._vector_retrieve(query)

    def get_top_k(self, k: int | None = None) -> int:
        """Get or set top_k value."""
        if k is not None:
            self.top_k = k
        return self.top_k
