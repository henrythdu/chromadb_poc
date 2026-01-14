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
            embedding_model: Embedding model name
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

        # Initialize retrievers
        self._setup_retrievers()

    def _setup_retrievers(self):
        """Set up vector retriever."""
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # Create Chroma vector store
        vector_store = ChromaVectorStore(
            chroma_collection=self.collection,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from existing collection
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        # Vector retriever
        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
        )

        # BM25 retrieever (will be built from documents later)
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
        """Retrieve using vector search only."""
        nodes = self.vector_retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else 0.0,
            })

        return results[: self.top_k]

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
