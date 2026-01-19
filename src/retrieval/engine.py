"""RAG query engine: retrieve → rerank → generate."""

import logging
from typing import Any

from ..config import settings
from ..generation.llm import OpenRouterLLM
from .hybrid_search import HybridSearchRetriever
from .reranker import CohereReranker

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """End-to-end RAG query pipeline."""

    def __init__(
        self,
        chroma_api_key: str | None = None,
        chroma_tenant: str | None = None,
        chroma_database: str | None = None,
        cohere_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        collection_name: str = "arxiv_papers_v1",
        top_k_retrieve: int | None = None,
        top_k_rerank: int | None = None,
        llm_model: str | None = None,
    ):
        """Initialize RAG query engine.

        Args:
            chroma_api_key: Chroma API key (from settings if None)
            chroma_tenant: Chroma Cloud tenant ID (from settings if None)
            chroma_database: Chroma Cloud database name (from settings if None)
            cohere_api_key: Cohere API key (from settings if None)
            openrouter_api_key: OpenRouter API key (from settings if None)
            collection_name: Chroma collection name
            top_k_retrieve: Initial retrieval count (from config.toml if None)
            top_k_rerank: Final result count after reranking (from config.toml if None)
            llm_model: LLM model to use (from config.toml if None)
        """
        # Use settings for defaults if not provided
        self.top_k_retrieve = top_k_retrieve or settings.top_k_retrieve
        self.top_k_rerank = top_k_rerank or settings.top_k_rerank
        self.llm_model = llm_model or settings.llm_model

        chroma_api_key = chroma_api_key or settings.chroma_cloud_api_key
        chroma_tenant = chroma_tenant or settings.chroma_tenant
        chroma_database = chroma_database or settings.chroma_database
        cohere_api_key = cohere_api_key or settings.cohere_api_key
        openrouter_api_key = openrouter_api_key or settings.openrouter_api_key

        # Initialize components
        self.retriever = HybridSearchRetriever(
            chroma_api_key=chroma_api_key,
            chroma_tenant=chroma_tenant,
            chroma_database=chroma_database,
            collection_name=collection_name,
            top_k=self.top_k_retrieve,
        )

        self.reranker = CohereReranker(api_key=cohere_api_key, top_n=self.top_k_rerank)

        self.llm = OpenRouterLLM(api_key=openrouter_api_key, model=self.llm_model)

    def query(
        self,
        question: str,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        """Execute end-to-end RAG query.

        Args:
            question: User question
            use_rerank: Whether to use reranker

        Returns:
            Result dict with answer and citations
        """
        logger.info("Executing new RAG query...")
        logger.debug(f"Query: {question}")  # Only log at debug level

        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(question)
        logger.info(f"Retrieved {len(retrieved)} chunks")

        if not retrieved:
            return {
                "answer": "No relevant information found in the papers.",
                "citations": [],
                "sources": [],
            }

        # Step 2: Rerank
        if use_rerank:
            reranked = self.reranker.rerank_results(question, retrieved)
            logger.info(f"Reranked to {len(reranked)} chunks")
        else:
            reranked = retrieved[: self.top_k_rerank]

        # Step 3: Generate answer
        answer = self.llm.answer_question(question, reranked)

        # Step 4: Extract citations
        citations = self._extract_citations(reranked)

        return {
            "answer": answer,
            "citations": citations,
            "sources": [r["metadata"] for r in reranked],
        }

    def query_with_collection(
        self,
        question: str,
        collection_name: str,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        """Query a specific collection by name.

        Args:
            question: User question
            collection_name: Collection name to query ("arxiv_papers_v1" or "contracts")
            use_rerank: Whether to use reranker

        Returns:
            Result dict with answer and citations

        Raises:
            ValueError: If collection_name is not valid
        """
        valid_collections = ["arxiv_papers_v1", "contracts"]
        if collection_name not in valid_collections:
            raise ValueError(
                f"Invalid collection: {collection_name}. "
                f"Valid options: {valid_collections}"
            )

        logger.info(f"Querying collection: {collection_name}")
        logger.debug(f"Query: {question}")

        # Create temporary retriever for specified collection
        retriever = HybridSearchRetriever(
            chroma_api_key=self.retriever.client._api_key,
            chroma_tenant=self.retriever.client._tenant,
            chroma_database=self.retriever.client._database,
            collection_name=collection_name,
            top_k=self.top_k_retrieve,
        )

        # Step 1: Retrieve
        retrieved = retriever.retrieve(question)
        logger.info(f"Retrieved {len(retrieved)} chunks from {collection_name}")

        if not retrieved:
            collection_type = "contracts" if collection_name == "contracts" else "papers"
            return {
                "answer": f"No relevant information found in the {collection_type}.",
                "citations": [],
                "sources": [],
            }

        # Step 2: Rerank
        if use_rerank:
            reranked = self.reranker.rerank_results(question, retrieved)
            logger.info(f"Reranked to {len(reranked)} chunks")
        else:
            reranked = retrieved[: self.top_k_rerank]

        # Step 3: Generate answer
        answer = self.llm.answer_question(question, reranked)

        # Step 4: Extract citations
        citations = self._extract_citations(reranked)

        return {
            "answer": answer,
            "citations": citations,
            "sources": [r["metadata"] for r in reranked],
        }

    def _extract_citations(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[str]:
        """Extract citation strings from chunks.

        Args:
            chunks: Reranked chunks

        Returns:
            List of formatted citations
        """
        from ..ingestion.metadata import MetadataBuilder

        metadata_builder = MetadataBuilder()
        citations = []
        seen = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            arxiv_id = metadata.get("arxiv_id", "Unknown")
            page = metadata.get("page_number", "?")

            key = f"{arxiv_id}:{page}"
            if key not in seen:
                citations.append(metadata_builder.format_citation(metadata))
                seen.add(key)

        return citations
