"""RAG query engine: retrieve → rerank → generate."""

import logging
from typing import Any

from ..generation.llm import OpenRouterLLM
from .hybrid_search import HybridSearchRetriever
from .reranker import CohereReranker

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """End-to-end RAG query pipeline."""

    def __init__(
        self,
        chroma_api_key: str,
        chroma_tenant: str,
        chroma_database: str,
        cohere_api_key: str,
        openrouter_api_key: str,
        collection_name: str = "arxiv_papers_v1",
        top_k_retrieve: int = 50,
        top_k_rerank: int = 5,
        llm_model: str = "anthropic/claude-3.5-sonnet",
    ):
        """Initialize RAG query engine.

        Args:
            chroma_api_key: Chroma API key
            chroma_tenant: Chroma Cloud tenant ID
            chroma_database: Chroma Cloud database name
            cohere_api_key: Cohere API key
            openrouter_api_key: OpenRouter API key
            collection_name: Chroma collection name
            top_k_retrieve: Initial retrieval count
            top_k_rerank: Final result count after reranking
            llm_model: LLM model to use
        """
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Initialize components
        self.retriever = HybridSearchRetriever(
            chroma_api_key=chroma_api_key,
            chroma_tenant=chroma_tenant,
            chroma_database=chroma_database,
            collection_name=collection_name,
            top_k=top_k_retrieve,
        )

        self.reranker = CohereReranker(api_key=cohere_api_key, top_n=top_k_rerank)

        self.llm = OpenRouterLLM(api_key=openrouter_api_key, model=llm_model)

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
