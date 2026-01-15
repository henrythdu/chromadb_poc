"""Metadata enrichment for documents and chunks."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """Build and enrich metadata for documents and chunks."""

    def build_document_metadata(self, paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build document-level metadata.

        Args:
            paper_metadata: Raw metadata from downloader

        Returns:
            Enriched document metadata with ChromaDB-compatible values
        """
        # Convert authors list to string for ChromaDB compatibility
        # ChromaDB only accepts: str, int, float, bool, or None
        authors = paper_metadata.get("authors", [])
        authors_str = ", ".join(authors) if authors else "Unknown"

        return {
            "document_id": paper_metadata["arxiv_id"],
            "arxiv_id": paper_metadata["arxiv_id"],
            "title": paper_metadata.get("title", "Unknown"),
            "authors": authors_str,
            "published_date": paper_metadata.get("published_date", ""),
            "pdf_url": paper_metadata.get("pdf_url", ""),
        }

    def build_chunk_metadata(
        self,
        doc_metadata: Dict[str, Any],
        chunk_index: int,
        section: str,
        page_number: int | None = None,
    ) -> Dict[str, Any]:
        """Build chunk-level metadata.

        Args:
            doc_metadata: Document metadata
            chunk_index: Index of this chunk in document
            section: Section name (e.g., "Introduction")
            page_number: Optional page number

        Returns:
            Enriched chunk metadata
        """
        chunk_metadata = {
            **doc_metadata,
            "chunk_index": chunk_index,
            "section": section,
        }

        if page_number is not None:
            chunk_metadata["page_number"] = page_number

        return chunk_metadata

    def format_citation(self, metadata: Dict[str, Any]) -> str:
        """Format citation string from metadata.

        Args:
            metadata: Chunk or document metadata

        Returns:
            Formatted citation string
        """
        title = metadata.get("title", "Unknown")[:50]
        arxiv_id = metadata.get("arxiv_id", "Unknown")
        page = metadata.get("page_number", "?")

        return f"[{title}... (arxiv:{arxiv_id}), page {page}]"
