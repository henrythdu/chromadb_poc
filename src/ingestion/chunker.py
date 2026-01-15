"""Document chunking using LlamaIndex node parsers."""

import logging
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk markdown documents with size-based splitting."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize SentenceSplitter for size-based chunking
        # This handles markdown structure naturally while respecting chunk size
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_markdown(
        self,
        markdown: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Chunk markdown content with metadata enrichment.

        Args:
            markdown: Markdown content to chunk
            metadata: Base metadata to attach to all chunks

        Returns:
            List of chunks with text and metadata
        """
        # Create LlamaIndex Document
        doc = Document(text=markdown, metadata=metadata)

        # Parse into nodes
        nodes = self.node_parser.get_nodes_from_documents([doc])

        # Enrich each chunk with metadata
        chunks = []
        for idx, node in enumerate(nodes):
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                # Extract section from node metadata if available
                "section": node.metadata.get(
                    "header", metadata.get("section", "Unknown")
                ),
            }

            chunks.append(
                {
                    "text": node.text,
                    "metadata": chunk_metadata,
                }
            )

        logger.info(f"Chunked into {len(chunks)} chunks")
        return chunks

    def enrich_metadata(
        self,
        chunk: Dict[str, Any],
        page_number: int | None = None,
    ) -> Dict[str, Any]:
        """Enrich chunk metadata with additional fields.

        Args:
            chunk: Chunk with text and base metadata
            page_number: Optional page number

        Returns:
            Enriched chunk
        """
        enriched = chunk.copy()
        enriched_metadata = enriched["metadata"].copy()

        if page_number is not None:
            enriched_metadata["page_number"] = page_number

        # Add contextual header for better retrieval
        title = chunk["metadata"].get("title", "Unknown")
        arxiv_id = chunk["metadata"].get("arxiv_id", "Unknown")
        section = chunk["metadata"].get("section", "Unknown")

        header = f"[Document: {title} | arxiv:{arxiv_id} | Section: {section}]"

        enriched["text"] = f"{header}\n\n{chunk['text']}"
        enriched["metadata"] = enriched_metadata

        return enriched
