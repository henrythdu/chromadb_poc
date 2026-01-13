"""Orchestrate ingestion pipeline: parse → chunk → embed → store."""
import logging
from typing import Any, Dict

from .chroma_store import ChromaStore
from .chunker import DocumentChunker
from .metadata import MetadataBuilder
from .parser import LlamaParserWrapper

logger = logging.getLogger(__name__)


class IngestionIndexer:
    """Orchestrate the full ingestion pipeline."""

    def __init__(
        self,
        llamaparse_api_key: str,
        chroma_api_key: str,
        chroma_tenant: str,
        chroma_database: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        collection_name: str = "papers",
    ):
        """Initialize indexer components.

        Args:
            llamaparse_api_key: LlamaParse API key
            chroma_api_key: ChromaDB API key
            chroma_tenant: ChromaDB tenant ID
            chroma_database: ChromaDB database name
            chunk_size: Chunk size for splitting
            chunk_overlap: Chunk overlap
            collection_name: Chroma collection name
        """
        self.parser = LlamaParserWrapper(api_key=llamaparse_api_key)
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.metadata_builder = MetadataBuilder()
        self.chroma = ChromaStore(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database,
            collection_name=collection_name,
        )

    def index_paper(
        self,
        pdf_path: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Index a single paper through the full pipeline.

        Args:
            pdf_path: Path to PDF file
            metadata: Paper metadata (arxiv_id, title, authors, etc.)

        Returns:
            Result dict with status and counts
        """
        logger.info(f"Indexing: {metadata.get('title', 'Unknown')}")

        # Step 1: Parse PDF to Markdown
        parse_result = self.parser.parse_pdf(pdf_path)

        if parse_result.get("error"):
            return {
                "status": "error",
                "error": parse_result["error"],
                "chunks_indexed": 0,
            }

        markdown = parse_result["markdown"]

        # Step 2: Build document metadata
        doc_metadata = self.metadata_builder.build_document_metadata(metadata)

        # Step 3: Chunk markdown
        chunks = self.chunker.chunk_markdown(markdown, doc_metadata)

        # Step 4: Enrich chunks with metadata
        enriched_chunks = []
        for chunk in chunks:
            enriched = self.chunker.enrich_metadata(chunk)
            enriched_chunks.append(enriched)

        # Step 5: Store in Chroma
        documents = [c["text"] for c in enriched_chunks]
        metadatas = [c["metadata"] for c in enriched_chunks]
        ids = [f"{metadata['arxiv_id']}_{i}" for i in range(len(chunks))]

        self.chroma.add_documents(documents, metadatas, ids)

        logger.info(f"Indexed {len(chunks)} chunks from {metadata['arxiv_id']}")

        return {
            "status": "success",
            "chunks_indexed": len(chunks),
            "arxiv_id": metadata["arxiv_id"],
        }

    def index_batch(
        self,
        papers: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Index multiple papers.

        Args:
            papers: List of papers with file_path and metadata

        Returns:
            Summary with success/failure counts
        """
        results = {
            "success": 0,
            "failed": 0,
            "total_chunks": 0,
        }

        for paper in papers:
            pdf_path = paper.get("file_path")
            metadata = {k: v for k, v in paper.items() if k != "file_path"}

            # Validate pdf_path exists
            if pdf_path is None:
                logger.error(f"Missing file_path for paper {metadata.get('arxiv_id', 'unknown')}")
                results["failed"] += 1
                continue

            result = self.index_paper(pdf_path, metadata)

            if result["status"] == "success":
                results["success"] += 1
                results["total_chunks"] += result["chunks_indexed"]
            else:
                results["failed"] += 1
                logger.error(f"Failed to index {metadata.get('arxiv_id')}: {result.get('error')}")

        return results
