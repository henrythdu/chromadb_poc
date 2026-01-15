"""Orchestrate ingestion pipeline: parse → chunk → embed → store."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict

from .chroma_store import ChromaStore
from .chunker import DocumentChunker
from .metadata import MetadataBuilder
from .parser import DoclingWrapper, LlamaParserWrapper

logger = logging.getLogger(__name__)


class IngestionIndexer:
    """Orchestrate the full ingestion pipeline."""

    def __init__(
        self,
        chroma_api_key: str,
        chroma_tenant: str,
        chroma_database: str,
        llamaparse_api_key: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        collection_name: str = "papers",
    ):
        """Initialize indexer components.

        Args:
            chroma_api_key: ChromaDB API key
            chroma_tenant: ChromaDB tenant ID
            chroma_database: ChromaDB database name
            llamaparse_api_key: LlamaParse API key (optional, uses Docling if not provided)
            chunk_size: Chunk size for splitting
            chunk_overlap: Chunk overlap
            collection_name: Chroma collection name
        """
        # Use Docling by default (free, local), fallback to LlamaParse if API key provided
        if llamaparse_api_key:
            logger.info("Using LlamaParse (cloud service)")
            self.parser = LlamaParserWrapper(api_key=llamaparse_api_key)
        else:
            logger.info("Using Docling (local, free)")
            self.parser = DoclingWrapper()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
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
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Index a single paper through the full pipeline.

        Args:
            pdf_path: Path to PDF file
            metadata: Paper metadata (arxiv_id, title, authors, etc.)
            skip_existing: If True, skip papers already in ChromaDB

        Returns:
            Result dict with status and counts
        """
        arxiv_id = metadata.get("arxiv_id")
        if not arxiv_id or not str(arxiv_id).strip():
            logger.error(
                f"Cannot index paper, arxiv_id is missing or blank from metadata. PDF: {pdf_path}"
            )
            return {
                "status": "error",
                "error": "Missing or blank arxiv_id in metadata",
                "chunks_indexed": 0,
                "arxiv_id": "unknown",
            }

        # Skip if already indexed
        if skip_existing and self.chroma.paper_exists(arxiv_id):
            logger.info(f"Skipping {arxiv_id} (already indexed)")
            return {
                "status": "skipped",
                "chunks_indexed": 0,
                "arxiv_id": arxiv_id,
            }

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
        enriched_chunks = [self.chunker.enrich_metadata(chunk) for chunk in chunks]

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
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Index multiple papers.

        Args:
            papers: List of papers with file_path and metadata
            skip_existing: If True, skip papers already in ChromaDB

        Returns:
            Summary with success/failure/skipped counts
        """
        results = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
        }

        for paper in papers:
            pdf_path = paper.get("file_path")
            metadata = {k: v for k, v in paper.items() if k != "file_path"}

            # Validate pdf_path exists
            if pdf_path is None:
                logger.error(
                    f"Missing file_path for paper {metadata.get('arxiv_id', 'unknown')}"
                )
                results["failed"] += 1
                continue

            result = self.index_paper(pdf_path, metadata, skip_existing=skip_existing)

            if result["status"] == "success":
                results["success"] += 1
                results["total_chunks"] += result["chunks_indexed"]
            elif result["status"] == "skipped":
                results["skipped"] += 1
            else:
                results["failed"] += 1
                logger.error(
                    f"Failed to index {metadata.get('arxiv_id')}: {result.get('error')}"
                )

        return results

    def index_batch_parallel(
        self,
        papers: list[Dict[str, Any]],
        max_workers: int = 5,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Index multiple papers in parallel using thread pool.

        Args:
            papers: List of papers with file_path and metadata
            max_workers: Maximum number of concurrent threads (default: 5)
            skip_existing: If True, skip papers already in ChromaDB

        Returns:
            Summary with success/failure/skipped counts

        Note:
            There is a potential race condition if multiple ingestion processes run
            concurrently on the same new paper. Both might see that it doesn't exist
            and both attempt to index it. The `upsert` in `add_documents` prevents
            data corruption, but computational work would be duplicated.
        """
        results = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
        }
        lock = Lock()  # Thread-safe counter updates

        # Filter out papers without file_path first
        valid_papers = [p for p in papers if p.get("file_path") is not None]

        if len(valid_papers) < len(papers):
            missing = len(papers) - len(valid_papers)
            logger.warning(f"Skipping {missing} papers without file_path")
            results["failed"] += missing

        # Batch check for existing papers to avoid N+1 queries
        papers_to_process = valid_papers
        if skip_existing and valid_papers:
            all_ids = [p.get("arxiv_id") for p in valid_papers if p.get("arxiv_id")]
            existing_ids = self.chroma.get_existing_paper_ids(all_ids)
            logger.info(
                f"Pre-flight check: {len(existing_ids)} papers already indexed, will skip"
            )

            # Filter out existing papers
            papers_to_process = [
                p for p in valid_papers if p.get("arxiv_id") not in existing_ids
            ]
            results["skipped"] = len(existing_ids)

        logger.info(
            f"Processing {len(papers_to_process)} new papers with {max_workers} workers"
        )

        def process_single_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single paper, returns result dict."""
            pdf_path = paper.get("file_path")
            metadata = {k: v for k, v in paper.items() if k != "file_path"}
            # skip_existing=False since we already filtered above
            return self.index_paper(pdf_path, metadata, skip_existing=False)

        # Process papers in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_paper = {
                executor.submit(process_single_paper, paper): paper
                for paper in papers_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    result = future.result()

                    # Thread-safe update of results
                    with lock:
                        if result["status"] == "success":
                            results["success"] += 1
                            results["total_chunks"] += result["chunks_indexed"]
                            logger.info(
                                f"✓ {result['arxiv_id']}: {result['chunks_indexed']} chunks"
                            )
                        else:
                            results["failed"] += 1
                            logger.error(
                                f"✗ {result.get('arxiv_id', 'unknown')}: "
                                f"{result.get('error', 'Unknown error')}"
                            )

                except Exception as e:
                    # Handle unexpected exceptions
                    with lock:
                        results["failed"] += 1
                    logger.error(
                        f"Exception processing {paper.get('arxiv_id', 'unknown')}: {e}",
                        exc_info=True,
                    )

        logger.info(
            f"Parallel processing complete: {results['success']} success, "
            f"{results['skipped']} skipped, {results['failed']} failed, "
            f"{results['total_chunks']} total chunks"
        )

        return results
