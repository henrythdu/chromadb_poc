#!/usr/bin/env python3
"""Ingest contract PDFs into ChromaDB.

This script scans the full_contract_pdf directory for SEC contract PDFs,
extracts text, chunks it, and stores it in ChromaDB's "contracts" collection.

Usage:
    python scripts/ingest_contracts.py

The script:
1. Scans full_contract_pdf/ for PDF files
2. Parses filenames to extract metadata (company, date, filing type, etc.)
3. Extracts text from PDFs using Docling
4. Chunks text using DocumentChunker
5. Stores in ChromaDB's "contracts" collection
6. Resumable: skips already-ingested contracts
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.chroma_store import ChromaStore
from src.ingestion.chunker import DocumentChunker
from src.ingestion.contract_filename_parser import (
    ContractFilenameParser,
    normalize_company_name,
    parse_contract_path,
)
from src.ingestion.parser import DoclingWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_contract_pdfs(contracts_dir: str | Path) -> list[Path]:
    """Find all PDF files in the contracts directory.

    Args:
        contracts_dir: Path to full_contract_pdf directory

    Returns:
        List of Path objects for PDF files
    """
    contracts_path = Path(contracts_dir)
    if not contracts_path.exists():
        logger.error(f"Contracts directory not found: {contracts_dir}")
        return []

    pdf_files = list(contracts_path.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {contracts_dir}")
    return pdf_files


def contract_exists(store: ChromaStore, document_id: str) -> bool:
    """Check if a contract already exists in ChromaDB.

    Args:
        store: ChromaStore instance
        document_id: Unique document ID (SHA256 of filename)

    Returns:
        True if contract exists, False otherwise
    """
    collection = store.get_collection("contracts")

    # Query for documents with this document_id in metadata
    results = collection.get(
        where={"document_id": document_id},
        limit=1,  # Only need to know if at least one exists
    )

    exists = len(results.get("ids", [])) > 0
    return exists


def index_contract(
    pdf_path: Path,
    parser: ContractFilenameParser,
    pdf_parser: DoclingWrapper,
    chunker: DocumentChunker,
    store: ChromaStore,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Index a single contract PDF.

    Args:
        pdf_path: Path to PDF file
        parser: ContractFilenameParser instance
        pdf_parser: DoclingWrapper instance
        chunker: DocumentChunker instance
        store: ChromaStore instance
        skip_existing: If True, skip already-indexed contracts

    Returns:
        Result dict with status and counts
    """
    try:
        # Step 1: Parse filename for metadata
        metadata = parse_contract_path(str(pdf_path))

        # Check if already exists
        if skip_existing and contract_exists(store, metadata["document_id"]):
            logger.info(f"Skipping {metadata['filename']} (already indexed)")
            return {
                "status": "skipped",
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
            }

        logger.info(f"Indexing: {metadata['filename']}")

        # Step 2: Parse PDF to Markdown
        parse_result = pdf_parser.parse_pdf(pdf_path)

        if parse_result.get("error"):
            return {
                "status": "error",
                "error": parse_result["error"],
                "document_id": metadata.get("document_id", "unknown"),
                "filename": metadata.get("filename", pdf_path.name),
            }

        markdown = parse_result["markdown"]

        # Step 3: Build document metadata for chunking
        # Include all parsed metadata plus normalized company name
        doc_metadata = {
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "company_name": normalize_company_name(metadata["company_name"]),
            "execution_date": metadata["execution_date"],
            "filing_type": metadata["filing_type"],
            "exhibit_number": metadata["exhibit_number"],
            "accession_number": metadata["accession_number"],
            "contract_type": metadata["contract_type"],
            "source": "SEC_contracts",
        }

        # Step 4: Chunk markdown
        chunks = chunker.chunk_markdown(markdown, doc_metadata)

        # Step 5: Enrich chunks with contextual header
        enriched_chunks = []
        for chunk in chunks:
            # Add contract-specific header for better retrieval
            company = chunk["metadata"].get("company_name", "Unknown")
            contract_type = chunk["metadata"].get("contract_type", "Unknown")
            date = chunk["metadata"].get("execution_date", "Unknown")

            header = f"[Contract: {company} | Type: {contract_type} | Date: {date}]"
            enriched_chunk = {
                "text": f"{header}\n\n{chunk['text']}",
                "metadata": chunk["metadata"],
            }
            enriched_chunks.append(enriched_chunk)

        # Step 6: Store in ChromaDB "contracts" collection
        documents = [c["text"] for c in enriched_chunks]
        metadatas = [c["metadata"] for c in enriched_chunks]
        ids = [f"{metadata['document_id']}_{i}" for i in range(len(chunks))]

        store.add_documents(
            documents,
            metadatas,
            ids,
            collection_name="contracts",  # Store in contracts collection
        )

        logger.info(
            f"✓ Indexed {len(chunks)} chunks from {metadata['filename']} "
            f"({metadata['company_name']} - {metadata['contract_type']})"
        )

        return {
            "status": "success",
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "company_name": metadata["company_name"],
            "contract_type": metadata["contract_type"],
            "chunks_indexed": len(chunks),
        }

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "filename": pdf_path.name,
        }


def main():
    """Main ingestion function."""
    logger.info("=" * 60)
    logger.info("Contract Ingestion Pipeline Starting")
    logger.info("=" * 60)

    # Validate credentials
    if not all([settings.chroma_cloud_api_key, settings.chroma_tenant, settings.chroma_database]):
        logger.error("ChromaDB credentials not configured. Set CHROMA_CLOUD_API_KEY, CHROMA_TENANT, CHROMA_DATABASE")
        return

    # Initialize components
    logger.info("Initializing ingestion components...")

    store = ChromaStore(
        api_key=settings.chroma_cloud_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
        collection_name="contracts",  # Use contracts collection
    )

    # Test connection
    if not store.test_connection():
        logger.error("Failed to connect to ChromaDB. Check credentials.")
        return

    # Get current count in contracts collection
    current_count = store.count(collection_name="contracts")
    logger.info(f"Current contracts collection has {current_count} chunks")

    # Initialize parsers and chunker
    parser = ContractFilenameParser()
    pdf_parser = DoclingWrapper(enable_ocr=False)  # OCR disabled for speed
    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=100,
    )

    # Find all contract PDFs
    contracts_dir = Path(__file__).parent.parent / "full_contract_pdf"
    pdf_files = find_contract_pdfs(contracts_dir)

    if not pdf_files:
        logger.error("No PDF files found. Exiting.")
        return

    # Process all contracts
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total_chunks": 0,
    }

    logger.info(f"Processing {len(pdf_files)} contract PDFs...")
    logger.info("-" * 60)

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        result = index_contract(
            pdf_path,
            parser,
            pdf_parser,
            chunker,
            store,
            skip_existing=True,  # Resumable: skip existing
        )

        if result["status"] == "success":
            results["success"] += 1
            results["total_chunks"] += result.get("chunks_indexed", 0)
        elif result["status"] == "skipped":
            results["skipped"] += 1
        else:
            results["failed"] += 1
            logger.error(f"✗ Failed: {result.get('filename')} - {result.get('error')}")

    # Final summary
    logger.info("-" * 60)
    logger.info("Ingestion Complete!")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Skipped: {results['skipped']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Total chunks added: {results['total_chunks']}")

    # Get final count
    final_count = store.count(collection_name="contracts")
    logger.info(f"Final contracts collection has {final_count} chunks")

    # Show parser statistics
    stats = parser.get_statistics()
    logger.info(f"Parser stats: {stats}")


if __name__ == "__main__":
    main()
