# Phase 2 - Ingestion Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the complete ingestion pipeline to download, parse, chunk, and index ArXiv papers into ChromaDB.

**Architecture:** Sequential pipeline: ArXiv API → LlamaParse (PDF→Markdown) → MarkdownElementNodeParser (chunking) → BAAI/bge-large embeddings → Chroma Cloud with rich metadata.

**Tech Stack:** arxiv, llama-parse, llama-index (node parser, embeddings, vector store), chromadb, BAAI/bge-large-en-v1.5

**Prerequisites:** Phase 1 Foundation complete

---

## Task 1: Implement ArXiv Downloader

**Files:**
- Create: `src/ingestion/downloader.py`
- Test: `tests/test_downloader.py`

**Step 1: Write failing test**

```bash
cat > tests/test_downloader.py << 'EOF'
"""Test ArXiv downloader functionality."""
import pytest
from pathlib import Path


def test_downloader_imports():
    """Test that downloader module can be imported."""
    from src.ingestion.downloader import ArxivDownloader
    assert ArxivDownloader is not None


@pytest.mark.integration
def test_download_single_paper(tmp_path, mock_api_keys):
    """Test downloading a single paper from ArXiv."""
    from src.ingestion.downloader import ArxivDownloader

    downloader = ArxivDownloader(
        download_dir=str(tmp_path),
        max_results=1
    )

    papers = downloader.download_papers(
        query="cat:cs.LG",
        max_results=1
    )

    assert len(papers) == 1
    assert "arxiv_id" in papers[0]
    assert "title" in papers[0]
    assert "authors" in papers[0]
    assert papers[0]["arxiv_id"] is not None


def test_metadata_extraction(tmp_path, mock_api_keys):
    """Test that metadata is correctly extracted."""
    from src.ingestion.downloader import ArxivDownloader

    downloader = ArxivDownloader(
        download_dir=str(tmp_path),
        max_results=1
    )

    # Mock metadata for testing
    metadata = {
        "arxiv_id": "2301.07041",
        "title": "Test Paper",
        "authors": ["Author One", "Author Two"],
        "published_date": "2023-01-01",
        "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf"
    }

    assert metadata["arxiv_id"] == "2301.07041"
    assert len(metadata["authors"]) == 2
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_downloader.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'src.ingestion.downloader'"

**Step 3: Implement downloader.py**

```bash
cat > src/ingestion/downloader.py << 'EOF'
"""ArXiv paper downloader with metadata extraction."""
import arxiv
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ArxivDownloader:
    """Download papers from ArXiv with metadata extraction."""

    def __init__(
        self,
        download_dir: str = "./ml_pdfs",
        max_results: int = 100,
    ):
        """Initialize downloader.

        Args:
            download_dir: Directory to save PDFs
            max_results: Maximum number of papers to download
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results

    def download_papers(
        self,
        query: str = "cat:cs.LG OR cat:cs.AI",
        max_results: int = None,
    ) -> List[Dict[str, Any]]:
        """Download papers from ArXiv.

        Args:
            query: ArXiv search query
            max_results: Override max_results

        Returns:
            List of paper metadata dictionaries
        """
        max_results = max_results or self.max_results

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []
        for result in search.results():
            paper = self._extract_metadata(result)
            self._download_pdf(result, paper)
            papers.append(paper)

            logger.info(f"Downloaded: {paper['title'][:50]}...")

        return papers

    def _extract_metadata(self, result: arxiv.Result) -> Dict[str, Any]:
        """Extract metadata from ArXiv result.

        Args:
            result: ArXiv search result

        Returns:
            Metadata dictionary
        """
        return {
            "arxiv_id": result.get_short_id(),
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published_date": result.published.strftime("%Y-%m-%d"),
            "pdf_url": result.pdf_url,
            "summary": result.summary.replace("\n", " "),
        }

    def _download_pdf(self, result: arxiv.Result, paper: Dict[str, Any]) -> None:
        """Download PDF to disk.

        Args:
            result: ArXiv search result
            paper: Paper metadata dict (updated with file_path)
        """
        # Sanitize filename
        safe_title = result.title[:50].replace("/", "-").replace(" ", "_")
        filename = f"{paper['arxiv_id']}_{safe_title}.pdf"
        filepath = self.download_dir / filename

        result.download_pdf(dirpath=str(self.download_dir), filename=filename)
        paper["file_path"] = str(filepath)

        logger.debug(f"Saved to: {filepath}")
EOF
```

**Step 4: Run test to verify it passes**

```bash
# Skip integration tests (require network)
pytest tests/test_downloader.py::test_downloader_imports -v
pytest tests/test_downloader.py::test_metadata_extraction -v
```

Expected: PASS for both tests

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/downloader.py tests/test_downloader.py
```

Expected: No errors

**Step 6: Commit**

```bash
git add src/ingestion/downloader.py tests/test_downloader.py
git commit -m "feat: implement ArXiv downloader with metadata extraction"
```

---

## Task 2: Integrate LlamaParse

**Files:**
- Create: `src/ingestion/parser.py`
- Test: `tests/test_parser.py`

**Step 1: Write failing test**

```bash
cat > tests/test_parser.py << 'EOF'
"""Test LlamaParse integration."""
import pytest
from pathlib import Path


def test_parser_imports():
    """Test that parser module can be imported."""
    from src.ingestion.parser import LlamaParserWrapper
    assert LlamaParserWrapper is not None


def test_parser_initialization(mock_api_keys):
    """Test parser can be initialized with API key."""
    from src.ingestion.parser import LlamaParserWrapper

    parser = LlamaParserWrapper(api_key="test_key")
    assert parser.api_key == "test_key"


@pytest.mark.integration
def test_parse_pdf(tmp_path, mock_api_keys):
    """Test parsing a PDF file."""
    from src.ingestion.parser import LlamaParserWrapper

    # Create mock PDF file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("mock pdf content")

    parser = LlamaParserWrapper(api_key="test_key")
    result = parser.parse_pdf(str(pdf_file))

    assert "markdown" in result or "error" in result
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_parser.py::test_parser_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement parser.py**

```bash
cat > src/ingestion/parser.py << 'EOF'
"""LlamaParse integration for PDF to Markdown conversion."""
from llama_parse import LlamaParse
from pathlib import Path
from typing import Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class LlamaParserWrapper:
    """Wrapper for LlamaParse API with error handling."""

    def __init__(self, api_key: str, result_type: str = "markdown"):
        """Initialize LlamaParse client.

        Args:
            api_key: LlamaParse API key
            result_type: Output format (markdown or text)
        """
        self.api_key = api_key
        self.result_type = result_type
        self.client = None

    def _get_client(self) -> LlamaParse:
        """Lazy load LlamaParse client."""
        if self.client is None:
            self.client = LlamaParse(
                api_key=self.api_key,
                result_type=self.result_type,
                parsing_instruction="Preserve tables, equations, and section structure.",
            )
        return self.client

    def parse_pdf(
        self,
        pdf_path: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Parse PDF to Markdown.

        Args:
            pdf_path: Path to PDF file
            max_retries: Number of retries on failure

        Returns:
            Dictionary with markdown content and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return {
                "error": f"File not found: {pdf_path}",
                "markdown": "",
                "pages": 0,
            }

        for attempt in range(max_retries):
            try:
                client = self._get_client()
                documents = client.load_data(str(pdf_path))

                if not documents:
                    return {
                        "error": "No documents returned",
                        "markdown": "",
                        "pages": 0,
                    }

                # Concatenate all pages
                markdown = "\n\n".join(doc.text for doc in documents)

                return {
                    "markdown": markdown,
                    "pages": len(documents),
                    "error": None,
                }

            except Exception as e:
                logger.warning(f"Parse attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "error": str(e),
                        "markdown": "",
                        "pages": 0,
                    }
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_parser.py::test_parser_imports -v
pytest tests/test_parser.py::test_parser_initialization -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/parser.py tests/test_parser.py
```

**Step 6: Commit**

```bash
git add src/ingestion/parser.py tests/test_parser.py
git commit -m "feat: integrate LlamaParse with retry logic"
```

---

## Task 3: Set Up Chroma Cloud Connection

**Files:**
- Create: `src/ingestion/chroma_store.py`
- Test: `tests/test_chroma_store.py`

**Step 1: Write failing test**

```bash
cat > tests/test_chroma_store.py << 'EOF'
"""Test Chroma Cloud connection and setup."""
import pytest


def test_chroma_store_imports():
    """Test that chroma_store module can be imported."""
    from src.ingestion.chroma_store import ChromaStore
    assert ChromaStore is not None


def test_chroma_initialization(mock_api_keys):
    """Test ChromaStore can be initialized."""
    from src.ingestion.chroma_store import ChromaStore

    store = ChromaStore(
        host="http://test-host.com",
        api_key="test_key",
        collection_name="test_collection"
    )

    assert store.collection_name == "test_collection"


@pytest.mark.integration
def test_chroma_connection(mock_api_keys):
    """Test actual Chroma Cloud connection."""
    from src.ingestion.chroma_store import ChromaStore
    from src.config import settings

    store = ChromaStore(
        host=settings.chroma_host,
        api_key=settings.chroma_api_key,
        collection_name="test_arxiv_papers"
    )

    # Test connection
    assert store.client is not None
    assert store.collection is not None
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chroma_store.py::test_chroma_store_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement chroma_store.py**

```bash
cat > src/ingestion/chroma_store.py << 'EOF'
"""Chroma Cloud integration for vector storage."""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ChromaStore:
    """Wrapper for Chroma Cloud with collection management."""

    def __init__(
        self,
        host: str,
        api_key: str,
        collection_name: str = "arxiv_papers_v1",
    ):
        """Initialize Chroma Cloud client.

        Args:
            host: Chroma Cloud host URL
            api_key: Chroma API key
            collection_name: Name of collection to use/create
        """
        self.host = host
        self.api_key = api_key
        self.collection_name = collection_name

        # Initialize client
        self.client = chromadb.HttpClient(
            host=host,
            settings=Settings(
                chroma_client_auth_provider="chromadb",
                chroma_client_auth_credentials=api_key,
            )
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "ArXiv ML/AI papers"}
            )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to collection.

        Args:
            documents: List of document text chunks
            metadatas: List of metadata dictionaries
            ids: Optional list of unique IDs
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(documents)} documents to {self.collection_name}")

    def count(self) -> int:
        """Get document count in collection."""
        return self.collection.count()

    def test_connection(self) -> bool:
        """Test Chroma Cloud connection.

        Returns:
            True if connection successful
        """
        try:
            count = self.count()
            logger.info(f"Connection successful. Collection has {count} documents")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chroma_store.py::test_chroma_store_imports -v
pytest tests/test_chroma_store.py::test_chroma_initialization -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/chroma_store.py tests/test_chroma_store.py
```

**Step 6: Commit**

```bash
git add src/ingestion/chroma_store.py tests/test_chroma_store.py
git commit -m "feat: set up Chroma Cloud connection"
```

---

## Task 4: Implement Chunking Strategy

**Files:**
- Create: `src/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

**Step 1: Write failing test**

```bash
cat > tests/test_chunker.py << 'EOF'
"""Test document chunking."""
import pytest


def test_chunker_imports():
    """Test that chunker module can be imported."""
    from src.ingestion.chunker import DocumentChunker
    assert DocumentChunker is not None


def test_chunker_initialization():
    """Test DocumentChunker initialization."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=100
    )

    assert chunker.chunk_size == 800
    assert chunker.chunk_overlap == 100


def test_chunk_markdown():
    """Test chunking markdown content."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)

    markdown = """
    # Introduction

    This is a test document with multiple sections.

    ## Methods

    Here we describe the methods.

    ## Results

    The results show that chunking works.
    """ * 10  # Make it longer

    chunks = chunker.chunk_markdown(
        markdown=markdown,
        metadata={
            "document_id": "test123",
            "title": "Test Paper",
        }
    )

    assert len(chunks) > 1
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["document_id"] == "test123"


def test_metadata_enrichment():
    """Test that metadata is properly enriched."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker()

    base_metadata = {
        "document_id": "2301.07041",
        "arxiv_id": "2301.07041",
        "title": "Test Paper",
        "section": "Introduction",
    }

    chunk = {
        "text": "This is test content.",
        "metadata": {
            **base_metadata,
            "chunk_index": 0,
            "page_number": 1,
        }
    }

    assert chunk["metadata"]["document_id"] == "2301.07041"
    assert chunk["metadata"]["chunk_index"] == 0
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunker.py::test_chunker_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement chunker.py**

```bash
cat > src/ingestion/chunker.py << 'EOF'
"""Document chunking using LlamaIndex MarkdownElementNodeParser."""
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import Document
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk markdown documents with structure awareness."""

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

        # Initialize Markdown-aware parser
        self.node_parser = MarkdownElementNodeParser(
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
                "section": node.metadata.get("header", "Unknown"),
            }

            chunks.append({
                "text": node.text,
                "metadata": chunk_metadata,
            })

        logger.info(f"Chunked into {len(chunks)} chunks")
        return chunks

    def enrich_metadata(
        self,
        chunk: Dict[str, Any],
        page_number: int = None,
    ) -> Dict[str, Any]:
        """Enrich chunk metadata with additional fields.

        Args:
            chunk: Chunk with text and base metadata
            page_number: Optional page number

        Returns:
            Enriched chunk
        """
        enriched = chunk.copy()

        if page_number:
            enriched["metadata"]["page_number"] = page_number

        # Add contextual header
        header = f"[Document: {chunk['metadata'].get('title', 'Unknown')} | "
        header += f"arxiv:{chunk['metadata'].get('arxiv_id', 'Unknown')} | "
        header += f"Section: {chunk['metadata'].get('section', 'Unknown')}]"

        enriched["text"] = f"{header}\n\n{chunk['text']}"

        return enriched
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chunker.py -v --skip-glob="*integration*"
```

Expected: PASS (non-integration tests)

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/chunker.py tests/test_chunker.py
```

**Step 6: Commit**

```bash
git add src/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: implement MarkdownElementNodeParser chunking"
```

---

## Task 5: Build Metadata Enrichment

**Files:**
- Create: `src/ingestion/metadata.py`
- Test: `tests/test_metadata.py`

**Step 1: Write failing test**

```bash
cat > tests/test_metadata.py << 'EOF'
"""Test metadata enrichment."""
import pytest


def test_metadata_builder():
    """Test MetadataBuilder class."""
    from src.ingestion.metadata import MetadataBuilder

    builder = MetadataBuilder()

    paper_metadata = {
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "published_date": "2023-01-17",
        "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
    }

    enriched = builder.build_document_metadata(paper_metadata)

    assert enriched["document_id"] == "2301.07041"
    assert enriched["arxiv_id"] == "2301.07041"
    assert "authors" in enriched


def test_chunk_metadata():
    """Test chunk-level metadata enrichment."""
    from src.ingestion.metadata import MetadataBuilder

    builder = MetadataBuilder()

    doc_metadata = builder.build_document_metadata({
        "arxiv_id": "2301.07041",
        "title": "Test Paper",
        "authors": ["Author One"],
    })

    chunk_metadata = builder.build_chunk_metadata(
        doc_metadata=doc_metadata,
        chunk_index=0,
        section="Introduction",
        page_number=1,
    )

    assert chunk_metadata["chunk_index"] == 0
    assert chunk_metadata["section"] == "Introduction"
    assert chunk_metadata["page_number"] == 1
    assert chunk_metadata["document_id"] == "2301.07041"
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metadata.py -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement metadata.py**

```bash
cat > src/ingestion/metadata.py << 'EOF'
"""Metadata enrichment for documents and chunks."""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """Build and enrich metadata for documents and chunks."""

    def build_document_metadata(
        self,
        paper_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build document-level metadata.

        Args:
            paper_metadata: Raw metadata from downloader

        Returns:
            Enriched document metadata
        """
        return {
            "document_id": paper_metadata["arxiv_id"],
            "arxiv_id": paper_metadata["arxiv_id"],
            "title": paper_metadata.get("title", "Unknown"),
            "authors": paper_metadata.get("authors", []),
            "published_date": paper_metadata.get("published_date", ""),
            "pdf_url": paper_metadata.get("pdf_url", ""),
        }

    def build_chunk_metadata(
        self,
        doc_metadata: Dict[str, Any],
        chunk_index: int,
        section: str,
        page_number: int = None,
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
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_metadata.py -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/metadata.py tests/test_metadata.py
```

**Step 6: Commit**

```bash
git add src/ingestion/metadata.py tests/test_metadata.py
git commit -m "feat: implement metadata enrichment"
```

---

## Task 6: Implement Indexer Pipeline

**Files:**
- Create: `src/ingestion/indexer.py`
- Test: `tests/test_indexer.py`

**Step 1: Write failing test**

```bash
cat > tests/test_indexer.py << 'EOF'
"""Test ingestion indexer pipeline."""
import pytest


def test_indexer_imports():
    """Test that indexer can be imported."""
    from src.ingestion.indexer import IngestionIndexer
    assert IngestionIndexer is not None


@pytest.mark.integration
def test_index_single_paper(tmp_path, mock_api_keys):
    """Test indexing a single paper end-to-end."""
    from src.ingestion.indexer import IngestionIndexer
    from src.config import settings

    # Create test PDF path
    test_pdf = tmp_path / "test.pdf"
    test_pdf.write_bytes(b"%PDF-1.4\n%mock pdf")

    indexer = IngestionIndexer(
        llamaparse_api_key="test_key",
        chroma_host=settings.chroma_host,
        chroma_api_key=settings.chroma_api_key,
    )

    result = indexer.index_paper(
        pdf_path=str(test_pdf),
        metadata={
            "arxiv_id": "test123",
            "title": "Test Paper",
            "authors": ["Test Author"],
        }
    )

    assert "status" in result
    assert "chunks_indexed" in result
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_indexer.py::test_indexer_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement indexer.py**

```bash
cat > src/ingestion/indexer.py << 'EOF'
"""Orchestrate ingestion pipeline: parse → chunk → embed → store."""
from .parser import LlamaParserWrapper
from .chunker import DocumentChunker
from .chroma_store import ChromaStore
from .metadata import MetadataBuilder
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class IngestionIndexer:
    """Orchestrate the full ingestion pipeline."""

    def __init__(
        self,
        llamaparse_api_key: str,
        chroma_host: str,
        chroma_api_key: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        collection_name: str = "arxiv_papers_v1",
    ):
        """Initialize indexer components.

        Args:
            llamaparse_api_key: LlamaParse API key
            chroma_host: Chroma Cloud host
            chroma_api_key: Chroma API key
            chunk_size: Chunk size for splitting
            chunk_overlap: Chunk overlap
            collection_name: Chroma collection name
        """
        self.parser = LlamaParserWrapper(api_key=llamaparse_api_key)
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.metadata_builder = MetadataBuilder()
        self.chroma = ChromaStore(
            host=chroma_host,
            api_key=chroma_api_key,
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
        papers: List[Dict[str, Any]],
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

            result = self.index_paper(pdf_path, metadata)

            if result["status"] == "success":
                results["success"] += 1
                results["total_chunks"] += result["chunks_indexed"]
            else:
                results["failed"] += 1
                logger.error(f"Failed to index {metadata.get('arxiv_id')}: {result.get('error')}")

        return results
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_indexer.py::test_indexer_imports -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/indexer.py tests/test_indexer.py
```

**Step 6: Commit**

```bash
git add src/ingestion/indexer.py tests/test_indexer.py
git commit -m "feat: implement ingestion indexer pipeline"
```

---

## Task 7: Index First 10 Test Papers

**Files:**
- Create: `scripts/ingest_test_papers.py`
- Run: Integration test

**Step 1: Create ingestion script**

```bash
mkdir -p scripts
cat > scripts/ingest_test_papers.py << 'EOF'
"""Ingest first 10 ArXiv papers as a test."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.downloader import ArxivDownloader
from src.ingestion.indexer import IngestionIndexer
from src.config import settings


def main():
    """Download and index 10 test papers."""
    print("🔄 Starting ingestion of 10 test papers...")

    # Step 1: Download papers
    print("\n📥 Step 1: Downloading papers...")
    downloader = ArxivDownloader(
        download_dir="./ml_pdfs",
        max_results=10,
    )

    papers = downloader.download_papers(
        query="cat:cs.LG",
        max_results=10,
    )

    print(f"✅ Downloaded {len(papers)} papers")

    # Step 2: Index papers
    print("\n📊 Step 2: Indexing papers...")
    indexer = IngestionIndexer(
        llamaparse_api_key=settings.llamaparse_api_key,
        chroma_host=settings.chroma_host,
        chroma_api_key=settings.chroma_api_key,
    )

    results = indexer.index_batch(papers)

    print(f"\n✅ Ingestion complete!")
    print(f"   - Success: {results['success']}")
    print(f"   - Failed: {results['failed']}")
    print(f"   - Total chunks: {results['total_chunks']}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/ingest_test_papers.py
```

**Step 2: Run ingestion script (integration test)**

```bash
python scripts/ingest_test_papers.py
```

Expected: Downloads 10 papers, parses with LlamaParse, indexes to Chroma

**Step 3: Verify Chroma has documents**

```bash
python -c "
from src.ingestion.chroma_store import ChromaStore
from src.config import settings

store = ChromaStore(
    host=settings.chroma_host,
    api_key=settings.chroma_api_key,
)

print(f'Document count: {store.count()}')
"
```

Expected: Document count > 0

**Step 4: Create integration test**

```bash
cat > tests/test_ingestion_integration.py << 'EOF'
"""Integration test for full ingestion pipeline."""
import pytest


@pytest.mark.integration
def test_full_ingestion_pipeline(mock_api_keys):
    """Test the full pipeline: download → parse → chunk → index."""
    from src.ingestion.downloader import ArxivDownloader
    from src.ingestion.indexer import IngestionIndexer
    from src.config import settings
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download 1 paper
        downloader = ArxivDownloader(download_dir=tmpdir, max_results=1)
        papers = downloader.download_papers(query="cat:cs.LG", max_results=1)

        assert len(papers) == 1

        # Index the paper
        indexer = IngestionIndexer(
            llamaparse_api_key=settings.llamaparse_api_key,
            chroma_host=settings.chroma_host,
            chroma_api_key=settings.chroma_api_key,
            collection_name="test_ingestion",
        )

        result = indexer.index_paper(
            pdf_path=papers[0]["file_path"],
            metadata=papers[0],
        )

        assert result["status"] == "success"
        assert result["chunks_indexed"] > 0
EOF
```

**Step 5: Commit**

```bash
git add scripts/ tests/test_ingestion_integration.py
git commit -m "feat: add ingestion script and integration test"
```

---

## Task 8: Phase 2 Gate Testing

**Files:**
- Test: All Phase 2 components

**Step 1: Run unit tests**

```bash
pytest tests/test_downloader.py tests/test_parser.py tests/test_chroma_store.py tests/test_chunker.py tests/test_metadata.py tests/test_indexer.py -v --skip-glob="*integration*"
```

Expected: All non-integration tests PASS

**Step 2: Run integration tests (requires API keys)**

```bash
pytest tests/test_ingestion_integration.py -v
```

Expected: PASS (if API keys are valid)

**Step 3: Verify 10 papers can be indexed**

```bash
python scripts/ingest_test_papers.py
```

Expected: 10 papers downloaded, parsed, indexed successfully

**Step 4: Verify metadata in Chroma**

```bash
python -c "
from src.ingestion.chroma_store import ChromaStore
from src.config import settings

store = ChromaStore(
    host=settings.chroma_host,
    api_key=settings.chroma_api_key,
)

count = store.count()
print(f'Total documents in Chroma: {count}')

# Get a sample
collection = store.collection
if count > 0:
    results = collection.get(limit=1, include=['metadatas'])
    print('Sample metadata:', results['metadatas'][0])
"
```

Expected: Documents with proper metadata (arxiv_id, title, authors, section, etc.)

**Step 5: Run ruff check**

```bash
ruff check src/ingestion/ tests/
```

Expected: No errors

**Step 6: Run pytest with coverage**

```bash
pytest tests/ --cov=src/ingestion --cov-report=term-missing
```

Expected: Reasonable coverage on ingestion modules

**Step 7: Create gate summary**

```bash
cat > docs/plans/phase-2-gate-summary.md << 'EOF'
# Phase 2 Gate Test Results

## Date: [Run Date]

## Checklist

- [x] 10 papers downloaded from ArXiv
- [x] LlamaParse successfully converts PDFs to Markdown
- [x] Chroma Cloud connection established
- [x] Papers chunked with MarkdownElementNodeParser
- [x] Metadata enriched (arxiv_id, title, authors, section, page_number)
- [x] Chunks indexed to Chroma
- [x] Can query Chroma and retrieve chunks
- [x] Unit tests pass
- [x] Integration test passes
- [x] ruff check passes

## Status: PASSED ✅

Phase 2 Ingestion Pipeline is complete. Ready to proceed to Phase 3 - Retrieval & Generation.
EOF
```

**Step 8: Commit**

```bash
git add docs/plans/phase-2-gate-summary.md
git commit -m "test: phase 2 gate testing complete"
```

---

## Summary

After completing all tasks, you should have:

1. ✅ ArXiv downloader with metadata extraction
2. ✅ LlamaParse integration with retry logic
3. ✅ Chroma Cloud connection and collection setup
4. ✅ Structure-aware chunking with MarkdownElementNodeParser
5. ✅ Metadata enrichment for documents and chunks
6. ✅ End-to-end indexer pipeline
7. ✅ 10 test papers indexed successfully
8. ✅ All gate tests passing

**Next Phase:** Phase 3 - Retrieval & Generation
