# ArXiv RAG System - Design Document

**Date**: 2026-01-13
**Project**: ChromaDB POC
**Status**: Design Phase

## Overview

A proof-of-concept document Q&A system built over ArXiv ML/AI papers using a modern RAG (Retrieval-Augmented Generation) pipeline with hybrid search, reranking, and citation support.

### Goals

- Demonstrate high-quality retrieval using hybrid search (vector + BM25) + Cohere Rerank v3
- Provide accurate answers with proper citations to source papers
- Build a simple Gradio chat interface for user interaction
- Establish patterns for future full-document retrieval (Phase 5 epic)

### Non-Goals (POC Scope)

- Production-scale orchestration (retries, queuing, scheduling)
- Automated evaluation suite (RAGAS/TruLens)
- Observability dashboard (Langfuse/Phoenix)
- Real-time data refresh from ArXiv
- Multi-user authentication

## Architecture

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: INGESTION                             │
│                                                                         │
│  ArXiv PDFs → LlamaParse → LlamaIndex → Embeddings → Chroma Cloud      │
│  (Markdown)    (Chunking)     (bge-large)         (Metadata Store)     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          STAGE 2: RETRIEVAL                             │
│                                                                         │
│  User Query → Hybrid Search → Top 50 → Cohere Rerank → Top 5           │
│              (Vector + BM25)        (Re-score)      (Filtered)         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          STAGE 3: GENERATION                            │
│                                                                         │
│  Query + Top 5 Chunks → OpenRouter LLM → Answer + Citations            │
│                        (Claude 3.5 / GPT-4o)   (Gradio UI)             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Parsing** | LlamaParse (Agentic Tier) | PDF → Markdown with table/structure preservation |
| **Orchestration** | LlamaIndex | Pipeline management, retrieval, LLM integration |
| **Vector DB** | Chroma Cloud | Serverless hybrid search (vector + BM25) |
| **Embeddings** | BAAI/bge-large-en-v1.5 | High-quality open-source embeddings |
| **Reranker** | Cohere Rerank v3 | Cross-encoder relevance filtering |
| **LLM** | OpenRouter (Claude 3.5 / GPT-4o) | Answer generation |
| **UI** | Gradio | Chat interface with citations |

### Why This Stack Works

- **Hybrid Search**: Catches both semantic meaning AND exact keywords (formulas, citations)
- **Reranking**: Eliminates irrelevant chunks before they reach the LLM (top 50 → top 5)
- **LlamaParse**: Solves the #1 RAG failure point upfront - poor PDF parsing
- **Modular Design**: Each component can be swapped or upgraded independently

## Module Structure

```
ChromaDB_POC/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Gradio app entry point
│   ├── config.py               # Environment + config.toml loader
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── downloader.py       # ArXiv API integration
│   │   ├── parser.py           # LlamaParse wrapper
│   │   └── indexer.py          # Chunking + Chroma ingestion
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── engine.py           # LlamaIndex query engine
│   │   └── hybrid_search.py    # Hybrid search configuration
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py              # OpenRouter LLM setup
│   └── ui/
│       ├── __init__.py
│       └── gradio_app.py       # Chat interface with citations
├── ml_pdfs/                    # Downloaded ArXiv PDFs
├── data/
│   └── chroma_state/          # Local cache (if needed)
├── logs/                       # Ingestion & retrieval logs
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_ui.py
├── docs/
│   └── plans/
│       └── 2026-01-13-arxiv-rag-chromadb-design.md
├── .env                        # API keys (gitignored)
├── .env.example                # Template (committed)
├── config.toml                 # App settings (committed)
├── requirements.txt
├── pyproject.toml              # Ruff + pytest config
└── README.md
```

## Data Flow & Metadata Schema

### Ingestion Flow

```python
# 1. Download
downloader.py:
  - Query arxiv API (cat:cs.LG OR cat:cs.AI)
  - Download PDFs to ml_pdfs/
  - Extract: arxiv_id, title, authors, published_date, pdf_url
  - Save metadata.json alongside PDF

# 2. Parse
parser.py (LlamaParse):
  - Input: PDF file
  - Output: Markdown with preserved tables/structure
  - Extract: page_numbers from LlamaParse response

# 3. Chunk & Index
indexer.py:
  - Use MarkdownElementNodeParser for structure-aware chunking
  - Chunk size: ~500-1000 tokens (preserves sections, tables)
  - Generate embeddings via BAAI/bge-large-en-v1.5
  - Store in Chroma Cloud with metadata
```

### Metadata Schema (Chroma)

```python
{
    "document_id": str,      # Unique per paper (arxiv_id)
    "arxiv_id": str,         # e.g., "2301.07041"
    "title": str,            # Paper title
    "authors": list[str],    # Author names
    "published_date": str,   # ISO format
    "pdf_url": str,          # ArXiv PDF URL
    "page_number": int,      # Page from LlamaParse
    "section": str,          # e.g., "Introduction", "Methods"
    "chunk_index": int,      # Position within document
}
```

### Query Flow

```python
# User question → Gradio → engine.py

1. Hybrid search: vector + BM25 → retrieve top 50 chunks
2. Cohere Rerank v3: re-score → filter to top 5
3. Build context: format chunks with citation metadata
4. LLM call: OpenRouter with question + context
5. Parse response: extract answer + citation markers
6. Format citations: [Title (arxiv_id), page X]
```

### Contextual Chunking

Each chunk gets prepended with provenance:

```
[Document: Attention Is All You Need | arxiv:1706.03762 | Section: 3.2]
<original chunk content>
```

This ensures the LLM knows the source of every piece of text.

## Error Handling

### Ingestion Pipeline

| Error Type | Handling |
|------------|----------|
| Download fails | Log error, skip paper, track in `failed_downloads.json` |
| Parse fails | Retry 3x with exponential backoff, then skip. Track in `failed_parses.json` |
| Empty content | Skip PDF if parsed Markdown < 100 characters |
| Invalid metadata | Use defaults ("Unknown" for missing authors) |

### Retrieval Pipeline

| Error Type | Handling |
|------------|----------|
| Chroma timeout | Retry 2x, return cached if available |
| Reranker API fails | Fall back to top 5 from hybrid search only |
| LLM API fails | Retry 3x, show "Unable to generate answer, here are sources" |
| No results | Return "No relevant papers found. Try rephrasing." |

### Cost Controls

```python
MAX_PAPERS_TO_PARSE = 200  # Start with subset
LLM_MAX_TOKENS = 1000      # Limit response length
RERANK_TOP_K = 5           # Don't rerank more than needed
```

## Configuration

### `.env` (gitignored)

```bash
# API Keys
LLAMAPARSE_API_KEY=your_key_here
CHROMA_HOST=your_chroma_cloud_host
CHROMA_API_KEY=your_api_key
OPENROUTER_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

### `config.toml` (committed)

```toml
[models]
llm = "anthropic/claude-3.5-sonnet"
embedding = "BAAI/bge-large-en-v1.5"
embedding_provider = "openrouter"

[pipeline]
max_papers = 200
chunk_size = 800
chunk_overlap = 100
top_k_retrieve = 50
top_k_rerank = 5

[paths]
pdf_dir = "./ml_pdfs"
log_dir = "./logs"
```

### `src/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys (from .env)
    llamaparse_api_key: str
    chroma_host: str
    chroma_api_key: str
    openrouter_api_key: str
    cohere_api_key: str

    class Config:
        env_file = ".env"
```

## Testing Strategy

### Unit Tests (pytest)

```python
tests/test_ingestion.py:
  - test_arxiv_downloader(): Mock API, verify metadata extraction
  - test_metadata_schema(): Validate required fields
  - test_chunk_size(): Verify 500-1000 token range

tests/test_retrieval.py:
  - test_hybrid_search(): Verify vector+BM25 both called
  - test_reranker(): Verify top 5 returned
  - test_citation_formatting(): Verify citation format

tests/test_ui.py:
  - test_input_validation(): Max/min length enforcement
```

### Manual Quality Tests

```
Test Questions:
1. "What is the attention mechanism in transformers?"
2. "How does BatchNorm work?"
3. "What are the limitations of BERT?"
4. "Explain the backpropagation algorithm"
5. "What is the difference between SGD and Adam?"

Criteria:
- Answers include relevant citations
- Citations link to correct papers
- No obvious hallucinations
- Response time < 30 seconds
```

### Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*"]
```

## Implementation Phases

### Phase 1: Foundation

**Tasks:**
1. Create project structure (directories, requirements.txt, pyproject.toml)
2. Configure environment (.env + config.toml)
3. Implement `config.py` with pydantic validation
4. Set up basic test framework

**Phase Gate Criteria:**
- [ ] `pytest` runs successfully with placeholder tests
- [ ] `ruff check .` passes with no errors
- [ ] `python -c "from src.config import settings"` loads config without errors
- [ ] All API keys can be loaded from `.env`

**Deliverable:** Working project skeleton with validated configuration

---

### Phase 2: Ingestion Pipeline

**Tasks:**
1. Implement ArXiv downloader (`downloader.py`)
2. Integrate LlamaParse (`parser.py`)
3. Set up Chroma Cloud connection
4. Implement chunking with `MarkdownElementNodeParser`
5. Build indexer (`indexer.py`)

**Phase Gate Criteria:**
- [ ] Can download 10 test papers from ArXiv
- [ ] LlamaParse successfully converts PDFs to Markdown
- [ ] Chroma Cloud connection established
- [ ] 10 papers indexed with valid metadata
- [ ] Can query Chroma and retrieve chunks
- [ ] Unit tests for ingestion pass: `pytest tests/test_ingestion.py -v`

**Test Command:**
```bash
python -m src.ingestion.downloader --count 10
python -m src.ingestion.indexer --count 10
```

**Deliverable:** Working ingestion pipeline with 10 indexed papers

---

### Phase 3: Retrieval & Generation

**Tasks:**
1. Implement hybrid search (vector + BM25)
2. Integrate Cohere Rerank v3
3. Set up OpenRouter LLM with prompt template
4. Build end-to-end query engine
5. Add citation formatting

**Phase Gate Criteria:**
- [ ] Hybrid search returns relevant chunks for test queries
- [ ] Reranker successfully filters to top 5
- [ ] LLM generates answers with citations
- [ ] Citations format: [Title (arxiv_id), page X]
- [ ] Unit tests for retrieval pass: `pytest tests/test_retrieval.py -v`
- [ ] Manual quality test: 5 questions answered satisfactorily

**Test Command:**
```bash
python -c "
from src.retrieval.engine import QueryEngine
engine = QueryEngine()
result = engine.query('What is attention in transformers?')
print(result.answer)
print(result.citations)
"
```

**Deliverable:** End-to-end RAG pipeline working via Python API

---

### Phase 4: UI & Polish

**Tasks:**
1. Build Gradio chat interface (`gradio_app.py`)
2. Add citation display with clickable ArXiv links
3. Add user-facing error messages
4. Scale to 100-200 papers
5. Final quality testing

**Phase Gate Criteria:**
- [ ] Gradio app launches without errors
- [ ] Chat interface accepts questions and returns answers
- [ ] Citations are clickable links to ArXiv
- [ ] Error messages are user-friendly
- [ ] 100+ papers indexed successfully
- [ ] Manual quality tests pass (all 5 test questions)
- [ ] Response time < 30 seconds per query
- [ ] Full test suite passes: `pytest`

**Test Command:**
```bash
python src/main.py  # Launches Gradio on http://localhost:7860
```

**Deliverable:** Working RAG chat application ready for demo

---

### Phase 5: (Future Epic) Full Document Retrieval

**Out of scope for POC - planned as follow-up work**

**Tasks:**
1. Implement metadata filtering by `document_id`
2. Add "Show full document" button to UI
3. Build document summarization mode

**Phase Gate Criteria (for future):**
- [ ] Can retrieve all chunks for a specific paper
- [ ] UI button displays full document
- [ ] Summarization generates concise paper overviews

---

## Dependencies

### requirements.txt

```
# Core
llama-index>=0.10.0
llama-index-llms-openrouter>=0.1.0
llama-index-vector-stores-chroma>=0.1.0
llama-index-embeddings-openrouter>=0.1.0
llama-index-node-parser>=0.1.0
llama-parse>=0.4.0

# Databases & APIs
chromadb>=0.4.0
cohere>=5.0.0
arxiv>=2.0.0

# UI
gradio>=4.0.0

# Config
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Development
pytest>=7.0.0
pytest-cov>=4.0.0
ruff>=0.1.0
```

### pyproject.toml

```toml
[project]
name = "chromadb-poc"
version = "0.1.0"
requires-python = ">=3.10"

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*"]
```

## Success Criteria

The POC is considered successful when:

- [ ] All 4 phases complete with phase gates passed
- [ ] 100-200 ArXiv papers indexed
- [ ] Returns relevant answers with proper citations
- [ ] Gradio UI is functional and user-friendly
- [ ] Response time < 30 seconds per query
- [ ] No obvious hallucinations in test answers
- [ ] Design document and code are committed to git

## Future Enhancements (Post-POC)

1. **Ingestion Orchestration**: Celery for background processing, retries, scheduling
2. **Evaluation Suite**: RAGAS/TruLens for automated quality metrics
3. **Observability**: Langfuse/Phoenix for tracing and monitoring
4. **Data Refresh**: Scheduled ArXiv updates, versioned indexes
5. **Full Document Mode**: Phase 5 epic implementation
6. **Multi-user**: Authentication, per-user history
7. **Advanced Reranking**: Citation verification, attribution scoring

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Cohere Rerank](https://docs.cohere.com/reference/rerank)
- [Gradio Documentation](https://www.gradio.app/docs)
- [ArXiv API](https://arxiv.org/help/api/)
