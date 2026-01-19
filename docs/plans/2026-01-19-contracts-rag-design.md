# Contracts RAG System - Design Document

**Date**: 2026-01-19
**Project**: ChromaDB_POC
**Status**: Design Phase

## Overview

Extend the existing ArXiv RAG system to support contract PDF queries with metadata filtering. Contracts from `full_contract_pdf/` will be indexed in a separate ChromaDB collection with zero-cost metadata extraction from filenames and folder structure.

### Goals

- Create a new ChromaDB collection for contracts (separate from ArXiv papers)
- Extract metadata from filenames and folder structure (no LLM cost)
- Support metadata filtering by company, contract type, filing type, date range
- Provide unified Gradio interface with collection selector

### Non-Goals (POC Scope)

- LLM-based content extraction (party names, clauses, etc.)
- Manual metadata tagging
- Contract comparison or analysis features
- Full contract document display

## Architecture

### Two-Source Metadata Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   SOURCE 1: FILESYSTEM METADATA                         │
│                                                                         │
│  full_contract_pdf/Part_I/Affiliate_Agreements/contract.pdf            │
│       ├─> contract_type: "Affiliate_Agreements"                        │
│       └─> file_path: full path for traceability                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   SOURCE 2: FILENAME PARSING                            │
│                                                                         │
│  CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate.pdf  │
│       ├─> company_name: "CreditcardscomInc"                            │
│       ├─> execution_date: "2007-08-10"                                 │
│       ├─> filing_type: "S-1"                                           │
│       ├─> exhibit_number: "EX-10.33"                                   │
│       ├─> accession_number: "362297"                                   │
│       └─> description: "Affiliate Agreement"                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Parsing** | LlamaParse (existing) | PDF → Markdown |
| **Vector DB** | ChromaDB Cloud | Separate collection for contracts |
| **Embeddings** | BAAI/bge-large-en-v1.5 (existing) | Vector search |
| **Reranker** | Cohere Rerank v3 (existing) | Result filtering |
| **LLM** | Gemini 2.5 Flash Lite (existing) | Answer generation |
| **UI** | Gradio (existing) | Unified interface |

## Module Structure

```
src/ingestion/
├── contract_filename_parser.py  # NEW: Extract metadata from filenames
├── contract_indexer.py           # NEW: Orchestrate contract ingestion
├── chroma_store.py               # UPDATE: Support multiple collections
├── parser.py                     # EXISTING: Reuse for PDF parsing
├── chunker.py                    # EXISTING: Reuse for chunking
└── ...

scripts/
├── ingest_contracts.py           # NEW: Contract ingestion script
└── ...

src/retrieval/
├── engine.py                     # UPDATE: Support multiple collections
└── ...

src/ui/
├── gradio_app.py                 # UPDATE: Add collection selector
└── ...
```

## Metadata Schema

### Contract Collection Metadata

```python
{
    # Document Identity
    "document_id": str,           # Unique per contract (hash of filename)
    "filename": str,              # Original filename

    # Filesystem Metadata
    "contract_type": str,         # e.g., "Affiliate_Agreements", "License_Agreements"
    "file_path": str,             # Full path for traceability

    # Parsed Filename Metadata
    "company_name": str,          # Extracted from filename start
    "execution_date": str,        # ISO format (YYYY-MM-DD), parsed from filename
    "filing_type": str,           # e.g., "S-1", "10-K", "8-K"
    "exhibit_number": str,        # e.g., "EX-10.33"
    "accession_number": str,      # e.g., "362297"

    # Ingestion Metadata
    "ingested_at": str,           # ISO timestamp of when contract was indexed
    "chunk_index": int,           # Position within document
    "page_number": int,           # From LlamaParse

    # Content Type
    "content_type": str,          # Always "contract" (for future multi-collection use)
}
```

### Filename Parsing Pattern

```python
# Example: CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf
# Pattern: {Company}_{Date}_{FilingType}_{Exhibit}_{AccessionNum}_{Description}.pdf

import re

pattern = r'^([^_]+)_(\d{8})_([A-Z0-9-]+)_EX-([\d.]+)_(\d+)_EX-[\d.]+_(.+)\.pdf$'

# Groups:
# 1: company_name
# 2: execution_date (YYYYMMDD -> convert to YYYY-MM-DD)
# 3: filing_type
# 4: exhibit_number
# 5: accession_number
# 6: description (used for validation)
```

## Ingestion Pipeline

### Flow

```python
# scripts/ingest_contracts.py

def ingest_contracts(pdf_dir: str = "./full_contract_pdf"):
    """Ingest all contract PDFs into ChromaDB contracts collection"""

    for pdf_file in scan_directory(pdf_dir):
        # 1. Parse filename
        metadata = contract_filename_parser.parse(pdf_file)

        # 2. Check if already ingested
        if chroma_store.paper_exists(metadata["document_id"], collection="contracts"):
            continue

        # 3. Parse PDF (reuse existing LlamaParse wrapper)
        markdown_content = parser.parse_pdf(pdf_file)

        # 4. Chunk document (reuse existing chunker)
        chunks = chunker.chunk(markdown_content)

        # 5. Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": metadata["document_id"],
                "chunk_index": i,
                "ingested_at": datetime.now().isoformat(),
                **metadata  # Include all parsed metadata
            })

        # 6. Index in ChromaDB
        chroma_store.add_documents(
            collection="contracts",
            documents=chunks,
            embeddings=generate_embeddings(chunks)
        )
```

### Error Handling

| Error Type | Handling |
|------------|----------|
| Invalid filename format | Log warning, skip file, track in `failed_filenames.json` |
| Parse fails (LlamaParse) | Retry 2x, then skip. Track in `failed_parses.json` |
| Duplicate document_id | Skip silently (idempotent) |
| ChromaDB error | Log error, continue with next file |

## Retrieval & Query

### Query Engine Updates

```python
class RAGQueryEngine:
    """Extend existing engine to support multiple collections"""

    def query(self, question: str, collection: str = "arxiv_papers", filters: dict = None):
        """
        Args:
            collection: "arxiv_papers" or "contracts"
            filters: Optional metadata filters
                For contracts: {
                    "company_name": "CreditcardscomInc",
                    "contract_type": "Affiliate_Agreements",
                    "filing_type": "10-K",
                    "execution_date": {"gte": "2020-01-01", "lte": "2025-12-31"}
                }
        """
        # 1. Get collection
        chroma_collection = chroma_store.get_collection(collection)

        # 2. Hybrid search with metadata filters
        results = hybrid_search.query(
            question=question,
            collection=chroma_collection,
            filters=filters,
            top_k=50
        )

        # 3. Rerank (reuse existing reranker)
        reranked = reranker.rerank(question, results, top_k=5)

        # 4. Generate answer (reuse existing LLM)
        answer = llm.generate(question, reranked)

        return answer, citations
```

### Metadata Filter Examples

```python
# Filter by company
filters = {"company_name": "CreditcardscomInc"}

# Filter by contract type
filters = {"contract_type": "License_Agreements"}

# Filter by date range
filters = {"execution_date": {"$gte": "2020-01-01", "$lte": "2025-12-31"}}

# Combine filters
filters = {
    "company_name": "CreditcardscomInc",
    "contract_type": "Affiliate_Agreements",
    "filing_type": "10-K"
}
```

## UI Design

### Unified Gradio Interface

```python
# src/ui/gradio_app.py

def create_interface():
    with gr.Blocks(theme=theme) as interface:
        # Collection Selector
        collection = gr.Radio(
            choices=["ArXiv Papers", "Contracts"],
            value="ArXiv Papers",
            label="Search Collection"
        )

        # Contract Filters (hidden by default)
        with gr.Group(visible=False) as contract_filters:
            gr.Markdown("### Filter Contracts")
            company_name = gr.Dropdown(
                choices=[],  # Populated dynamically from unique values
                allow_custom_value=True,
                label="Company"
            )
            contract_type = gr.Dropdown(
                choices=[],  # Populated dynamically
                label="Contract Type"
            )
            filing_type = gr.Dropdown(
                choices=["S-1", "10-K", "10-Q", "8-K"],
                label="Filing Type"
            )

        # Shared Components
        question = gr.Textbox(label="Your Question")
        submit_btn = gr.Button("Ask")
        answer = gr.Markdown()
        citations = gr.JSON(label="Sources")

        # Event Handlers
        collection.change(
            fn=toggle_contract_filters,
            inputs=collection,
            outputs=contract_filters
        )

        submit_btn.click(
            fn=query_paper_stream if collection == "ArXiv Papers" else query_contract_stream,
            inputs=[question, collection, company_name, contract_type, filing_type],
            outputs=[answer, citations]
        )

def toggle_contract_filters(collection):
    """Show contract filters only when Contracts is selected"""
    return gr.Group(visible=(collection == "Contracts"))
```

## Configuration

### `config.toml` Updates

```toml
[contracts]
enabled = true
pdf_dir = "./full_contract_pdf"
collection_name = "contracts"

[contracts.metadata]
# Available contract types (for filter dropdown)
contract_types = [
    "Affiliate_Agreements",
    "Agency_Agreements",
    "Collaboration",
    "Consulting_Agreements",
    "Co_Branding",
    "Development",
    "Distributor",
    "Endorsement",
    "Franchise",
    "Hosting",
    "IP",
    "Joint_Venture",
    "License_Agreements",
    "Maintenance",
    "Manufacturing",
    "Marketing",
    "Non_Compete_Non_Solicit",
    "Outsourcing",
    "Promotion",
    "Reseller",
    "Service",
    "Sponsorship",
    "Supply",
    "Transportation"
]

# Available filing types
filing_types = ["S-1", "10-K", "10-Q", "8-K", "8-KA", "SB-2", "SB-2A", "10-12G"]
```

## Implementation Phases

### Phase 1: Filename Parser & Metadata Extraction

**Tasks:**
1. Create `src/ingestion/contract_filename_parser.py`
2. Implement regex pattern for filename parsing
3. Add unit tests for various filename formats
4. Handle edge cases (missing fields, non-standard formats)

**Phase Gate Criteria:**
- [ ] Parses 100% of sample filenames correctly
- [ ] Returns normalized metadata dict
- [ ] Unit tests pass: `pytest tests/test_contract_filename_parser.py -v`

**Deliverable:** Working filename parser with test coverage

---

### Phase 2: ChromaStore Multi-Collection Support

**Tasks:**
1. Update `src/ingestion/chroma_store.py` to support dynamic collection names
2. Add `get_collection(name: str)` method
3. Ensure backward compatibility with existing "arxiv_papers" collection
4. Add unit tests for multi-collection operations

**Phase Gate Criteria:**
- [ ] Can create/access multiple collections
- [ ] Existing ArXiv collection still works
- [ ] Unit tests pass: `pytest tests/test_chroma_store.py -v`

**Deliverable:** ChromaStore with multi-collection support

---

### Phase 3: Contract Ingestion Pipeline

**Tasks:**
1. Create `scripts/ingest_contracts.py`
2. Create `src/ingestion/contract_indexer.py`
3. Scan `full_contract_pdf/` directory
4. Parse PDFs with existing LlamaParse wrapper
5. Chunk with existing chunker
6. Index to ChromaDB "contracts" collection
7. Add logging and error tracking

**Phase Gate Criteria:**
- [ ] Ingests all contracts without errors
- [ ] Metadata is correctly stored
- [ ] Can query contracts collection
- [ ] Manual test: Query returns relevant results

**Test Command:**
```bash
python scripts/ingest_contracts.py --count 10  # Test with 10 contracts first
```

**Deliverable:** Working contract ingestion pipeline

---

### Phase 4: Query Engine Updates

**Tasks:**
1. Update `src/retrieval/engine.py` to support collection parameter
2. Add metadata filter support
3. Update `src/ui/gradio_app.py` with collection selector
4. Add contract filter panel
5. Implement toggle logic for filter visibility

**Phase Gate Criteria:**
- [ ] Can query both ArXiv and contracts
- [ ] Metadata filters work correctly
- [ ] UI switches between modes smoothly
- [ ] Manual test: Query contracts with filters returns filtered results

**Test Command:**
```bash
python src/main.py  # Launch Gradio and test both collections
```

**Deliverable:** Unified RAG interface for ArXiv + Contracts

---

### Phase 5: Polish & Testing

**Tasks:**
1. Populate filter dropdowns dynamically from actual data
2. Add error messages for failed queries
3. Performance testing (query speed)
4. Full test suite run
5. Documentation updates

**Phase Gate Criteria:**
- [ ] All filters work correctly
- [ ] Query response time < 30 seconds
- [ ] Full test suite passes: `pytest`
- [ ] Manual quality test: 5 contract queries return relevant results

**Deliverable:** Production-ready contracts RAG system

---

## Testing Strategy

### Unit Tests

```python
tests/test_contract_filename_parser.py:
  - test_parse_standard_filename()
  - test_parse_with_underscores_in_company()
  - test_parse_with_hyphenated_filing_type()
  - test_parse_invalid_format_returns_none()
  - test_date_conversion_to_iso_format()

tests/test_contract_indexer.py:
  - test_ingest_single_contract()
  - test_metadata_enrichment()
  - test_duplicate_skip()

tests/test_chroma_store.py:
  - test_create_contracts_collection()
  - test_add_to_contracts_collection()
  - test_query_with_metadata_filters()
```

### Manual Quality Tests

```
Contract Queries:
1. "What are the termination clauses in CreditcardscomInc contracts?"
2. "Show me all affiliate agreements from 2020-2025"
3. "What payment terms are in the License_Agreements?"
4. "Find contracts with 10-K filing types"
5. "What obligations does the distributor have?"

Criteria:
- Answers include relevant citations
- Filters correctly narrow results
- No obvious hallucinations
- Response time < 30 seconds
```

## Dependencies

### No New Dependencies Required

All existing dependencies support this feature:
- `llama-parse` - PDF parsing (existing)
- `chromadb` - Multi-collection support (existing)
- `llama-index` - Query engine (existing)
- `gradio` - UI components (existing)
- `python-dateutil` - Date parsing (if not already included)

## Success Criteria

The POC is considered successful when:

- [ ] All contracts from `full_contract_pdf/` are indexed
- [ ] Metadata filters work for company, contract_type, filing_type, date_range
- [ ] Unified UI allows switching between ArXiv and Contracts
- [ ] Contract queries return relevant answers with citations
- [ ] Query response time < 30 seconds
- [ ] Full test suite passes

## Future Enhancements (Post-POC)

1. **LLM-Based Extraction**: Extract party names, clauses, key terms with LLM
2. **Contract Comparison**: Compare similar contracts across companies
3. **Advanced Filtering**: Filter by clause types, jurisdictions, values
4. **Document OCR**: Handle scanned/image-based PDFs
5. **Contract Analytics**: Visualize trends across contract portfolio
6. **Export**: Export filtered results as PDF/CSV

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Gradio Documentation](https://www.gradio.app/docs)
- Existing ArXiv RAG Design: `docs/plans/2026-01-13-arxiv-rag-chromadb-design.md`
