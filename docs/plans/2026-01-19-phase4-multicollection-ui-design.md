# Phase 4: Multi-Collection RAG UI Design

**Goal:** Update query engine and UI to support both ArXiv papers and SEC contracts with unified interface.

**Date:** 2026-01-19
**Status:** Design Approved

---

## Architecture Overview

The design adds collection selection at three layers:

1. **UI Layer (Gradio)**: Dropdown selector for `["ArXiv Papers", "SEC Contracts"]`
2. **Engine Layer (RAGQueryEngine)**: Per-query collection selection via `query_with_collection()`
3. **Citation Layer**: `CitationFormatter` class with collection-specific formatting

### Data Flow
```
User selects collection → Gradio passes selection → RAGQueryEngine queries that collection → Citations formatted based on collection type
```

---

## Component Implementation

### 1. CitationFormatter Class

**File:** `src/retrieval/citation_formatter.py`

```python
class CitationFormatter:
    """Format citations for different collection types."""

    def format_citation(self, metadata: dict) -> str:
        """Format citation based on collection type detection."""

    def _format_arxiv_paper(self, metadata: dict) -> str:
        """Format: 'Title - Authors (arxiv:id)' with clickable link to arxiv.org"""

    def _format_contract(self, metadata: dict) -> str:
        """Format: 'Company [Contract Type] executed on Date (Filing: X Exhibit: Y)'"""
```

**Detection Logic:**
- Papers: `arxiv_id` present
- Contracts: `document_id` present

---

### 2. RAGQueryEngine Updates

**File:** `src/retrieval/engine.py`

**New Method:**
```python
def query_with_collection(
    self,
    question: str,
    collection_name: str,
    use_rerank: bool = True,
) -> dict[str, Any]:
    """Query a specific collection by name.

    Args:
        question: User question
        collection_name: "arxiv_papers_v1" or "contracts"
        use_rerank: Whether to use reranker

    Returns:
        Result dict with answer and citations
    """
```

**Updated `_extract_citations`:**
- Import and use `CitationFormatter`
- Use `arxiv_id` or `document_id` for deduplication key

---

### 3. Gradio UI Updates

**File:** `src/ui/gradio_app.py`

**New UI Elements:**
- `gr.Dropdown`: Collection selector at top of interface
- `gr.Markdown`: Dynamic description of current collection

**Updated Function Signature:**
```python
def query_paper_stream(
    message: str,
    history: list[tuple[str, str]],
    collection_choice: str,  # New parameter
):
```

**Collection Mapping:**
```python
collection_map = {
    "ArXiv Papers": "arxiv_papers_v1",
    "SEC Contracts": "contracts"
}
```

**Unified Branding:**
- Title: `"📚 RAG Research Assistant"`
- Description: Mentions both papers and contracts

**Example Queries:**
- Papers: "What is attention mechanism?"
- Contracts: "co-branding agreement terms"

---

## Error Handling

### Collection-Specific Messages
```python
if not retrieved:
    if collection_choice == "SEC Contracts":
        yield "🔍 No relevant contract clauses found. Try different keywords."
    else:
        yield "🔍 No relevant information found in the papers."
```

### Invalid Collection Handling
```python
valid_collections = ["arxiv_papers_v1", "contracts"]
if collection_name not in valid_collections:
    raise ValueError(f"Invalid collection: {collection_name}")
```

### Missing Metadata Fallbacks
- `CitationFormatter` uses `"Unknown <field>"` for missing values
- Gracefully handles incomplete metadata

---

## Testing Strategy

### Unit Tests (`tests/test_citation_formatter.py`)
- `_format_arxiv_paper` with complete/missing metadata
- `_format_contract` with complete/missing metadata
- `format_citation` auto-detection logic

### Integration Tests (`tests/test_rag_engine_multicollection.py`)
- `query_with_collection` for papers collection
- `query_with_collection` for contracts collection
- Invalid collection raises ValueError
- Citation extraction per collection type

### Manual UI Testing
- Dropdown functionality
- Both collections return results
- Citations format correctly

---

## Implementation Order

1. **CitationFormatter** (class + tests)
2. **RAGQueryEngine** (`query_with_collection` + tests)
3. **Update `_extract_citations`** to use CitationFormatter
4. **Gradio UI** (dropdown + collection choice)
5. **Update branding** (title/description)
6. **Manual testing**

---

## Success Criteria

- ✅ Dropdown selector visible and functional
- ✅ Queries work for both ArXiv papers and SEC contracts
- ✅ Citations format correctly per collection type
- ✅ Unified branding applied
- ✅ All tests pass
- ✅ Manual testing confirms both collections work
