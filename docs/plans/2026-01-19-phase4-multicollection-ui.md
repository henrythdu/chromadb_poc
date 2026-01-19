# Phase 4 Multi-Collection UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update RAG query engine and Gradio UI to support both ArXiv papers and SEC contracts with unified interface and collection selector dropdown.

**Architecture:** Add CitationFormatter class for collection-specific citation formatting, extend RAGQueryEngine with query_with_collection method, update Gradio app with dropdown selector and unified branding.

**Tech Stack:** Python 3.10+, pytest, gradio, chromadb

---

## Task 1: Create CitationFormatter Class

**Files:**
- Create: `src/retrieval/citation_formatter.py`
- Test: `tests/test_citation_formatter.py`

**Step 1: Write the failing tests**

Create `tests/test_citation_formatter.py`:

```python
"""Tests for CitationFormatter."""

import pytest

from src.retrieval.citation_formatter import CitationFormatter


def test_format_arxiv_paper_complete():
    """Test formatting complete ArXiv paper citation."""
    formatter = CitationFormatter()

    metadata = {
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al."
    }

    result = formatter.format_citation(metadata)

    # Should contain title, authors, arxiv ID, and markdown link
    assert "Attention Is All You Need" in result
    assert "Vaswani et al." in result
    assert "arxiv:2301.07041" in result
    assert "https://arxiv.org/abs/2301.07041" in result
    assert result.startswith("[") and "](" in result  # Markdown link format


def test_format_arxiv_paper_missing_fields():
    """Test formatting ArXiv paper with missing fields uses fallbacks."""
    formatter = CitationFormatter()

    metadata = {
        "arxiv_id": "2301.07041",
        # Missing title and authors
    }

    result = formatter.format_citation(metadata)

    assert "Unknown Title" in result
    assert "Unknown Authors" in result
    assert "arxiv:2301.07041" in result


def test_format_contract_complete():
    """Test formatting complete contract citation."""
    formatter = CitationFormatter()

    metadata = {
        "document_id": "abc123",
        "company_name": "Acme Corp",
        "contract_type": "Co_Branding",
        "execution_date": "2020-01-15",
        "filing_type": "S-1",
        "exhibit_number": "EX-10.1"
    }

    result = formatter.format_citation(metadata)

    assert "Acme Corp" in result
    assert "Co_Branding" in result
    assert "2020-01-15" in result
    assert "Filing: S-1" in result
    assert "Exhibit: EX-10.1" in result


def test_format_contract_missing_fields():
    """Test formatting contract with missing fields uses fallbacks."""
    formatter = CitationFormatter()

    metadata = {
        "document_id": "abc123",
        "company_name": "Acme Corp",
        # Missing other fields
    }

    result = formatter.format_citation(metadata)

    assert "Acme Corp" in result
    assert "Unknown Agreement" in result
    assert "Unknown Date" in result


def test_format_citation_auto_detects_paper():
    """Test format_citation auto-detects ArXiv paper from arxiv_id field."""
    formatter = CitationFormatter()

    metadata = {
        "arxiv_id": "2301.07041",
        "title": "Test Paper"
    }

    result = formatter.format_citation(metadata)

    assert "arxiv:2301.07041" in result
    assert "https://arxiv.org/abs/2301.07041" in result


def test_format_citation_auto_detects_contract():
    """Test format_citation auto-detects contract from document_id field."""
    formatter = CitationFormatter()

    metadata = {
        "document_id": "abc123",
        "company_name": "Test Corp",
        "contract_type": "Test Agreement"
    }

    result = formatter.format_citation(metadata)

    assert "Test Corp" in result
    assert "Test Agreement" in result


def test_format_citation_unknown_type():
    """Test format_citation handles unknown metadata type."""
    formatter = CitationFormatter()

    metadata = {
        "unknown_field": "some_value"
    }

    result = formatter.format_citation(metadata)

    assert "Unknown source" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_citation_formatter.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'src.retrieval.citation_formatter'`

**Step 3: Implement CitationFormatter class**

Create `src/retrieval/citation_formatter.py`:

```python
"""Format citations for different collection types."""

import logging

logger = logging.getLogger(__name__)


class CitationFormatter:
    """Format citations for different collection types.

    Detects collection type from metadata fields and formats appropriately:
    - ArXiv papers: arxiv_id field present
    - SEC contracts: document_id field present
    """

    def format_citation(self, metadata: dict) -> str:
        """Format citation based on collection type detection.

        Args:
            metadata: Chunk metadata dictionary

        Returns:
            Formatted citation string (markdown format for papers, plain text for contracts)
        """
        if "arxiv_id" in metadata:
            return self._format_arxiv_paper(metadata)
        elif "document_id" in metadata:
            return self._format_contract(metadata)
        else:
            logger.warning(f"Unknown metadata type: {list(metadata.keys())}")
            return f"Unknown source: {metadata}"

    def _format_arxiv_paper(self, metadata: dict) -> str:
        """Format ArXiv paper citation with clickable link.

        Args:
            metadata: Must contain arxiv_id, may contain title and authors

        Returns:
            Markdown link format: [Title - Authors (arxiv:id)](url)
        """
        title = metadata.get("title", "Unknown Title")
        authors = metadata.get("authors", "Unknown Authors")
        arxiv_id = metadata.get("arxiv_id", "")

        citation_text = f"{title} - {authors} (arxiv:{arxiv_id})"
        url = f"https://arxiv.org/abs/{arxiv_id}"

        return f"[{citation_text}]({url})"

    def _format_contract(self, metadata: dict) -> str:
        """Format SEC contract citation with descriptive narrative.

        Args:
            metadata: Must contain document_id, may contain company_name,
                     contract_type, execution_date, filing_type, exhibit_number

        Returns:
            Plain text format: Company [Contract Type] executed on Date (Filing: X Exhibit: Y)
        """
        company = metadata.get("company_name", "Unknown Company")
        contract_type = metadata.get("contract_type", "Unknown Agreement")
        date = metadata.get("execution_date", "Unknown Date")
        filing = metadata.get("filing_type", "N/A")
        exhibit = metadata.get("exhibit_number", "N/A")

        return f"{company} [{contract_type}] executed on {date} (Filing: {filing} Exhibit: {exhibit})"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_citation_formatter.py -v`
Expected: PASS (all 7 tests pass)

**Step 5: Commit**

```bash
git add src/retrieval/citation_formatter.py tests/test_citation_formatter.py
git commit -m "feat: add CitationFormatter for multi-collection citation formatting"
```

---

## Task 2: Update RAGQueryEngine with query_with_collection Method

**Files:**
- Modify: `src/retrieval/engine.py`
- Test: `tests/test_rag_engine_multicollection.py`

**Step 1: Write the failing tests**

Create `tests/test_rag_engine_multicollection.py`:

```python
"""Tests for multi-collection RAG query engine."""

from unittest.mock import MagicMock, patch

import pytest


def test_query_with_collection_papers():
    """Test querying arxiv_papers_v1 collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
        return

    from src.retrieval.engine import RAGQueryEngine

    if chromadb is None:
        pytest.skip("chromadb not installed")

    with patch("src.retrieval.engine.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock heartbeat
        mock_instance.heartbeat.return_value = True

        # Mock collections
        mock_papers_collection = MagicMock()
        mock_papers_collection.query.return_value = {
            "documents": [["Test document"]],
            "metadatas": [[{"arxiv_id": "2301.07041"}]],
            "distances": [[0.1]]
        }
        mock_instance.get_or_create_collection.return_value = mock_papers_collection

        # Mock LLM response
        with patch("src.retrieval.engine.OpenRouterLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.answer_question.return_value = "Test answer"
            mock_llm_class.return_value = mock_llm

            # Mock reranker
            with patch("src.retrieval.engine.CohereReranker") as mock_reranker_class:
                mock_reranker = MagicMock()
                mock_reranker.rerank_results.return_value = [
                    {"text": "Test", "metadata": {"arxiv_id": "2301.07041"}}
                ]
                mock_reranker_class.return_value = mock_reranker

                engine = RAGQueryEngine(
                    chroma_api_key="test_key",
                    chroma_tenant="test_tenant",
                    chroma_database="test_db",
                )

                result = engine.query_with_collection(
                    question="test query",
                    collection_name="arxiv_papers_v1",
                    use_rerank=False,
                )

                # Verify we got a result
                assert "answer" in result
                assert result["answer"] == "Test answer"


def test_query_with_collection_contracts():
    """Test querying contracts collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
        return

    from src.retrieval.engine import RAGQueryEngine

    if chromadb is None:
        pytest.skip("chromadb not installed")

    with patch("src.retrieval.engine.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        mock_instance.heartbeat.return_value = True

        mock_contracts_collection = MagicMock()
        mock_contracts_collection.query.return_value = {
            "documents": [["Test contract clause"]],
            "metadatas": [[{"document_id": "abc123", "company_name": "Test Corp"}]],
            "distances": [[0.1]]
        }
        mock_instance.get_or_create_collection.return_value = mock_contracts_collection

        with patch("src.retrieval.engine.OpenRouterLLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.answer_question.return_value = "Test contract answer"
            mock_llm_class.return_value = mock_llm

            with patch("src.retrieval.engine.CohereReranker") as mock_reranker_class:
                mock_reranker = MagicMock()
                mock_reranker.rerank_results.return_value = [
                    {"text": "Test", "metadata": {"document_id": "abc123"}}
                ]
                mock_reranker_class.return_value = mock_reranker

                engine = RAGQueryEngine(
                    chroma_api_key="test_key",
                    chroma_tenant="test_tenant",
                    chroma_database="test_db",
                )

                result = engine.query_with_collection(
                    question="test query",
                    collection_name="contracts",
                    use_rerank=False,
                )

                assert "answer" in result
                assert result["answer"] == "Test contract answer"


def test_query_with_collection_invalid_collection():
    """Test that invalid collection name raises ValueError."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
        return

    from src.retrieval.engine import RAGQueryEngine

    if chromadb is None:
        pytest.skip("chromadb not installed")

    with patch("src.retrieval.engine.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.heartbeat.return_value = True

        engine = RAGQueryEngine(
            chroma_api_key="test_key",
            chroma_tenant="test_tenant",
            chroma_database="test_db",
        )

        with pytest.raises(ValueError, match="Invalid collection"):
            engine.query_with_collection(
                question="test query",
                collection_name="invalid_collection_name",
            )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_engine_multicollection.py -v`
Expected: FAIL - `AttributeError: 'RAGQueryEngine' object has no attribute 'query_with_collection'`

**Step 3: Implement query_with_collection method**

Modify `src/retrieval/engine.py` - add the method after the existing `query` method (around line 111):

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
            collection_name: Collection name to query ("arxiv_papers_v1" or "contracts")
            use_rerank: Whether to use reranker

        Returns:
            Result dict with answer and citations

        Raises:
            ValueError: If collection_name is not valid
        """
        valid_collections = ["arxiv_papers_v1", "contracts"]
        if collection_name not in valid_collections:
            raise ValueError(
                f"Invalid collection: {collection_name}. "
                f"Valid options: {valid_collections}"
            )

        logger.info(f"Querying collection: {collection_name}")
        logger.debug(f"Query: {question}")

        # Create temporary retriever for specified collection
        retriever = HybridSearchRetriever(
            chroma_api_key=self.retriever.client._api_key,
            chroma_tenant=self.retriever.client._tenant,
            chroma_database=self.retriever.client._database,
            collection_name=collection_name,
            top_k=self.top_k_retrieve,
        )

        # Step 1: Retrieve
        retrieved = retriever.retrieve(question)
        logger.info(f"Retrieved {len(retrieved)} chunks from {collection_name}")

        if not retrieved:
            collection_type = "contracts" if collection_name == "contracts" else "papers"
            return {
                "answer": f"No relevant information found in the {collection_type}.",
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_engine_multicollection.py -v`
Expected: PASS (all 3 tests pass)

**Step 5: Commit**

```bash
git add src/retrieval/engine.py tests/test_rag_engine_multicollection.py
git commit -m "feat: add query_with_collection method to RAGQueryEngine"
```

---

## Task 3: Update _extract_citations to Use CitationFormatter

**Files:**
- Modify: `src/retrieval/engine.py`
- Test: Update existing tests or add new test in `tests/test_rag_engine_multicollection.py`

**Step 1: Update _extract_citations method**

Modify `src/retrieval/engine.py` - replace the existing `_extract_citations` method (around line 113):

```python
    def _extract_citations(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[str]:
        """Extract citation strings from chunks using collection-aware formatter.

        Args:
            chunks: Reranked chunks

        Returns:
            List of formatted citations (arxiv papers have links, contracts have narrative)
        """
        from .citation_formatter import CitationFormatter

        formatter = CitationFormatter()
        citations = []
        seen = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Use arxiv_id for papers, document_id for contracts
            doc_id = metadata.get("arxiv_id") or metadata.get("document_id")
            page = metadata.get("page_number", "?")

            if doc_id:
                key = f"{doc_id}:{page}"
                if key not in seen:
                    citations.append(formatter.format_citation(metadata))
                    seen.add(key)

        return citations
```

**Step 2: Update retrieval/__init__.py to export CitationFormatter**

Modify `src/retrieval/__init__.py` - add CitationFormatter to imports:

```python
"""Retrieval components for RAG pipeline."""

from .citation_formatter import CitationFormatter
from .engine import RAGQueryEngine
from .hybrid_search import HybridSearchRetriever
from .reranker import CohereReranker

__all__ = [
    "CitationFormatter",
    "RAGQueryEngine",
    "HybridSearchRetriever",
    "CohereReranker",
]
```

**Step 3: Run tests to verify they still pass**

Run: `pytest tests/test_rag_engine_multicollection.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/retrieval/engine.py src/retrieval/__init__.py
git commit -m "refactor: update _extract_citations to use CitationFormatter"
```

---

## Task 4: Update Gradio UI with Collection Selector

**Files:**
- Modify: `src/ui/gradio_app.py`

**Step 1: Add collection mapping and update query function**

Modify `src/ui/gradio_app.py` - add the collection mapping after the examples list (around line 194):

```python
# Collection mapping for dropdown
COLLECTION_MAP = {
    "ArXiv Papers": "arxiv_papers_v1",
    "SEC Contracts": "contracts",
}
```

Update the `query_paper_stream` function signature and implementation (replace existing function around line 89):

```python
def query_paper_stream(
    message: str,
    history: list[tuple[str, str]],
    collection_choice: str,  # New parameter
):
    """Process a user query through the RAG pipeline with streaming.

    Args:
        message: User message
        history: Conversation history (list of tuples)
        collection_choice: Selected collection from dropdown

    Yields:
        Assistant response string (streaming)
    """
    # Validate input
    is_valid, error_msg = validate_input(message)
    if not is_valid:
        yield f"⚠️ {error_msg}"
        return

    try:
        engine = _get_engine()

        # Map dropdown choice to collection name
        collection_name = COLLECTION_MAP[collection_choice]

        # Check cache first
        key = engine._cache_key(message, collection_name)
        cached_result = None

        with engine._lock:
            if key in engine._cache:
                timestamp, result = engine._cache[key]
                if (time.time() - timestamp) < engine.cache_ttl_seconds:
                    engine._cache_hits += 1
                    cached_result = result
                    logger.info(f"Cache HIT for: {message[:50]}... (collection: {collection_name})")

        # If cache hit, return immediately
        if cached_result:
            final_answer = cached_result["answer"]
            citations = cached_result.get("citations", [])
            if citations:
                final_answer += format_citations(citations)
            yield final_answer
            return

        # Cache miss - increment counter
        with engine._lock:
            engine._cache_misses += 1

        # Create retriever for selected collection
        from ..retrieval.hybrid_search import HybridSearchRetriever

        retriever = HybridSearchRetriever(
            chroma_api_key=engine.base_engine.retriever.client._api_key,
            chroma_tenant=engine.base_engine.retriever.client._tenant,
            chroma_database=engine.base_engine.retriever.client._database,
            collection_name=collection_name,
            top_k=engine.base_engine.top_k_retrieve,
        )

        # Step 1: Retrieve chunks
        retrieved = retriever.retrieve(message)

        if not retrieved:
            collection_type = "contracts" if collection_name == "contracts" else "papers"
            yield f"🔍 No relevant information found in the {collection_type}. Try different keywords."
            return

        # Step 2: Rerank
        reranked = engine.base_engine.reranker.rerank_results(message, retrieved)

        # Step 3: Stream the LLM response
        full_answer = ""
        for chunk in engine.base_engine.llm.answer_question_stream(message, reranked):
            full_answer = chunk
            yield full_answer

        # Step 4: Add citations
        from ..retrieval.citation_formatter import CitationFormatter

        formatter = CitationFormatter()
        citations = []
        seen = set()

        for r in reranked:
            metadata = r.get("metadata", {})
            doc_id = metadata.get("arxiv_id") or metadata.get("document_id")
            page = metadata.get("page_number", "?")

            if doc_id:
                key = f"{doc_id}:{page}"
                if key not in seen:
                    citations.append(formatter.format_citation(metadata))
                    seen.add(key)

        if citations:
            if collection_name == "contracts":
                # Contract citations are plain text, no markdown links
                citation_text = "\n\n---\n**📚 Sources:**\n" + "\n".join(f"- {c}" for c in citations)
            else:
                citation_text = format_citations(citations)
            full_answer += citation_text
            yield full_answer

        # Store result in cache
        final_result_to_cache = {
            "answer": full_answer,
            "citations": citations,
            "sources": [r["metadata"] for r in reranked]
        }

        with engine._lock:
            engine._cache[key] = (time.time(), final_result_to_cache)
            # Eviction logic
            if len(engine._cache) > engine.max_cache_size:
                oldest_key = min(engine._cache, key=lambda k: engine._cache[k][0])
                del engine._cache[oldest_key]
                logger.debug(f"Cache EVICTED: {len(engine._cache)}/{engine.max_cache_size} entries")

    except ValueError as e:
        error_msg = f"⚠️ Configuration Error: {str(e)}"
        logger.error(f"Configuration error: {e}")
        yield error_msg

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(f"Query processing error: {e}")
        yield error_msg
```

**Step 2: Update CachedRAGEngine cache key to include collection**

Modify `src/retrieval/cached_engine.py` - update the `_cache_key` method (or add if not exists):

```python
    def _cache_key(self, query: str, collection: str = "arxiv_papers_v1") -> str:
        """Generate cache key including collection name."""
        return f"{query}:{collection}"
```

**Step 3: Update create_interface to include dropdown**

Modify `src/ui/gradio_app.py` - replace the `create_interface` function (around line 197):

```python
def create_interface() -> gr.Blocks:
    """Create and return the professional Gradio interface with collection selector.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(theme=theme, title="RAG Research Assistant") as interface:
        # Collection selector at top
        gr.Markdown("# 📚 RAG Research Assistant")
        gr.Markdown(
            "### Ask questions about **ArXiv papers** or **SEC contracts**\n\n"
            "I'll search through the selected collection and provide answers with **citations**. "
            "Powered by hybrid search, Cohere Rerank v3, and Claude 3.5 Sonnet."
        )

        with gr.Row():
            collection_dropdown = gr.Dropdown(
                choices=["ArXiv Papers", "SEC Contracts"],
                value="ArXiv Papers",
                label="📁 Select Collection",
                interactive=True,
            )

        # Chat interface
        chatbot = gr.Chatbot(height="calc(100vh - 350px)", show_copy_button=True)
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask about research papers or contract clauses...",
            autofocus=True,
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear History", variant="secondary")

        # Examples
        gr.Examples(
            examples=[
                ["What is transformer architecture?", "ArXiv Papers"],
                ["Explain reinforcement learning", "ArXiv Papers"],
                ["How do diffusion models work?", "ArXiv Papers"],
                ["co-branding agreement obligations", "SEC Contracts"],
                ["termination clause provisions", "SEC Contracts"],
                ["exclusive partnership terms", "SEC Contracts"],
            ],
            inputs=[msg, collection_dropdown],
            label="Example Queries",
        )

        # Event handlers
        def submit_message(message, history, collection):
            """Submit message and stream response."""
            if not message.strip():
                return history, ""
            history = history + [[message, None]]
            return history, ""

        def stream_response(message, history, collection):
            """Stream the response."""
            for response in query_paper_stream(message, history, collection):
                history[-1][1] = response
                yield history

        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot, collection_dropdown],
            outputs=[chatbot, msg],
        ).then(
            stream_response,
            inputs=[msg, chatbot, collection_dropdown],
            outputs=chatbot,
        )

        msg.submit(
            submit_message,
            inputs=[msg, chatbot, collection_dropdown],
            outputs=[chatbot, msg],
        ).then(
            stream_response,
            inputs=[msg, chatbot, collection_dropdown],
            outputs=chatbot,
        )

        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg],
        )

    return interface
```

**Step 4: Update launch function**

Modify `src/ui/gradio_app.py` - update the launch function to return the interface:

```python
def launch(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
):
    """Launch the Gradio interface.

    Args:
        server_name: Server hostname
        server_port: Server port
        share: Whether to create a public link
    """
    interface = create_interface()

    logger.info(f"Launching Gradio interface on http://{server_name}:{server_port}")

    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )
```

**Step 5: Manual test the UI**

Run: `python -m src.ui.gradio_app`
Expected: Gradio UI launches with collection selector dropdown

**Step 6: Commit**

```bash
git add src/ui/gradio_app.py src/retrieval/cached_engine.py
git commit -m "feat: add collection selector dropdown to Gradio UI"
```

---

## Task 5: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (including new multi-collection tests)

**Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

**Step 3: Format code**

Run: `ruff format src/ tests/`

**Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "style: format code with ruff"
```

---

## Task 6: Update Phase 4 Beads Issue

**Step 1: Close Phase 4 issue**

Run: `bd close ChromaDB_POC-7jc --reason="Implemented multi-collection UI with collection selector, CitationFormatter for collection-specific formatting, and query_with_collection method. All tests passing."`

**Step 2: Sync beads**

Run: `bd sync`

**Step 3: Push to git**

```bash
git push
```

---

## Phase 4 Gate Verification

**Run all verification commands:**

```bash
# 1. All tests pass
pytest tests/ -v

# 2. Linting passes
ruff check src/ tests/

# 3. Can import CitationFormatter
python -c "from src.retrieval import CitationFormatter; print('✓ CitationFormatter imports')"

# 4. Can create RAGQueryEngine with query_with_collection
python -c "
from unittest.mock import patch, MagicMock
with patch('src.retrieval.engine.chromadb.CloudClient') as mock:
    mock.return_value.heartbeat.return_value = True
    from src.retrieval import RAGQueryEngine
    engine = RAGQueryEngine(api_key='test', tenant='test', database='test')
    print('✓ query_with_collection method exists:', hasattr(engine, 'query_with_collection'))
"

# 5. Gradio app has collection selector
python -c "
import inspect
from src.ui.gradio_app import create_interface
source = inspect.getsource(create_interface)
assert 'collection_dropdown' in source or 'gr.Dropdown' in source
print('✓ Gradio interface has collection selector')
"
```

**Expected Results:**
- ✅ All tests pass (new + existing)
- ✅ No linting errors
- ✅ CitationFormatter imports correctly
- ✅ query_with_collection method exists on RAGQueryEngine
- ✅ Gradio interface includes collection dropdown
- ✅ Phase 4 issue closed in beads
- ✅ All commits pushed to remote

---

**Total Estimated Time:** 2-3 hours
**Test Coverage:** ~10 new tests for multi-collection functionality
**Lines of Code:** ~300 (new CitationFormatter) + ~100 (engine updates) + ~150 (UI updates)
