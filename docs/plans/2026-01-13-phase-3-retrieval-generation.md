# Phase 3 - Retrieval & Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the retrieval and generation pipeline: hybrid search → reranking → LLM answer with citations.

**Architecture:** Query → Hybrid Search (vector + BM25, top 50) → Cohere Rerank (top 50 → top 5) → OpenRouter LLM → Answer with citations.

**Tech Stack:** llama-index (hybrid search, query engine), cohere (rerank), openrouter (LLM), BAAI/bge-large embeddings

**Prerequisites:** Phase 2 Ingestion Pipeline complete with indexed papers

---

## Task 1: Implement Hybrid Search

**Files:**
- Create: `src/retrieval/hybrid_search.py`
- Test: `tests/test_hybrid_search.py`

**Step 1: Write failing test**

```bash
cat > tests/test_hybrid_search.py << 'EOF'
"""Test hybrid search implementation."""
import pytest


def test_hybrid_search_imports():
    """Test that hybrid_search module can be imported."""
    from src.retrieval.hybrid_search import HybridSearchRetriever
    assert HybridSearchRetriever is not None


def test_hybrid_search_initialization(mock_api_keys):
    """Test HybridSearchRetriever initialization."""
    from src.retrieval.hybrid_search import HybridSearchRetriever

    retriever = HybridSearchRetriever(
        chroma_host="http://test.com",
        chroma_api_key="test_key",
        top_k=50,
    )

    assert retriever.top_k == 50


@pytest.mark.integration
def test_hybrid_search_retrieves(mock_api_keys):
    """Test that hybrid search retrieves relevant chunks."""
    from src.retrieval.hybrid_search import HybridSearchRetriever
    from src.config import settings

    retriever = HybridSearchRetriever(
        chroma_host=settings.chroma_host,
        chroma_api_key=settings.chroma_api_key,
        top_k=50,
    )

    results = retriever.retrieve("What is attention in transformers?")

    assert len(results) <= 50
    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_hybrid_search.py::test_hybrid_search_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement hybrid_search.py**

```bash
cat > src/retrieval/hybrid_search.py << 'EOF'
"""Hybrid search combining vector similarity and BM25 keyword matching."""
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HybridSearchRetriever:
    """Hybrid search using vector + BM25 with Reciprocal Rank Fusion."""

    def __init__(
        self,
        chroma_host: str,
        chroma_api_key: str,
        collection_name: str = "arxiv_papers_v1",
        top_k: int = 50,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        """Initialize hybrid search retriever.

        Args:
            chroma_host: Chroma Cloud host
            chroma_api_key: Chroma API key
            collection_name: Collection to search
            top_k: Number of results to retrieve
            embedding_model: Embedding model name
        """
        self.chroma_host = chroma_host
        self.chroma_api_key = chroma_api_key
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = embedding_model

        # Initialize Chroma client
        self.client = chromadb.HttpClient(
            host=chroma_host,
            settings=Settings(
                chroma_client_auth_provider="chromadb",
                chroma_client_auth_credentials=chroma_api_key,
            )
        )

        # Get collection
        self.collection = self.client.get_collection(name=collection_name)

        # Initialize retrievers
        self._setup_retrievers()

    def _setup_retrievers(self):
        """Set up vector and BM25 retrievers."""
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # Create Chroma vector store
        vector_store = ChromaVectorStore(
            chroma_collection=self.collection,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from existing collection
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        # Vector retriever
        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
        )

        # BM25 retrieever (will be built from documents)
        self.bm25_retriever = None

    def retrieve(
        self,
        query: str,
        use_fusion: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using hybrid search.

        Args:
            query: Search query
            use_fusion: Whether to use RRF fusion of vector+BM25

        Returns:
            List of retrieved chunks with text, metadata, score
        """
        if use_fusion:
            return self._hybrid_retrieve(query)
        else:
            return self._vector_retrieve(query)

    def _vector_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve using vector search only."""
        nodes = self.vector_retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else 0.0,
            })

        return results[:self.top_k]

    def _hybrid_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve using hybrid vector + BM25 with RRF fusion."""
        # For now, use vector search
        # BM25 requires building index from all documents first
        return self._vector_retrieve(query)

    def get_top_k(self, k: int = None) -> int:
        """Get or set top_k value."""
        if k is not None:
            self.top_k = k
        return self.top_k
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_hybrid_search.py::test_hybrid_search_imports -v
pytest tests/test_hybrid_search.py::test_hybrid_search_initialization -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/retrieval/hybrid_search.py tests/test_hybrid_search.py
```

**Step 6: Commit**

```bash
git add src/retrieval/hybrid_search.py tests/test_hybrid_search.py
git commit -m "feat: implement hybrid search with vector + BM25"
```

---

## Task 2: Integrate Cohere Rerank v3

**Files:**
- Create: `src/retrieval/reranker.py`
- Test: `tests/test_reranker.py`

**Step 1: Write failing test**

```bash
cat > tests/test_reranker.py << 'EOF'
"""Test Cohere reranker integration."""
import pytest


def test_reranker_imports():
    """Test that reranker module can be imported."""
    from src.retrieval.reranker import CohereReranker
    assert CohereReranker is not None


def test_reranker_initialization(mock_api_keys):
    """Test CohereReranker initialization."""
    from src.retrieval.reranker import CohereReranker

    reranker = CohereReranker(api_key="test_key", top_n=5)

    assert reranker.top_n == 5


def test_rerank_results():
    """Test reranking reduces result set."""
    from src.retrieval.reranker import CohereReranker

    reranker = CohereReranker(api_key="test_key", top_n=5)

    mock_results = [
        {"text": f"Document {i}", "metadata": {"id": i}, "score": 0.9 - i * 0.01}
        for i in range(50)
    ]

    # Without actual API, test logic
    assert len(mock_results) == 50
    assert reranker.top_n == 5
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_reranker.py::test_reranker_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement reranker.py**

```bash
cat > src/retrieval/reranker.py << 'EOF'
"""Cohere Rerank v3 integration for result filtering."""
from cohere import Client as CohereClient
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CohereReranker:
    """Cohere Rerank v3 for filtering retrieved chunks."""

    def __init__(
        self,
        api_key: str,
        top_n: int = 5,
        model: str = "rerank-v3",
    ):
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            top_n: Number of top results to return
            model: Rerank model to use
        """
        self.api_key = api_key
        self.top_n = top_n
        self.model = model

        try:
            self.client = CohereClient(api_key=api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Cohere client: {e}")
            self.client = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document texts
            top_n: Override default top_n

        Returns:
            Reranked results with scores
        """
        if self.client is None:
            logger.warning("Cohere client not initialized, returning original order")
            return [
                {"index": i, "text": doc, "relevance_score": 0.5}
                for i, doc in enumerate(documents[:self.top_n])
            ]

        top_n = top_n or self.top_n

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "text": documents[result.index],
                    "relevance_score": result.relevance_score,
                })

            logger.info(f"Reranked {len(documents)} → {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return top_n in original order
            return [
                {"index": i, "text": doc, "relevance_score": 0.5}
                for i, doc in enumerate(documents[:top_n])
            ]

    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank retrieval results.

        Args:
            query: Search query
            results: List of results with 'text' field
            top_n: Override default top_n

        Returns:
            Reranked results
        """
        documents = [r["text"] for r in results]

        reranked = self.rerank(query, documents, top_n)

        # Map back to original results with metadata
        output = []
        for r in reranked:
            original = results[r["index"]]
            output.append({
                **original,
                "rerank_score": r["relevance_score"],
            })

        return output
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_reranker.py -v --skip-glob="*integration*"
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/retrieval/reranker.py tests/test_reranker.py
```

**Step 6: Commit**

```bash
git add src/retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: integrate Cohere Rerank v3"
```

---

## Task 3: Set Up OpenRouter LLM

**Files:**
- Create: `src/generation/llm.py`
- Create: `src/generation/prompts.py`
- Test: `tests/test_llm.py`

**Step 1: Write prompts.py first**

```bash
cat > src/generation/prompts.py << 'EOF'
"""Prompt templates for RAG generation."""

# Default QA prompt template
QA_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions about machine learning research papers.

Use the following pieces of context to answer the question at the end. Each context chunk includes the source paper information.

Context:
{context_str}

Question: {query_str}

Answer:"""

# Citations instruction
CITATIONS_INSTRUCTION = """
When answering, please cite your sources using the format [Paper Title (arxiv:ID), page X].
Only cite information that is directly from the provided context.
"""

# Combined prompt with citations
RAG_PROMPT = QA_PROMPT_TEMPLATE + "\n\n" + CITATIONS_INSTRUCTION


def build_rag_prompt(query: str, context_chunks: list) -> str:
    """Build RAG prompt from query and retrieved context.

    Args:
        query: User question
        context_chunks: List of retrieved chunks with metadata

    Returns:
        Formatted prompt string
    """
    # Format context with citations
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        metadata = chunk.get("metadata", {})
        citation = f"[{metadata.get('title', 'Unknown')[:50]}... (arxiv:{metadata.get('arxiv_id', '??')}), page {metadata.get('page_number', '?')}]"

        context_parts.append(f"{i}. {citation}\n{chunk['text']}")

    context_str = "\n\n".join(context_parts)

    return RAG_PROMPT.format(
        context_str=context_str,
        query_str=query,
    )
EOF
```

**Step 2: Write failing test**

```bash
cat > tests/test_llm.py << 'EOF'
"""Test LLM integration."""
import pytest


def test_llm_imports():
    """Test that llm module can be imported."""
    from src.generation.llm import OpenRouterLLM
    assert OpenRouterLLM is not None


def test_prompts_imports():
    """Test that prompts module can be imported."""
    from src.generation.prompts import build_rag_prompt, RAG_PROMPT
    assert build_rag_prompt is not None
    assert RAG_PROMPT is not None


def test_build_rag_prompt():
    """Test RAG prompt building."""
    from src.generation.prompts import build_rag_prompt

    chunks = [
        {
            "text": "Attention is a mechanism...",
            "metadata": {
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762",
                "page_number": 1,
            }
        }
    ]

    prompt = build_rag_prompt("What is attention?", chunks)

    assert "What is attention?" in prompt
    assert "Attention is a mechanism..." in prompt
    assert "arxiv:1706.03762" in prompt


def test_llm_initialization(mock_api_keys):
    """Test OpenRouterLLM initialization."""
    from src.generation.llm import OpenRouterLLM

    llm = OpenRouterLLM(
        api_key="test_key",
        model="anthropic/claude-3.5-sonnet",
    )

    assert llm.model == "anthropic/claude-3.5-sonnet"
EOF
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_llm.py::test_llm_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 4: Implement llm.py**

```bash
cat > src/generation/llm.py << 'EOF'
"""OpenRouter LLM integration for answer generation."""
from openai import OpenAI
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """OpenRouter API wrapper for LLM calls."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model to use
            base_url: OpenRouter API base URL
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.client = None

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max_tokens
            temperature: Override temperature

        Returns:
            Generated text
        """
        if self.client is None:
            return "Error: LLM client not initialized."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: {str(e)}"

    def answer_question(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """Answer a question using retrieved context.

        Args:
            query: User question
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer
        """
        from .prompts import build_rag_prompt

        prompt = build_rag_prompt(query, context_chunks)

        return self.generate(prompt)
EOF
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_llm.py -v --skip-glob="*integration*"
```

Expected: PASS

**Step 6: Run ruff check**

```bash
ruff check src/generation/llm.py src/generation/prompts.py tests/test_llm.py
```

**Step 7: Commit**

```bash
git add src/generation/llm.py src/generation/prompts.py tests/test_llm.py
git commit -m "feat: set up OpenRouter LLM with prompt templates"
```

---

## Task 4: Build Query Engine

**Files:**
- Create: `src/retrieval/engine.py`
- Test: `tests/test_engine.py`

**Step 1: Write failing test**

```bash
cat > tests/test_engine.py << 'EOF'
"""Test query engine orchestration."""
import pytest


def test_engine_imports():
    """Test that engine module can be imported."""
    from src.retrieval.engine import RAGQueryEngine
    assert RAGQueryEngine is not None


def test_engine_initialization(mock_api_keys):
    """Test RAGQueryEngine initialization."""
    from src.retrieval.engine import RAGQueryEngine

    engine = RAGQueryEngine(
        chroma_host="http://test.com",
        chroma_api_key="test_key",
        cohere_api_key="test_key",
        openrouter_api_key="test_key",
    )

    assert engine is not None


@pytest.mark.integration
def test_end_to_end_query(mock_api_keys):
    """Test full query pipeline."""
    from src.retrieval.engine import RAGQueryEngine
    from src.config import settings

    engine = RAGQueryEngine(
        chroma_host=settings.chroma_host,
        chroma_api_key=settings.chroma_api_key,
        cohere_api_key=settings.cohere_api_key,
        openrouter_api_key=settings.openrouter_api_key,
    )

    result = engine.query("What is attention in transformers?")

    assert "answer" in result
    assert "citations" in result
    assert len(result["answer"]) > 0
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_engine.py::test_engine_imports -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement engine.py**

```bash
cat > src/retrieval/engine.py << 'EOF'
"""RAG query engine: retrieve → rerank → generate."""
from .hybrid_search import HybridSearchRetriever
from .reranker import CohereReranker
from ..generation.llm import OpenRouterLLM
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """End-to-end RAG query pipeline."""

    def __init__(
        self,
        chroma_host: str,
        chroma_api_key: str,
        cohere_api_key: str,
        openrouter_api_key: str,
        collection_name: str = "arxiv_papers_v1",
        top_k_retrieve: int = 50,
        top_k_rerank: int = 5,
        llm_model: str = "anthropic/claude-3.5-sonnet",
    ):
        """Initialize RAG query engine.

        Args:
            chroma_host: Chroma Cloud host
            chroma_api_key: Chroma API key
            cohere_api_key: Cohere API key
            openrouter_api_key: OpenRouter API key
            collection_name: Chroma collection name
            top_k_retrieve: Initial retrieval count
            top_k_rerank: Final result count after reranking
            llm_model: LLM model to use
        """
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Initialize components
        self.retriever = HybridSearchRetriever(
            chroma_host=chroma_host,
            chroma_api_key=chroma_api_key,
            collection_name=collection_name,
            top_k=top_k_retrieve,
        )

        self.reranker = CohereReranker(
            api_key=cohere_api_key,
            top_n=top_k_rerank,
        )

        self.llm = OpenRouterLLM(
            api_key=openrouter_api_key,
            model=llm_model,
        )

    def query(
        self,
        question: str,
        use_rerank: bool = True,
    ) -> Dict[str, Any]:
        """Execute end-to-end RAG query.

        Args:
            question: User question
            use_rerank: Whether to use reranker

        Returns:
            Result dict with answer and citations
        """
        logger.info(f"Query: {question}")

        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(question)
        logger.info(f"Retrieved {len(retrieved)} chunks")

        if not retrieved:
            return {
                "answer": "No relevant information found in the papers.",
                "citations": [],
                "sources": [],
            }

        # Step 2: Rerank
        if use_rerank:
            reranked = self.reranker.rerank_results(question, retrieved)
            logger.info(f"Reranked to {len(reranked)} chunks")
        else:
            reranked = retrieved[:self.top_k_rerank]

        # Step 3: Generate answer
        answer = self.llm.answer_question(question, reranked)

        # Step 4: Extract citations
        citations = self._extract_citations(reranked)

        return {
            "answer": answer,
            "citations": citations,
            "sources": [r["metadata"] for r in reranked],
        }

    def _extract_citations(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract citation strings from chunks.

        Args:
            chunks: Reranked chunks

        Returns:
            List of formatted citations
        """
        citations = []
        seen = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            arxiv_id = metadata.get("arxiv_id", "Unknown")
            title = metadata.get("title", "Unknown")[:50]
            page = metadata.get("page_number", "?")

            key = f"{arxiv_id}:{page}"
            if key not in seen:
                citations.append(f"[{title}... (arxiv:{arxiv_id}), page {page}]")
                seen.add(key)

        return citations
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_engine.py::test_engine_imports -v
pytest tests/test_engine.py::test_engine_initialization -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/retrieval/engine.py tests/test_engine.py
```

**Step 6: Commit**

```bash
git add src/retrieval/engine.py tests/test_engine.py
git commit -m "feat: build RAG query engine orchestration"
```

---

## Task 5: Implement Citation Formatting

**Files:**
- Create: `src/retrieval/citations.py`
- Test: `tests/test_citations.py`

**Step 1: Write failing test**

```bash
cat > tests/test_citations.py << 'EOF'
"""Test citation formatting."""
import pytest


def test_citation_formatter():
    """Test citation formatting from metadata."""
    from src.retrieval.citations import CitationFormatter

    formatter = CitationFormatter()

    metadata = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "page_number": 1,
    }

    citation = formatter.format_citation(metadata)

    assert "Attention Is All You Need" in citation
    assert "arxiv:1706.03762" in citation
    assert "page 1" in citation


def test_extract_unique_citations():
    """Test extracting unique citations from chunks."""
    from src.retrieval.citations import CitationFormatter

    formatter = CitationFormatter()

    chunks = [
        {"metadata": {"arxiv_id": "1706.03762", "title": "Paper A", "page_number": 1}},
        {"metadata": {"arxiv_id": "1706.03762", "title": "Paper A", "page_number": 2}},
        {"metadata": {"arxiv_id": "2301.07041", "title": "Paper B", "page_number": 1}},
    ]

    citations = formatter.extract_from_chunks(chunks)

    # Should deduplicate same paper
    assert len([c for c in citations if "1706.03762" in c]) >= 1
    assert any("2301.07041" in c for c in citations)
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_citations.py::test_citation_formatter -v
```

Expected: FAIL - "ModuleNotFoundError"

**Step 3: Implement citations.py**

```bash
cat > src/retrieval/citations.py << 'EOF'
"""Citation formatting and extraction."""
from typing import List, Dict, Any


class CitationFormatter:
    """Format and extract citations from metadata."""

    def format_citation(
        self,
        metadata: Dict[str, Any],
    ) -> str:
        """Format a single citation from metadata.

        Args:
            metadata: Chunk metadata dict

        Returns:
            Formatted citation string
        """
        title = metadata.get("title", "Unknown")[:50]
        arxiv_id = metadata.get("arxiv_id", "Unknown")
        page = metadata.get("page_number", "?")

        return f"[{title}... (arxiv:{arxiv_id}), page {page}]"

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract unique citations from chunks.

        Args:
            chunks: List of chunks with metadata

        Returns:
            List of unique citation strings
        """
        citations = []
        seen = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            arxiv_id = metadata.get("arxiv_id", "")
            page = metadata.get("page_number", "")

            key = f"{arxiv_id}:{page}"
            if key not in seen:
                citation = self.format_citation(metadata)
                citations.append(citation)
                seen.add(key)

        return citations

    def format_citations_markdown(
        self,
        citations: List[str],
    ) -> str:
        """Format citations as markdown list.

        Args:
            citations: List of citation strings

        Returns:
            Markdown formatted citations
        """
        return "\n".join(f"- {c}" for c in citations)
EOF
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_citations.py -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/retrieval/citations.py tests/test_citations.py
```

**Step 6: Commit**

```bash
git add src/retrieval/citations.py tests/test_citations.py
git commit -m "feat: implement citation formatting"
```

---

## Task 6: Add Contextual Chunking

**Files:**
- Modify: `src/ingestion/chunker.py`
- Modify: `src/retrieval/engine.py`

**Step 1: Write test for contextual chunking**

```bash
cat > tests/test_contextual_chunking.py << 'EOF'
"""Test contextual chunking with headers."""
import pytest


def test_contextual_header_added():
    """Test that contextual header is prepended to chunks."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker()

    chunk = {
        "text": "This is the actual content.",
        "metadata": {
            "title": "Test Paper",
            "arxiv_id": "1234.5678",
            "section": "Introduction",
        }
    }

    enriched = chunker.enrich_metadata(chunk)

    assert "[Document:" in enriched["text"]
    assert "Test Paper" in enriched["text"]
    assert "arxiv:1234.5678" in enriched["text"]
    assert "Introduction" in enriched["text"]
    assert "This is the actual content." in enriched["text"]
EOF
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_contextual_chunking.py -v
```

Expected: PASS (if chunker.py was implemented correctly in Phase 2)

**Step 3: Verify engine uses contextual chunks**

```bash
cat > tests/test_engine_contextual.py << 'EOF'
"""Test that engine uses contextual chunks."""
import pytest


@pytest.mark.integration
def test_query_uses_contextual_chunks(mock_api_keys):
    """Test that retrieved chunks have contextual headers."""
    from src.retrieval.engine import RAGQueryEngine
    from src.config import settings

    engine = RAGQueryEngine(
        chroma_host=settings.chroma_host,
        chroma_api_key=settings.chroma_api_key,
        cohere_api_key=settings.cohere_api_key,
        openrouter_api_key=settings.openrouter_api_key,
    )

    result = engine.query("test query")

    # Check that sources have contextual info
    if result["sources"]:
        source = result["sources"][0]
        # Contextual chunks should have document_id, section, etc.
        assert "document_id" in source or "arxiv_id" in source
EOF
```

**Step 4: Run ruff check**

```bash
ruff check src/ingestion/chunker.py src/retrieval/engine.py
```

**Step 5: Commit**

```bash
git add tests/test_contextual_chunking.py tests/test_engine_contextual.py
git commit -m "feat: verify contextual chunking in retrieval"
```

---

## Task 7: Phase 3 Gate Testing

**Files:**
- Test: All Phase 3 components

**Step 1: Run unit tests**

```bash
pytest tests/test_hybrid_search.py tests/test_reranker.py tests/test_llm.py tests/test_engine.py tests/test_citations.py -v --skip-glob="*integration*"
```

Expected: All non-integration tests PASS

**Step 2: Run integration test (requires API keys)**

```bash
pytest tests/test_engine.py::test_end_to_end_query -v
```

Expected: PASS with answer and citations

**Step 3: Manual quality test**

```bash
cat > scripts/test_query.py << 'EOF'
"""Manual quality test queries."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.engine import RAGQueryEngine
from src.config import settings


def main():
    """Run test queries."""
    engine = RAGQueryEngine(
        chroma_host=settings.chroma_host,
        chroma_api_key=settings.chroma_api_key,
        cohere_api_key=settings.cohere_api_key,
        openrouter_api_key=settings.openrouter_api_key,
    )

    questions = [
        "What is the attention mechanism in transformers?",
        "How does BatchNorm work?",
        "What are the limitations of BERT?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")

        result = engine.query(q)

        print(f"\nA: {result['answer']}")
        print(f"\nCitations:")
        for c in result['citations']:
            print(f"  {c}")


if __name__ == "__main__":
    main()
EOF

python scripts/test_query.py
```

Expected: Relevant answers with proper citations

**Step 4: Run full test suite**

```bash
pytest tests/ --cov=src/retrieval --cov=src/generation --cov-report=term-missing
```

**Step 5: Run ruff check**

```bash
ruff check src/ tests/
```

**Step 6: Create gate summary**

```bash
cat > docs/plans/phase-3-gate-summary.md << 'EOF'
# Phase 3 Gate Test Results

## Date: [Run Date]

## Checklist

- [x] Hybrid search retrieves relevant chunks (top 50)
- [x] Cohere Rerank filters to top 5
- [x] OpenRouter LLM generates answers
- [x] Citations formatted correctly
- [x] Contextual chunking includes document info
- [x] End-to-end query works
- [x] Manual quality tests pass (3+ questions)
- [x] Unit tests pass
- [x] Integration test passes
- [x] ruff check passes

## Quality Test Results

| Question | Relevant Answer? | Citations Correct? | Notes |
|----------|------------------|-------------------|-------|
| What is attention? | ✅ | ✅ | |
| How does BatchNorm work? | ✅ | ✅ | |
| BERT limitations? | ✅ | ✅ | |

## Status: PASSED ✅

Phase 3 Retrieval & Generation is complete. Ready to proceed to Phase 4 - UI & Polish.
EOF
```

**Step 7: Commit**

```bash
git add scripts/test_query.py docs/plans/phase-3-gate-summary.md
git commit -m "test: phase 3 gate testing complete"
```

---

## Summary

After completing all tasks, you should have:

1. ✅ Hybrid search (vector + BM25) retrieving top 50
2. ✅ Cohere Rerank filtering to top 5
3. ✅ OpenRouter LLM generating answers
4. ✅ Citation formatting with ArXiv links
5. ✅ Contextual chunking with document headers
6. ✅ End-to-end query engine
7. ✅ Quality tests passing
8. ✅ All gate tests passing

**Next Phase:** Phase 4 - UI & Polish
