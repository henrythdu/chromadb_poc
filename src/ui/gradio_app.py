"""Gradio chat interface for RAG system with streaming."""

import logging
import re
import time

import gradio as gr
from dotenv import load_dotenv

from ..retrieval.cached_engine import CachedRAGEngine

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize engine once at module load time (thread-safe singleton)
logger.info("Initializing RAG engine...")
_engine: CachedRAGEngine | None = None


def _get_engine() -> CachedRAGEngine:
    """Get or create the RAG engine instance (thread-safe singleton)."""
    global _engine
    if _engine is None:
        _engine = CachedRAGEngine()
        logger.info("RAG engine initialized")
    return _engine


# Custom professional theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_border_width="0px",
    block_shadow="*shadow_sm",
)


def validate_input(message: str) -> tuple[bool, str]:
    """Validate user input.

    Args:
        message: User message

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check minimum length
    if len(message) < 10:
        return False, "Query must be at least 10 characters long."

    # Check maximum length
    if len(message) > 500:
        return False, "Query must be less than 500 characters."

    return True, ""


def format_citations(citations: list[str]) -> str:
    """Format citations as clickable ArXiv links.

    Args:
        citations: List of citation strings

    Returns:
        Formatted markdown string with links
    """
    if not citations:
        return ""

    parts = ["\n\n---\n**Sources:**\n"]

    for citation in citations:
        # Extract arxiv ID from citation
        match = re.search(r"arxiv:(\d+\.\d+)", citation)
        if match:
            arxiv_id = match.group(1)
            url = f"https://arxiv.org/abs/{arxiv_id}"
            parts.append(f"- [{citation}]({url})")
        else:
            parts.append(f"- {citation}")

    return "\n".join(parts)


def query_paper_stream(
    message: str,
    history: list[tuple[str, str]],
):
    """Process a user query through the RAG pipeline with streaming.

    Args:
        message: User message
        history: Conversation history (list of tuples)

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

        # Check cache first for potential hit
        key = engine._cache_key(message)
        cached_result = None

        with engine._lock:
            if key in engine._cache:
                timestamp, result = engine._cache[key]
                if (time.time() - timestamp) < engine.cache_ttl_seconds:
                    engine._cache_hits += 1
                    cached_result = result
                    logger.info(f"Cache HIT for: {message[:50]}...")

        # If cache hit, return immediately (no streaming needed)
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

        # Step 1: Retrieve chunks
        retrieved = engine.base_engine.retriever.retrieve(message)

        if not retrieved:
            yield "No relevant information found in the papers."
            return

        # Step 2: Rerank
        reranked = engine.base_engine.reranker.rerank_results(message, retrieved)

        # Step 3: Stream the LLM response
        full_answer = ""
        for chunk in engine.base_engine.llm.answer_question_stream(message, reranked):
            full_answer = chunk
            yield full_answer

        # Step 4: Add citations
        citations = engine.base_engine._extract_citations(reranked)

        if citations:
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
                oldest_key = min(engine._cache, key=lambda k: self._cache[k][0])
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


# Example questions
examples = [
    "What is transformer architecture?",
    "Explain reinforcement learning",
    "How do diffusion models work?",
    "What is attention mechanism?",
]


def create_interface() -> gr.ChatInterface:
    """Create and return the Gradio ChatInterface.

    Returns:
        Gradio ChatInterface
    """
    return gr.ChatInterface(
        fn=query_paper_stream,
        title="📚 ArXiv Research Assistant",
        description=(
            "Ask questions about machine learning research papers. "
            "I'll search through the database and provide answers with citations."
        ),
        examples=examples,
        cache_examples=False,
        save_history=True,
        autofocus=True,
    )


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


if __name__ == "__main__":
    launch()
