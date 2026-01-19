"""Professional Gradio chat interface for RAG system with streaming."""

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


# Professional custom theme with modern design
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
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

    parts = ["\n\n---\n**📚 Sources:**\n"]

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


# Professional example questions with clear, helpful prompts
examples = [
    "What is transformer architecture?",
    "Explain reinforcement learning",
    "How do diffusion models work?",
    "What is attention mechanism?",
    "What is backpropagation in neural networks?",
    "How does batch normalization work?",
]

# Collection mapping for dropdown
COLLECTION_MAP = {
    "ArXiv Papers": "arxiv_papers_v1",
    "SEC Contracts": "contracts",
}


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


def launch(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
):
    """Launch the Gradio interface."""
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
