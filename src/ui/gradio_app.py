"""Gradio chat interface for RAG system with streaming."""

import logging
import re

import gradio as gr
from dotenv import load_dotenv

from ..retrieval.cached_engine import CachedRAGEngine

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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
        Formatted HTML string with links
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
    history: list[dict],
) -> tuple[str, list[dict]]:
    """Process a user query through the RAG pipeline with streaming.

    Args:
        message: User message
        history: Conversation history (messages format)

    Returns:
        Tuple of (empty_string, updated_history)
    """
    # Validate input
    is_valid, error_msg = validate_input(message)
    if not is_valid:
        return "", history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": f"⚠️ {error_msg}"}]

    try:
        # Initialize cached engine on first use
        if not hasattr(query_paper_stream, "engine"):
            query_paper_stream.engine = CachedRAGEngine()
            logger.info("RAG engine initialized")

        # Stream the response
        full_answer = ""
        for chunk in query_paper_stream.engine.base_engine.llm.answer_question_stream(
            message, []
        ):
            full_answer = chunk
            yield "", history + [{"role": "user", "content": message},
                                {"role": "assistant", "content": full_answer}]

        # Get the full result with citations
        result = query_paper_stream.engine.query(message, use_rerank=True)
        citations = result.get("citations", [])

        # Add citation links if available
        if citations:
            citation_text = format_citations(citations)
            final_answer = result["answer"] + citation_text
        else:
            final_answer = result["answer"]

        yield "", history + [{"role": "user", "content": message},
                            {"role": "assistant", "content": final_answer}]

    except ValueError as e:
        error_msg = f"⚠️ Configuration Error: {str(e)}"
        logger.error(f"Configuration error: {e}")
        return "", history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": error_msg}]

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(f"Query processing error: {e}")
        return "", history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": error_msg}]


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
        type="messages",
        title="📚 ArXiv Research Assistant",
        description=(
            "Ask questions about machine learning research papers. "
            "I'll search through the database and provide answers with citations."
        ),
        examples=examples,
        theme=theme,
        cache_examples=False,
        save_history=True,
        autofocus=True,
        fill_height=True,
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
