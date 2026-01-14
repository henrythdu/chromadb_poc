"""Gradio chat interface for RAG paper Q&A."""
import logging
from typing import Any

import gradio as gr

from src.config import settings
from src.retrieval.engine import RAGQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize RAG engine globally (reuse across requests)
_rag_engine: RAGQueryEngine | None = None


def get_rag_engine() -> RAGQueryEngine:
    """Get or create the RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        logger.info("Initializing RAG engine...")
        _rag_engine = RAGQueryEngine(
            chroma_api_key=settings.chroma_cloud_api_key,
            chroma_tenant=settings.chroma_tenant,
            chroma_database=settings.chroma_database,
            cohere_api_key=settings.cohere_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            collection_name="arxiv_papers_v1",
        )
        logger.info("RAG engine initialized")
    return _rag_engine


def format_response(response: dict[str, Any]) -> tuple[str, str]:
    """Format RAG response for display.

    Args:
        response: Response dict from RAGQueryEngine.query()

    Returns:
        Tuple of (answer_text, citations_text)
    """
    answer = response.get("answer", "No answer generated.")
    citations = response.get("citations", [])
    sources = response.get("sources", [])

    # Format citations
    if citations:
        citations_text = "## Sources\n\n" + "\n".join(f"- {cite}" for cite in citations)
    else:
        citations_text = "## Sources\n\nNo sources available."

    # Add source count info
    if sources:
        citations_text += f"\n\n*Retrieved {len(sources)} source chunks*"

    return answer, citations_text


def query_paper(
    message: str,
    history: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]], str]:
    """Process user query through RAG pipeline.

    Args:
        message: User's question
        history: Chat history

    Returns:
        Tuple of (empty_str, updated_history, citations)
    """
    if not message.strip():
        return "", history, ""

    try:
        logger.info("Processing new query")

        # Get RAG engine and query
        engine = get_rag_engine()
        response = engine.query(message, use_rerank=True)

        # Format response
        answer, citations = format_response(response)

        # Update history with user message and bot response
        history.append((message, answer))

        logger.info(f"Response: {len(answer)} chars, {len(citations)} citations")

        return "", history, citations

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        error_msg = f"Sorry, an error occurred: {str(e)}"
        history.append((message, error_msg))
        return "", history, f"## Error\n\n{error_msg}"


def clear_history() -> tuple[str, str, list]:
    """Clear chat history.

    Returns:
        Tuple of (empty_message, empty_citations, empty_history)
    """
    return "", "", []


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface.

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(
        title="ArXiv Paper Q&A",
        theme=gr.themes.Soft(),
        css="""
        .chat-container {height: 500px;}
        .citation-box {background-color: #f0f0f0; padding: 15px; border-radius: 8px;}
        """
    ) as interface:
        gr.Markdown(
            """
            # 📚 ArXiv Research Paper Q&A

            Ask questions about machine learning research papers and get answers with citations.

            **Features:**
            - 🔍 Hybrid search (vector + keyword matching)
            - 🎯 Cohere Rerank v3 for relevance
            - 🤖 Claude 3.5 Sonnet for answers
            - 📖 Citations with paper references
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about machine learning concepts, papers, methods...",
                        scale=4,
                        lines=2,
                    )
                    submit = gr.Button("Submit", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation", variant="secondary")

            with gr.Column(scale=1):
                citations_output = gr.Markdown(
                    label="Sources",
                    value="*Ask a question to see sources*",
                    elem_classes=["citation-box"]
                )

        # Example questions
        gr.Examples(
            examples=[
                "What is attention in transformers?",
                "Explain backpropagation in neural networks",
                "What are the main challenges in reinforcement learning?",
                "How does batch normalization work?",
                "What is the difference between SGD and Adam?",
            ],
            inputs=msg,
        )

        # Event handlers
        submit.click(
            fn=query_paper,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, citations_output],
        )

        msg.submit(
            fn=query_paper,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, citations_output],
        )

        clear_btn.click(
            fn=clear_history,
            outputs=[msg, citations_output, chatbot],
        )

    return interface


if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()

    logger.info("Starting Gradio interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
