"""Gradio chat interface for RAG system."""

import html
import logging
import os
import re

import gradio as gr
from dotenv import load_dotenv

from ..retrieval.engine import RAGQueryEngine

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ChatInterface:
    """Gradio chat interface for RAG queries."""

    def __init__(self):
        """Initialize the chat interface."""
        # Get API keys
        self.chroma_api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Initialize RAG engine (lazy load on first query)
        self.rag_engine = None
        self.engine_initialized = False

    def _ensure_engine(self):
        """Ensure RAG engine is initialized."""
        if self.engine_initialized:
            return

        try:
            # Validate required keys
            if not all(
                [
                    self.chroma_api_key,
                    self.chroma_tenant,
                    self.chroma_database,
                    self.cohere_api_key,
                    self.openrouter_api_key,
                ]
            ):
                raise ValueError(
                    "Missing required API keys. Please check your .env file."
                )

            # Initialize RAG engine
            self.rag_engine = RAGQueryEngine(
                chroma_api_key=self.chroma_api_key,
                chroma_tenant=self.chroma_tenant,
                chroma_database=self.chroma_database,
                cohere_api_key=self.cohere_api_key,
                openrouter_api_key=self.openrouter_api_key,
            )

            self.engine_initialized = True
            logger.info("RAG engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise

    def validate_input(self, query: str) -> tuple[bool, str, str]:
        """Validate user input.

        Args:
            query: User query string

        Returns:
            Tuple of (is_valid, error_message, sanitized_query)
        """
        # Check minimum length
        if len(query) < 10:
            return False, "Query must be at least 10 characters long.", ""

        # Check maximum length
        if len(query) > 500:
            return False, "Query must be less than 500 characters.", ""

        # Sanitize input - remove HTML/XML tags
        sanitized = re.sub(r"<[^>]+>", "", query)

        # Check if HTML tags were present (potential injection)
        if len(sanitized) < len(query):
            # Tags were removed, this is suspicious
            return False, "Invalid input: please use plain text only.", ""

        # Remove excessive whitespace
        sanitized = " ".join(sanitized.split())

        return True, "", sanitized

    def format_citations(self, citations: list[str]) -> str:
        """Format citations as clickable ArXiv links.

        Args:
            citations: List of citation strings

        Returns:
            Formatted HTML string with links
        """
        if not citations:
            return ""

        html_parts = ["<h4>Sources:</h4><ul>"]

        for citation in citations:
            # Extract arxiv ID from citation
            # Format: "Author et al. (arxiv:1234.56789, page X)"
            match = re.search(r"arxiv:(\d+\.\d+)", citation)
            # Escape citation to prevent XSS
            safe_citation = html.escape(citation)
            if match:
                arxiv_id = match.group(1)
                url = f"https://arxiv.org/abs/{arxiv_id}"
                html_parts.append(
                    f'<li><a href="{url}" target="_blank">{safe_citation}</a></li>'
                )
            else:
                html_parts.append(f"<li>{safe_citation}</li>")

        html_parts.append("</ul>")
        return "\n".join(html_parts)

    def process_query(
        self,
        message: str,
        history: list[tuple[str, str]],
    ) -> tuple[str, list[tuple[str, str]]]:
        """Process a user query through the RAG pipeline.

        Args:
            message: User message
            history: Conversation history

        Returns:
            Tuple of (response, updated_history)
        """
        try:
            # Validate input and get sanitized query
            is_valid, error_msg, sanitized_message = self.validate_input(message)
            if not is_valid:
                return error_msg, history + [[message, error_msg]]

            # Ensure engine is initialized
            if not self.engine_initialized:
                self._ensure_engine()

            # Query the RAG engine with sanitized message
            result = self.rag_engine.query(sanitized_message, use_rerank=True)

            # Format response
            answer = result.get("answer", "No answer generated.")
            citations = result.get("citations", [])

            # Add citation links if available
            if citations:
                citation_html = self.format_citations(citations)
                response = f"{answer}\n\n{citation_html}"
            else:
                response = answer

            # Update history
            history = history + [[message, response]]

            return "", history

        except ValueError as e:
            error_msg = f"⚠️ Configuration Error: {str(e)}"
            logger.error(f"Configuration error: {e}")
            return error_msg, history + [[message, error_msg]]

        except Exception as e:
            error_msg = f"❌ Error processing your query: {str(e)}"
            logger.error(f"Query processing error: {e}")
            return error_msg, history + [[message, error_msg]]

    def clear_history(self) -> list[tuple[str, str]]:
        """Clear conversation history.

        Returns:
            Empty history
        """
        return []


def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    chat = ChatInterface()

    with gr.Blocks(title="ArXiv Research Assistant") as interface:
        gr.Markdown(
            """
            # 📚 ArXiv Research Assistant

            Ask questions about machine learning research papers. I'll search through the database
            and provide answers with citations to the relevant papers.

            **Example questions:**
            - What is attention in transformer models?
            - How does reinforcement learning work?
            - Explain the concept of gradient descent
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )

            with gr.Column(scale=1):
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about ML research...",
                        lines=3,
                        scale=4,
                    )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        gr.Markdown(
            """
            ---
            **Note:** This is a research prototype using ChromaDB vector database and LLM-based retrieval.
            Responses are generated based on the indexed papers and may not always be accurate.
            """
        )

        # Event handlers
        submit_btn.click(
            fn=chat.process_query,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        msg.submit(
            fn=chat.process_query,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )

        clear_btn.click(
            fn=chat.clear_history,
            outputs=[chatbot],
        )

    return interface


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
        theme=gr.themes.Soft(),
        css="""
        .container {max-width: 900px; margin: auto;}
        .chat-container {height: 500px;}
        """,
    )


if __name__ == "__main__":
    launch()
