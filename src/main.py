"""Main entry point for ChromaDB POC."""
import logging
import sys

from src.ui.gradio_app import launch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Launch the Gradio application."""
    try:
        logger.info("Starting ArXiv Research Assistant...")

        # Launch Gradio interface
        launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Set to True for public link
        )

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
