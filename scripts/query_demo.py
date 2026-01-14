#!/usr/bin/env python3
"""Interactive demo script for querying the RAG system."""

import logging
import os
import sys

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval.engine import RAGQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run interactive query demo."""
    # Load environment variables
    load_dotenv()

    # Get API keys
    chroma_api_key = os.getenv("CHROMA_CLOUD_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    # Validate required keys
    missing = []
    if not chroma_api_key:
        missing.append("CHROMA_CLOUD_API_KEY")
    if not chroma_tenant:
        missing.append("CHROMA_TENANT")
    if not chroma_database:
        missing.append("CHROMA_DATABASE")
    if not cohere_api_key:
        missing.append("COHERE_API_KEY")
    if not openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please set them in your .env file")
        sys.exit(1)

    # Initialize RAG engine
    logger.info("Initializing RAG Query Engine...")
    engine = RAGQueryEngine(
        chroma_api_key=chroma_api_key,
        chroma_tenant=chroma_tenant,
        chroma_database=chroma_database,
        cohere_api_key=cohere_api_key,
        openrouter_api_key=openrouter_api_key,
    )

    print("\n" + "=" * 60)
    print("RAG Query Demo - ArXiv Papers")
    print("=" * 60)
    print("\nAsk questions about machine learning research papers.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Example questions
    examples = [
        "What is attention in transformer models?",
        "How does reinforcement learning work?",
        "Explain the concept of gradient descent.",
        "What are the limitations of large language models?",
    ]

    print(f"Example questions you can ask:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex}")

    print()

    # Interactive loop
    while True:
        try:
            question = input("\nYour question: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\n" + "-" * 60)
            print("Querying...")
            print("-" * 60)

            # Query the engine
            result = engine.query(question, use_rerank=True)

            # Display answer
            print("\n📝 Answer:")
            print(result["answer"])

            # Display citations
            if result["citations"]:
                print("\n📚 Citations:")
                for i, citation in enumerate(result["citations"], 1):
                    print(f"  {i}. {citation}")

            # Display sources count
            sources_count = len(result["sources"])
            print(f"\n📊 Used {sources_count} source chunks")

            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during query: {e}")
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
