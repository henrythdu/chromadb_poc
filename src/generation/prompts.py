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
    from ..ingestion.metadata import MetadataBuilder

    metadata_builder = MetadataBuilder()

    # Format context with citations
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        metadata = chunk.get("metadata", {})
        citation = metadata_builder.format_citation(metadata)

        context_parts.append(f"{i}. {citation}\n{chunk['text']}")

    context_str = "\n\n".join(context_parts)

    return RAG_PROMPT.format(context_str=context_str, query_str=query)
