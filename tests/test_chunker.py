"""Test document chunking."""


def test_chunker_imports():
    """Test that chunker module can be imported."""
    from src.ingestion.chunker import DocumentChunker

    assert DocumentChunker is not None


def test_chunker_initialization():
    """Test DocumentChunker initialization."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)

    assert chunker.chunk_size == 800
    assert chunker.chunk_overlap == 100


def test_chunk_markdown():
    """Test chunking markdown content."""
    from src.ingestion.chunker import DocumentChunker

    # Use smaller chunk size to test splitting
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)

    # Create longer markdown content that will exceed chunk_size
    long_content = (
        """
    This is detailed content that discusses various aspects of machine learning and neural networks.
    """
        * 50
    )  # Repeat to create enough content for multiple chunks

    markdown = f"""
    # Introduction

    {long_content}

    ## Methods

    {long_content}

    ## Results

    {long_content}
    """

    chunks = chunker.chunk_markdown(
        markdown=markdown,
        metadata={
            "document_id": "test123",
            "title": "Test Paper",
        },
    )

    assert len(chunks) > 1
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["document_id"] == "test123"


def test_metadata_enrichment():
    """Test that metadata is properly enriched."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker()

    base_metadata = {
        "document_id": "2301.07041",
        "arxiv_id": "2301.07041",
        "title": "Test Paper",
        "section": "Introduction",
    }

    chunk = {
        "text": "This is test content.",
        "metadata": {
            **base_metadata,
            "chunk_index": 0,
            "page_number": 1,
        },
    }

    # Test enrich_metadata method
    enriched = chunker.enrich_metadata(chunk, page_number=1)

    assert enriched["metadata"]["document_id"] == "2301.07041"
    assert enriched["metadata"]["chunk_index"] == 0
    assert enriched["metadata"]["page_number"] == 1
    # Check that the header was added
    assert "[Document:" in enriched["text"]
