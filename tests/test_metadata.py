"""Test metadata enrichment."""


def test_metadata_builder():
    """Test MetadataBuilder class."""
    from src.ingestion.metadata import MetadataBuilder

    builder = MetadataBuilder()

    paper_metadata = {
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "published_date": "2023-01-17",
        "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
    }

    enriched = builder.build_document_metadata(paper_metadata)

    assert enriched["document_id"] == "2301.07041"
    assert enriched["arxiv_id"] == "2301.07041"
    assert "authors" in enriched


def test_chunk_metadata():
    """Test chunk-level metadata enrichment."""
    from src.ingestion.metadata import MetadataBuilder

    builder = MetadataBuilder()

    doc_metadata = builder.build_document_metadata(
        {
            "arxiv_id": "2301.07041",
            "title": "Test Paper",
            "authors": ["Author One"],
        }
    )

    chunk_metadata = builder.build_chunk_metadata(
        doc_metadata=doc_metadata,
        chunk_index=0,
        section="Introduction",
        page_number=1,
    )

    assert chunk_metadata["chunk_index"] == 0
    assert chunk_metadata["section"] == "Introduction"
    assert chunk_metadata["page_number"] == 1
    assert chunk_metadata["document_id"] == "2301.07041"


def test_format_citation():
    """Test citation formatting."""
    from src.ingestion.metadata import MetadataBuilder

    builder = MetadataBuilder()

    metadata = {
        "title": "Attention Is All You Need",
        "arxiv_id": "2301.07041",
        "page_number": 3,
    }

    citation = builder.format_citation(metadata)

    assert "Attention Is All You Need" in citation
    assert "2301.07041" in citation
    assert "page 3" in citation
