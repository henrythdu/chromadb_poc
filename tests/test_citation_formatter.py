"""Tests for CitationFormatter."""

import pytest

from src.retrieval.citation_formatter import CitationFormatter


def test_format_arxiv_paper_complete():
    """Test formatting complete ArXiv paper citation."""
    formatter = CitationFormatter()

    metadata = {
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al."
    }

    result = formatter.format_citation(metadata)

    # Should contain title, authors, arxiv ID, and markdown link
    assert "Attention Is All You Need" in result
    assert "Vaswani et al." in result
    assert "arxiv:2301.07041" in result
    assert "https://arxiv.org/abs/2301.07041" in result
    assert result.startswith("[") and "](" in result  # Markdown link format


def test_format_arxiv_paper_missing_fields():
    """Test formatting ArXiv paper with missing fields uses fallbacks."""
    formatter = CitationFormatter()

    metadata = {
        "arxiv_id": "2301.07041",
        # Missing title and authors
    }

    result = formatter.format_citation(metadata)

    assert "Unknown Title" in result
    assert "Unknown Authors" in result
    assert "arxiv:2301.07041" in result


def test_format_contract_complete():
    """Test formatting complete contract citation."""
    formatter = CitationFormatter()

    metadata = {
        "document_id": "abc123",
        "company_name": "Acme Corp",
        "contract_type": "Co_Branding",
        "execution_date": "2020-01-15",
        "filing_type": "S-1",
        "exhibit_number": "EX-10.1"
    }

    result = formatter.format_citation(metadata)

    assert "Acme Corp" in result
    assert "Co_Branding" in result
    assert "2020-01-15" in result
    assert "Filing: S-1" in result
    assert "Exhibit: EX-10.1" in result


def test_format_contract_missing_fields():
    """Test formatting contract with missing fields uses fallbacks."""
    formatter = CitationFormatter()

    metadata = {
        "document_id": "abc123",
        "company_name": "Acme Corp",
        # Missing other fields
    }

    result = formatter.format_citation(metadata)

    assert "Acme Corp" in result
    assert "Unknown Agreement" in result
    assert "Unknown Date" in result


def test_format_citation_auto_detects_paper():
    """Test format_citation auto-detects ArXiv paper from arxiv_id field."""
    formatter = CitationFormatter()

    metadata = {
        "arxiv_id": "2301.07041",
        "title": "Test Paper"
    }

    result = formatter.format_citation(metadata)

    assert "arxiv:2301.07041" in result
    assert "https://arxiv.org/abs/2301.07041" in result


def test_format_citation_auto_detects_contract():
    """Test format_citation auto-detects contract from document_id field."""
    formatter = CitationFormatter()

    metadata = {
        "document_id": "abc123",
        "company_name": "Test Corp",
        "contract_type": "Test Agreement"
    }

    result = formatter.format_citation(metadata)

    assert "Test Corp" in result
    assert "Test Agreement" in result


def test_format_citation_unknown_type():
    """Test format_citation handles unknown metadata type."""
    formatter = CitationFormatter()

    metadata = {
        "unknown_field": "some_value"
    }

    result = formatter.format_citation(metadata)

    assert "Unknown source" in result
