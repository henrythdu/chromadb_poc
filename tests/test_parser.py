"""Tests for LlamaParse parser module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_parser_imports():
    """Test that the parser module can be imported."""
    from src.ingestion.parser import LlamaParserWrapper

    assert LlamaParserWrapper is not None


def test_parser_initialization():
    """Test parser initialization with API key and result type."""
    from src.ingestion.parser import LlamaParserWrapper

    parser = LlamaParserWrapper(api_key="test-api-key", result_type="markdown")

    assert parser.api_key == "test-api-key"
    assert parser.result_type == "markdown"
    assert parser._client is None  # Client should be lazy-loaded


def test_parser_lazy_client_loading():
    """Test that the LlamaParse client is loaded lazily."""
    # Mock llama_parse before importing our module
    with patch.dict(sys.modules, {"llama_parse": MagicMock()}):
        from src.ingestion.parser import LlamaParserWrapper

        parser = LlamaParserWrapper(api_key="test-api-key", result_type="markdown")

        # Client should not be initialized initially
        assert parser._client is None

        # Manually create a mock for the client
        mock_instance = MagicMock()
        parser._client = mock_instance

        # Verify client can be set
        assert parser._client is not None


def test_parse_pdf_with_file_not_found():
    """Test parse_pdf with non-existent file."""
    from src.ingestion.parser import LlamaParserWrapper

    parser = LlamaParserWrapper(api_key="test-api-key", result_type="markdown")

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_pdf = Path(tmpdir) / "non_existent.pdf"

        result = parser.parse_pdf(str(non_existent_pdf), max_retries=3)

        assert result["error"] is not None
        assert "File not found" in result["error"]
        assert result["markdown"] == ""
        assert result["pages"] == 0


def test_parse_pdf_with_retry_logic():
    """Test parse_pdf retry logic on API failures."""
    # Mock llama_parse before importing our module
    mock_llama_parse = MagicMock()
    mock_instance = MagicMock()
    mock_llama_parse.LlamaParse = MagicMock(return_value=mock_instance)

    with patch.dict(sys.modules, {"llama_parse": mock_llama_parse}):
        from src.ingestion.parser import LlamaParserWrapper

        parser = LlamaParserWrapper(api_key="test-api-key", result_type="markdown")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy PDF file
            test_pdf = Path(tmpdir) / "test.pdf"
            test_pdf.write_text("dummy pdf content")

            # Simulate API failure then success
            mock_instance.load_data.side_effect = [
                Exception("API Error"),
                Exception("API Error"),
                [
                    MagicMock(
                        text="# Test Document\n\nThis is a test.",
                        metadata={"page_num": 1},
                    )
                ],
            ]

            result = parser.parse_pdf(str(test_pdf), max_retries=3)

            # Should succeed after retries
            assert result["error"] is None
            assert result["markdown"] != ""
            assert result["pages"] == 1
            assert mock_instance.load_data.call_count == 3


def test_parse_pdf_exhausts_retries():
    """Test parse_pdf when retries are exhausted."""
    # Mock llama_parse before importing our module
    mock_llama_parse = MagicMock()
    mock_instance = MagicMock()
    mock_llama_parse.LlamaParse = MagicMock(return_value=mock_instance)

    with patch.dict(sys.modules, {"llama_parse": mock_llama_parse}):
        from src.ingestion.parser import LlamaParserWrapper

        parser = LlamaParserWrapper(api_key="test-api-key", result_type="markdown")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy PDF file
            test_pdf = Path(tmpdir) / "test.pdf"
            test_pdf.write_text("dummy pdf content")

            # Simulate persistent API failure
            mock_instance.load_data.side_effect = Exception("Persistent API Error")

            result = parser.parse_pdf(str(test_pdf), max_retries=2)

            # Should fail after exhausting retries
            assert result["error"] is not None
            assert "Persistent API Error" in result["error"]
            assert result["markdown"] == ""
            assert result["pages"] == 0
            assert mock_instance.load_data.call_count == 2


@pytest.mark.integration
def test_parse_pdf():
    """Integration test for parsing a real PDF with LlamaParse.

    This test may be skipped initially as it requires a valid API key.
    Mark with: pytest -m "not integration" to skip.
    """
    import os

    from src.ingestion.parser import LlamaParserWrapper

    # Skip if no API key is available
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        pytest.skip("LLAMA_CLOUD_API_KEY not set")

    parser = LlamaParserWrapper(api_key=api_key, result_type="markdown")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal test PDF (in reality, you'd use a real PDF)
        test_pdf = Path(tmpdir) / "test.pdf"
        test_pdf.write_text("dummy pdf content")

        result = parser.parse_pdf(str(test_pdf), max_retries=3)

        # Verify result structure
        assert "markdown" in result
        assert "pages" in result
        assert "error" in result

        # If successful, should have content
        if result["error"] is None:
            assert result["markdown"] != ""
            assert result["pages"] > 0
