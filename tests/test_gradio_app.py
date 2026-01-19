"""Test Gradio UI components."""


def test_gradio_app_imports():
    """Test that gradio_app module can be imported."""
    from src.ui.gradio_app import (
        create_interface,
        format_citations,
        launch,
        query_paper_stream,
        validate_input,
    )

    assert validate_input is not None
    assert format_citations is not None
    assert query_paper_stream is not None
    assert create_interface is not None
    assert launch is not None


def test_input_validation_too_short():
    """Test input validation rejects too short queries."""
    from src.ui.gradio_app import validate_input

    is_valid, error_msg = validate_input("short")

    assert not is_valid
    assert "at least 10 characters" in error_msg


def test_input_validation_too_long():
    """Test input validation rejects too long queries."""
    from src.ui.gradio_app import validate_input

    long_query = "a" * 501  # 501 characters
    is_valid, error_msg = validate_input(long_query)

    assert not is_valid
    assert "less than 500" in error_msg


def test_input_valid():
    """Test input validation accepts valid queries."""
    from src.ui.gradio_app import validate_input

    valid_query = "What is attention in transformer models?"
    is_valid, error_msg = validate_input(valid_query)

    assert is_valid
    assert error_msg == ""


def test_format_citations():
    """Test citation formatting with ArXiv links."""
    from src.ui.gradio_app import format_citations

    citations = [
        "Author et al. (arxiv:1706.03762, page 1)",
        "Another Author (arxiv:2301.07041, page 5)",
    ]

    formatted = format_citations(citations)

    assert "https://arxiv.org/abs/1706.03762" in formatted
    assert "https://arxiv.org/abs/2301.07041" in formatted
    assert "Sources:" in formatted  # Updated for emoji format


def test_format_citations_empty():
    """Test citation formatting with empty list."""
    from src.ui.gradio_app import format_citations

    formatted = format_citations([])

    assert formatted == ""


def test_format_citations_no_arxiv_id():
    """Test citation formatting without ArXiv ID."""
    from src.ui.gradio_app import format_citations

    citations = ["Some citation without arxiv ID"]

    formatted = format_citations(citations)

    assert "Some citation without arxiv ID" in formatted
    assert formatted.count("- ") == 1


def test_create_interface():
    """Test Gradio interface creation."""
    import gradio as gr

    from src.ui.gradio_app import create_interface

    interface = create_interface()

    assert isinstance(interface, gr.ChatInterface)


def test_query_paper_stream_generator():
    """Test that query_paper_stream is a generator function."""
    from src.ui.gradio_app import query_paper_stream

    import inspect

    assert inspect.isgeneratorfunction(query_paper_stream)
