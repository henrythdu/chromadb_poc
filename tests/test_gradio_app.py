"""Test Gradio UI components."""


def test_gradio_app_imports():
    """Test that gradio_app module can be imported."""
    from src.ui.gradio_app import ChatInterface, create_interface, launch

    assert ChatInterface is not None
    assert create_interface is not None
    assert launch is not None


def test_chat_interface_initialization():
    """Test ChatInterface initialization."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()

    assert chat is not None
    assert not chat.engine_initialized


def test_input_validation_too_short():
    """Test input validation rejects too short queries."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()
    is_valid, error_msg, sanitized = chat.validate_input("short")

    assert not is_valid
    assert "at least 10 characters" in error_msg
    assert sanitized == ""


def test_input_validation_too_long():
    """Test input validation rejects too long queries."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()
    long_query = "a" * 501  # 501 characters
    is_valid, error_msg, sanitized = chat.validate_input(long_query)

    assert not is_valid
    assert "less than 500" in error_msg
    assert sanitized == ""


def test_input_valid():
    """Test input validation accepts valid queries."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()
    valid_query = "What is attention in transformer models?"
    is_valid, error_msg, sanitized = chat.validate_input(valid_query)

    assert is_valid
    assert error_msg == ""
    assert sanitized == valid_query


def test_input_sanitization_html():
    """Test input validation rejects HTML tags."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()
    html_query = (
        "<script>alert('xss')</script>" + "a" * 50
    )  # Add padding to pass length check
    is_valid, error_msg, sanitized = chat.validate_input(html_query)

    assert not is_valid
    assert "plain text only" in error_msg
    assert sanitized == ""


def test_input_sanitization_whitespace():
    """Test input validation removes excessive whitespace."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()
    query_with_spaces = "What    is   attention   in    transformers?   "
    is_valid, error_msg, sanitized = chat.validate_input(query_with_spaces)

    assert is_valid
    assert error_msg == ""
    assert sanitized == "What is attention in transformers?"


def test_format_citations():
    """Test citation formatting with ArXiv links."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()

    citations = [
        "Author et al. (arxiv:1706.03762, page 1)",
        "Another Author (arxiv:2301.07041, page 5)",
    ]

    formatted = chat.format_citations(citations)

    assert "https://arxiv.org/abs/1706.03762" in formatted
    assert "https://arxiv.org/abs/2301.07041" in formatted
    assert 'target="_blank"' in formatted
    assert "<a href=" in formatted


def test_clear_history():
    """Test clearing conversation history."""
    from src.ui.gradio_app import ChatInterface

    chat = ChatInterface()
    _ = [["question1", "answer1"], ["question2", "answer2"]]

    cleared = chat.clear_history()

    assert cleared == []


def test_create_interface():
    """Test Gradio interface creation."""
    import gradio as gr

    from src.ui.gradio_app import create_interface

    interface = create_interface()

    assert isinstance(interface, gr.Blocks)
