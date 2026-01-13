"""Test configuration loading."""

import os
import sys

import pytest
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_config_loads_from_env(monkeypatch):
    """Test that config loads from environment variables."""
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_HOST", "test_host")
    monkeypatch.setenv("CHROMA_API_KEY", "test_chroma_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    # Import after setting env vars
    from src.config import settings

    assert settings.llamaparse_api_key == "test_key"
    assert settings.chroma_host == "test_host"
    assert settings.chroma_api_key == "test_chroma_key"
    assert settings.openrouter_api_key == "test_openrouter_key"
    assert settings.cohere_api_key == "test_cohere_key"


def test_config_validation_fails_without_keys(monkeypatch):
    """Test that config validation fails without required keys."""
    # Clear all env vars
    for key in [
        "LLAMAPARSE_API_KEY",
        "CHROMA_HOST",
        "CHROMA_API_KEY",
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Force reload by clearing cached module
    if "src.config" in sys.modules:
        del sys.modules["src.config"]

    # Import should fail without env vars
    with pytest.raises(ValidationError):
        from src.config import settings  # noqa: F401
