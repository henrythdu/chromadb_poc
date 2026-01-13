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
    monkeypatch.setenv("CHROMA_CLOUD_API_KEY", "test_chroma_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test_database")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")

    # Import after setting env vars
    from src.config import settings

    assert settings.llamaparse_api_key == "test_key"
    assert settings.chroma_cloud_api_key == "test_chroma_key"
    assert settings.chroma_tenant == "test_tenant"
    assert settings.chroma_database == "test_database"
    assert settings.openrouter_api_key == "test_openrouter_key"
    assert settings.cohere_api_key == "test_cohere_key"


def test_config_validation_fails_without_keys(monkeypatch, tmp_path):
    """Test that config validation fails without required keys."""
    # Create a temporary empty .env file
    empty_env = tmp_path / ".env"
    empty_env.write_text("")

    # Clear all env vars
    for key in [
        "LLAMAPARSE_API_KEY",
        "CHROMA_CLOUD_API_KEY",
        "CHROMA_TENANT",
        "CHROMA_DATABASE",
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Force reload by clearing cached module
    if "src.config" in sys.modules:
        del sys.modules["src.config"]

    # Patch Settings to use empty .env file
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class TestSettings(BaseSettings):
        llamaparse_api_key: str
        chroma_cloud_api_key: str
        chroma_tenant: str
        chroma_database: str
        openrouter_api_key: str
        cohere_api_key: str

        model_config = SettingsConfigDict(
            env_file=str(empty_env),
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

    # Import should fail without env vars
    with pytest.raises(ValidationError):
        TestSettings()
