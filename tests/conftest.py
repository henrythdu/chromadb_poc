"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "llm_model": "test-model",
        "embedding_model": "test-embedding",
    }
