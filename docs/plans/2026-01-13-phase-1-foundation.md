# Phase 1 - Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up project infrastructure with validated configuration, tooling, and test framework.

**Architecture:** Python project structure with pydantic settings, ruff linting, pytest testing, and environment-based configuration.

**Tech Stack:** Python 3.10+, pydantic-settings, pytest, ruff, toml

---

## Task 1: Create Project Directory Structure

**Files:**
- Create: `src/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/retrieval/__init__.py`
- Create: `src/generation/__init__.py`
- Create: `src/ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `logs/.gitkeep`
- Create: `data/.gitkeep`

**Step 1: Create source directories**

```bash
mkdir -p src/ingestion src/retrieval src/generation src/ui tests logs data
```

**Step 2: Create __init__.py files**

```bash
# src/__init__.py
touch src/__init__.py src/ingestion/__init__.py src/retrieval/__init__.py
touch src/generation/__init__.py src/ui/__init__.py tests/__init__.py
```

**Step 3: Create placeholder files**

```bash
# tests/conftest.py - pytest fixtures
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and fixtures."""
import pytest

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "llm_model": "test-model",
        "embedding_model": "test-embedding",
    }
EOF

# logs/.gitkeep - keep empty dir in git
touch logs/.gitkeep data/.gitkeep
```

**Step 4: Verify structure**

```bash
tree src/ tests/ -L 2
```

Expected output:
```
src/
├── __init__.py
├── ingestion/
│   └── __init__.py
├── retrieval/
│   └── __init__.py
├── generation/
│   └── __init__.py
└── ui/
    └── __init__.py
tests/
├── __init__.py
└── conftest.py
```

**Step 5: Commit**

```bash
git add src/ tests/ logs/ data/
git commit -m "feat: create project directory structure"
```

---

## Task 2: Create requirements.txt

**Files:**
- Create: `requirements.txt`

**Step 1: Write requirements.txt**

```bash
cat > requirements.txt << 'EOF'
# Core RAG
llama-index>=0.10.0
llama-index-llms-openrouter>=0.1.0
llama-index-vector-stores-chroma>=0.1.0
llama-index-node-parser>=0.1.0

# Parsing
llama-parse>=0.4.0

# Database & APIs
chromadb>=0.4.0
cohere>=5.0.0
arxiv>=2.0.0

# UI
gradio>=4.0.0

# Config
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Development
pytest>=7.0.0
pytest-cov>=4.0.0
ruff>=0.1.0
EOF
```

**Step 2: Verify format**

```bash
cat requirements.txt
```

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add project dependencies"
```

---

## Task 3: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

**Step 1: Write pyproject.toml**

```bash
cat > pyproject.toml << 'EOF'
[project]
name = "chromadb-poc"
version = "0.1.0"
requires-python = ">=3.10"
description = "ArXiv RAG system with ChromaDB"

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*"]
EOF
```

**Step 2: Verify ruff can read config**

```bash
python -m ruff check --help | head -5
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: configure ruff and pytest"
```

---

## Task 4: Create .env.example Template

**Files:**
- Create: `.env.example`

**Step 1: Write .env.example**

```bash
cat > .env.example << 'EOF'
# LlamaParse API Key
LLAMAPARSE_API_KEY=your_llamaparse_api_key_here

# Chroma Cloud
CHROMA_HOST=your_chroma_cloud_host_here
CHROMA_API_KEY=your_chroma_api_key_here

# OpenRouter (LLM Provider)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Cohere Rerank
COHERE_API_KEY=your_cohere_api_key_here
EOF
```

**Step 2: Verify file exists**

```bash
cat .env.example
```

**Step 3: Commit**

```bash
git add .env.example
git commit -m "feat: add .env.example template"
```

---

## Task 5: Create config.toml

**Files:**
- Create: `config.toml`

**Step 1: Write config.toml**

```bash
cat > config.toml << 'EOF'
[models]
llm = "anthropic/claude-3.5-sonnet"
embedding = "BAAI/bge-large-en-v1.5"
embedding_provider = "openrouter"

[pipeline]
max_papers = 200
chunk_size = 800
chunk_overlap = 100
top_k_retrieve = 50
top_k_rerank = 5

[paths]
pdf_dir = "./ml_pdfs"
log_dir = "./logs"
EOF
```

**Step 2: Verify TOML is valid**

```bash
python -c "import tomli; print(tomli.load(open('config.toml')))" 2>/dev/null || \
python -c "import sys; import tomllib; print(tomllib.load(open('config.toml', 'rb')))"
```

Expected: Dictionary with models, pipeline, paths keys

**Step 3: Commit**

```bash
git add config.toml
git commit -m "feat: add config.toml with app settings"
```

---

## Task 6: Implement config.py with pydantic

**Files:**
- Create: `src/config.py`

**Step 1: Write failing test first**

```bash
cat > tests/test_config.py << 'EOF'
"""Test configuration loading."""
import os
import pytest
from pydantic import ValidationError

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
    for key in ["LLAMAPARSE_API_KEY", "CHROMA_HOST", "CHROMA_API_KEY",
                "OPENROUTER_API_KEY", "COHERE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    # Force reload by clearing cached module
    import sys
    if 'src.config' in sys.modules:
        del sys.modules['src.config']

    with pytest.raises(ValidationError):
        from src.config import settings
EOF
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'src.config'"

**Step 3: Implement config.py**

```bash
cat > src/config.py << 'EOF'
"""Configuration management using pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (required)
    llamaparse_api_key: str
    chroma_host: str
    chroma_api_key: str
    openrouter_api_key: str
    cohere_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance
settings = Settings()
EOF
```

**Step 4: Run test to verify it passes**

```bash
LLAMAPARSE_API_KEY=test CHROMA_HOST=test CHROMA_API_KEY=test \
OPENROUTER_API_KEY=test COHERE_API_KEY=test pytest tests/test_config.py::test_config_loads_from_env -v
```

Expected: PASS

**Step 5: Run ruff check**

```bash
ruff check src/config.py tests/test_config.py
```

Expected: No errors

**Step 6: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: implement config.py with pydantic validation"
```

---

## Task 7: Set Up Basic pytest Framework

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Enhance conftest.py with fixtures**

```bash
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and shared fixtures."""
import pytest
import os
import sys


# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Provide mock API keys for testing."""
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "test_llamaparse_key")
    monkeypatch.setenv("CHROMA_HOST", "http://test-chroma.com")
    monkeypatch.setenv("CHROMA_API_KEY", "test_chroma_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    return {
        "llamaparse": "test_llamaparse_key",
        "chroma": {"host": "http://test-chroma.com", "key": "test_chroma_key"},
        "openrouter": "test_openrouter_key",
        "cohere": "test_cohere_key",
    }


@pytest.fixture
def sample_arxiv_metadata():
    """Sample ArXiv paper metadata for testing."""
    return {
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "published_date": "2023-01-17",
        "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
    }


@pytest.fixture
def sample_chunk():
    """Sample document chunk for testing."""
    return {
        "document_id": "2301.07041",
        "arxiv_id": "2301.07041",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "page_number": 1,
        "section": "Introduction",
        "chunk_index": 0,
        "text": "The dominant sequence transduction models...",
    }
EOF
```

**Step 2: Write placeholder test file**

```bash
cat > tests/test_placeholder.py << 'EOF'
"""Placeholder tests to verify pytest works."""
import pytest


def test_pytest_works():
    """Verify pytest is functioning."""
    assert True


def test_fixture_works(mock_api_keys):
    """Verify fixtures load correctly."""
    assert mock_api_keys["llamaparse"] == "test_llamaparse_key"
    assert mock_api_keys["chroma"]["host"] == "http://test-chroma.com"


def test_sample_fixtures(sample_arxiv_metadata, sample_chunk):
    """Verify sample data fixtures work."""
    assert sample_arxiv_metadata["arxiv_id"] == "2301.07041"
    assert sample_chunk["section"] == "Introduction"
EOF
```

**Step 3: Run tests**

```bash
pytest tests/ -v
```

Expected: All 3 tests PASS

**Step 4: Run with coverage**

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

**Step 5: Commit**

```bash
git add tests/conftest.py tests/test_placeholder.py
git commit -m "test: set up pytest framework with fixtures"
```

---

## Task 8: Phase 1 Gate Testing

**Files:**
- Test: All Phase 1 components

**Step 1: Verify pytest runs successfully**

```bash
pytest tests/ -v
```

Expected: All tests pass (at minimum test_pytest_works, test_fixture_works, test_sample_fixtures, test_config_loads_from_env)

**Step 2: Verify ruff check passes**

```bash
ruff check .
```

Expected: No errors

**Step 3: Verify ruff format check passes**

```bash
ruff format --check .
```

Expected: No files need formatting

**Step 4: Verify config loads without errors**

```bash
LLAMAPARSE_API_KEY=test CHROMA_HOST=test CHROMA_API_KEY=test \
OPENROUTER_API_KEY=test COHERE_API_KEY=test \
python -c "from src.config import settings; print('Config loaded successfully')"
```

Expected: "Config loaded successfully"

**Step 5: Verify all files exist**

```bash
ls -la src/config.py requirements.txt pyproject.toml .env.example config.toml
```

Expected: All 5 files exist

**Step 6: Run full test suite with coverage**

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

Expected: Coverage report generated

**Step 7: Verify directory structure**

```bash
tree -L 2 src/ tests/
```

Expected: All module directories with __init__.py

**Step 8: Create gate test summary**

```bash
cat > docs/plans/phase-1-gate-summary.md << 'EOF'
# Phase 1 Gate Test Results

## Date: [Run Date]

## Checklist

- [x] pytest runs successfully (N tests, 0 failures)
- [x] ruff check passes with no errors
- [x] ruff format check passes
- [x] config.py loads from environment
- [x] All project files exist
- [x] Directory structure complete
- [x] Test coverage report generated

## Status: PASSED ✅

Phase 1 Foundation is complete. Ready to proceed to Phase 2 - Ingestion Pipeline.
EOF
```

**Step 9: Commit**

```bash
git add docs/plans/phase-1-gate-summary.md
git commit -m "test: phase 1 gate testing complete"
```

---

## Summary

After completing all tasks, you should have:

1. ✅ Complete project directory structure
2. ✅ All dependencies in requirements.txt
3. ✅ Ruff and pytest configured in pyproject.toml
4. ✅ .env.example template for API keys
5. ✅ config.toml with application settings
6. ✅ config.py with pydantic validation
7. ✅ Pytest framework with fixtures
8. ✅ All gate tests passing

**Next Phase:** Phase 2 - Ingestion Pipeline
