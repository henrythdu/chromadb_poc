# Code Style & Conventions - ChromaDB POC

## Python Conventions
- Follow PEP 8 style guide
- Use type hints for function signatures
- Use descriptive variable and function names (snake_case)
- Class names use PascalCase
- Constants use UPPER_SNAKE_CASE

## Docstrings
- Use Google-style docstrings
- Include description, args, returns, raises where applicable

## Example
```python
def process_documents(docs: list[str]) -> dict[str, any]:
    """Process a list of documents for vector embedding.

    Args:
        docs: List of document strings to process.

    Returns:
        Dictionary containing processed results and metadata.

    Raises:
        ValueError: If docs is empty.
    """
    ...
```

## Formatting
- **Tool**: ruff
- **Line length**: 88 (ruff default)
- **Quote style**: double quotes
- **Imports**: Grouped (stdlib, third-party, local)

## Linting
- **Tool**: ruff
- Run `ruff check .` before committing
- Fix auto-fixable issues with `ruff check --fix .`
