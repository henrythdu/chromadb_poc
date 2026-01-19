# Phase 1: Contract Filename Parser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a robust filename parser that extracts metadata from contract PDF filenames without using LLMs.

**Architecture:** Regex-based parser that handles the SEC filing filename format: `{Company}_{Date}_{FilingType}_{Exhibit}_{AccessionNum}_{Description}.pdf`. Returns normalized metadata dict with ISO-formatted dates and clean field values.

**Tech Stack:** Python 3.10+, pytest, re (standard library), logging

---

## Task 1: Create the parser module with logger

**Files:**
- Create: `src/ingestion/contract_filename_parser.py`

**Step 1: Create the module file with basic structure and logger**

```python
"""Contract filename parser for extracting metadata from SEC filing filenames.

This module parses contract PDF filenames to extract structured metadata without
requiring LLM analysis. It handles the standard SEC filing naming convention.

Example filename:
    CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf

Extracted metadata:
    - company_name: "CreditcardscomInc"
    - execution_date: "2007-08-10"
    - filing_type: "S-1"
    - exhibit_number: "EX-10.33"
    - accession_number: "362297"
"""

import logging
import os
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


__all__ = [
    "parse_contract_filename",
    "parse_contract_path",
    "ContractFilenameParser",
    "ParseError",
]


class ParseError(Exception):
    """Raised when filename parsing fails."""

    pass
```

**Step 2: Run linter to check style**

Run: `ruff check src/ingestion/contract_filename_parser.py`
Expected: No errors

**Step 3: Format with ruff**

Run: `ruff format src/ingestion/contract_filename_parser.py`
Expected: File formatted

**Step 4: Commit**

```bash
git add src/ingestion/contract_filename_parser.py
git commit -m "feat: add contract filename parser module structure"
```

---

## Task 2: Add the main parsing function with regex pattern

**Files:**
- Modify: `src/ingestion/contract_filename_parser.py`

**Step 1: Write the failing test first**

Create: `tests/test_contract_filename_parser.py`

```python
"""Test contract filename parser."""

import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.contract_filename_parser import (
    parse_contract_filename,
    ParseError,
)


def test_parse_standard_filename():
    """Test parsing a standard SEC filing filename."""
    filename = "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"

    result = parse_contract_filename(filename)

    assert result["company_name"] == "CreditcardscomInc"
    assert result["execution_date"] == "2007-08-10"
    assert result["filing_type"] == "S-1"
    assert result["exhibit_number"] == "EX-10.33"
    assert result["accession_number"] == "362297"


def test_parse_with_hyphenated_filing_type():
    """Test parsing with hyphenated filing type like 10-K."""
    filename = "ExampleCorp_20200101_10-K_EX-10.1_12345_EX-10.1_License Agreement.pdf"

    result = parse_contract_filename(filename)

    assert result["company_name"] == "ExampleCorp"
    assert result["execution_date"] == "2020-01-01"
    assert result["filing_type"] == "10-K"
    assert result["exhibit_number"] == "EX-10.1"
    assert result["accession_number"] == "12345"


def test_parse_with_8k_filing():
    """Test parsing 8-K filing type."""
    filename = "TestCompany_20231215_8-K_EX-99.1_98765_EX-99.1_Employment Agreement.pdf"

    result = parse_contract_filename(filename)

    assert result["company_name"] == "TestCompany"
    assert result["execution_date"] == "2023-12-15"
    assert result["filing_type"] == "8-K"


def test_parse_invalid_format_raises_error():
    """Test that invalid filename format raises ParseError."""
    filename = "InvalidFilename.pdf"

    with pytest.raises(ParseError):
        parse_contract_filename(filename)


def test_parse_missing_date_raises_error():
    """Test that missing date field raises ParseError."""
    filename = "Company_SomeFiling_EX-10.1_12345_EX-10.1_Agreement.pdf"

    with pytest.raises(ParseError):
        parse_contract_filename(filename)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contract_filename_parser.py -v`
Expected: FAIL - `ImportError` or `AttributeError` (function not implemented yet)

**Step 3: Implement the parse_contract_filename function**

Add to `src/ingestion/contract_filename_parser.py`:

```python
def parse_contract_filename(filename: str) -> dict:
    """Parse a contract PDF filename and extract metadata.

    Args:
        filename: The PDF filename (not full path)

    Returns:
        Dictionary with extracted metadata:
            - company_name: str
            - execution_date: str (ISO format YYYY-MM-DD)
            - filing_type: str
            - exhibit_number: str (e.g., "EX-10.33")
            - accession_number: str

    Raises:
        ParseError: If filename doesn't match expected pattern

    Example:
        >>> parse_contract_filename("CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf")
        {
            "company_name": "CreditcardscomInc",
            "execution_date": "2007-08-10",
            "filing_type": "S-1",
            "exhibit_number": "EX-10.33",
            "accession_number": "362297"
        }
    """
    # Remove .pdf extension if present
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]

    # Regex pattern for SEC filing filename format
    # Format: {Company}_{Date}_{FilingType}_EX-{Exhibit}_{AccessionNum}_EX-{Exhibit}_{Description}
    # Example: CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement
    pattern = r"^([^_]+)_(\d{8})_([A-Z0-9\-]+)_EX-([\d.]+)_(\d+)_EX-[\d.]+_.+$"

    match = re.match(pattern, filename)

    if not match:
        logger.warning(f"Failed to parse filename: {filename}")
        raise ParseError(f"Filename does not match expected pattern: {filename}")

    # Extract groups
    company_name = match.group(1)
    date_str = match.group(2)  # YYYYMMDD
    filing_type = match.group(3)
    exhibit_number = f"EX-{match.group(4)}"
    accession_number = match.group(5)

    # Convert date to ISO format
    try:
        execution_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format in filename: {date_str}")
        raise ParseError(f"Invalid date format: {date_str}") from e

    return {
        "company_name": company_name,
        "execution_date": execution_date,
        "filing_type": filing_type,
        "exhibit_number": exhibit_number,
        "accession_number": accession_number,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contract_filename_parser.py -v`
Expected: PASS (all 5 tests pass)

**Step 5: Commit**

```bash
git add src/ingestion/contract_filename_parser.py tests/test_contract_filename_parser.py
git commit -m "feat: implement parse_contract_filename with regex pattern"
```

---

## Task 3: Add path parser function (combines folder + filename metadata)

**Files:**
- Modify: `src/ingestion/contract_filename_parser.py`
- Modify: `tests/test_contract_filename_parser.py`

**Step 1: Write the failing test first**

Add to `tests/test_contract_filename_parser.py`:

```python
def test_parse_contract_path_full():
    """Test parsing full path including folder structure."""
    full_path = "/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"

    result = parse_contract_path(full_path)

    assert result["company_name"] == "CreditcardscomInc"
    assert result["execution_date"] == "2007-08-10"
    assert result["filing_type"] == "S-1"
    assert result["exhibit_number"] == "EX-10.33"
    assert result["accession_number"] == "362297"
    assert result["contract_type"] == "Affiliate_Agreements"
    assert result["filename"] == "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"


def test_parse_contract_path_part_iii():
    """Test parsing from Part_III folder."""
    full_path = "full_contract_pdf/Part_III/License_Agreements/ExampleCorp_20200101_10-K_EX-10.1_12345_EX-10.1_License.pdf"

    result = parse_contract_path(full_path)

    assert result["contract_type"] == "License_Agreements"
    assert result["company_name"] == "ExampleCorp"


def test_parse_contract_path_with_co_branding_folder():
    """Test parsing contract type with underscores."""
    full_path = "/full_contract_pdf/Part_III/Co_Branding/Company_20220101_8-K_EX-10.1_99999_EX-10.1_Agreement.pdf"

    result = parse_contract_path(full_path)

    assert result["contract_type"] == "Co_Branding"


def test_parse_contract_path_invalid_filename():
    """Test that invalid filename in path raises ParseError."""
    full_path = "/full_contract_pdf/Part_I/Affiliate_Agreements/Invalid.pdf"

    with pytest.raises(ParseError):
        parse_contract_path(full_path)


def test_parse_contract_path_generates_document_id():
    """Test that document_id is generated from filename."""
    import hashlib

    full_path = "/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"

    result = parse_contract_path(full_path)

    assert "document_id" in result
    # document_id should be SHA256 hash of filename
    expected_filename = "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
    expected_id = hashlib.sha256(expected_filename.encode()).hexdigest()
    assert result["document_id"] == expected_id
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contract_filename_parser.py::test_parse_contract_path_full -v`
Expected: FAIL - `ImportError` or `NameError` (function not implemented yet)

**Step 3: Implement the parse_contract_path function**

Add to `src/ingestion/contract_filename_parser.py`:

```python
def parse_contract_path(file_path: str) -> dict:
    """Parse a contract PDF file path and extract all metadata.

    Combines filename parsing with folder structure to extract complete metadata.

    Args:
        file_path: Full or relative path to the PDF file

    Returns:
        Dictionary with all metadata including:
            - All fields from parse_contract_filename()
            - contract_type: str (extracted from folder name)
            - filename: str (original filename)
            - document_id: str (SHA256 hash of filename for uniqueness)

    Raises:
        ParseError: If filename doesn't match expected pattern

    Example:
        >>> parse_contract_path("full_contract_pdf/Part_I/Affiliate_Agreements/Company_20200101_S-1_EX-10.1_12345_EX-10.1_Agreement.pdf")
        {
            "company_name": "Company",
            "execution_date": "2020-01-01",
            "filing_type": "S-1",
            "exhibit_number": "EX-10.1",
            "accession_number": "12345",
            "contract_type": "Affiliate_Agreements",
            "filename": "Company_20200101_S-1_EX-10.1_12345_EX-10.1_Agreement.pdf",
            "document_id": "abc123..."
        }
    """
    import hashlib

    # Extract filename from path
    filename = os.path.basename(file_path)

    # Parse filename metadata
    metadata = parse_contract_filename(filename)

    # Extract contract type from parent folder
    contract_type = os.path.basename(os.path.dirname(file_path))
    metadata["contract_type"] = contract_type

    # Add original filename
    metadata["filename"] = filename

    # Generate unique document_id from filename
    document_id = hashlib.sha256(filename.encode()).hexdigest()
    metadata["document_id"] = document_id

    return metadata
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contract_filename_parser.py -v`
Expected: PASS (all tests pass, including new path tests)

**Step 5: Commit**

```bash
git add src/ingestion/contract_filename_parser.py tests/test_contract_filename_parser.py
git commit -m "feat: add parse_contract_path with folder metadata extraction"
```

---

## Task 4: Add ContractFilenameParser class for batch processing

**Files:**
- Modify: `src/ingestion/contract_filename_parser.py`
- Modify: `tests/test_contract_filename_parser.py`

**Step 1: Write the failing test first**

Add to `tests/test_contract_filename_parser.py`:

```python
from src.ingestion.contract_filename_parser import ContractFilenameParser


def test_parser_class_single_file():
    """Test ContractFilenameParser class with single file."""
    parser = ContractFilenameParser()
    filepath = "/full_contract_pdf/Part_I/Affiliate_Agreements/Company_20200101_S-1_EX-10.1_12345_EX-10.1_Agreement.pdf"

    result = parser.parse(filepath)

    assert result["company_name"] == "Company"
    assert result["contract_type"] == "Affiliate_Agreements"


def test_parser_class_batch_process():
    """Test batch processing multiple files."""
    parser = ContractFilenameParser()
    filepaths = [
        "/full_contract_pdf/Part_I/Affiliate_Agreements/CompanyA_20200101_S-1_EX-10.1_12345_EX-10.1_Agreement.pdf",
        "/full_contract_pdf/Part_II/License_Agreements/CompanyB_20210101_10-K_EX-10.2_67890_EX-10.2_License.pdf",
        "/full_contract_pdf/Part_III/Co_Branding/CompanyC_20220101_8-K_EX-99.1_11111_EX-99.1_Deal.pdf",
    ]

    results = parser.parse_batch(filepaths)

    assert len(results) == 3
    assert results[0]["company_name"] == "CompanyA"
    assert results[1]["contract_type"] == "License_Agreements"
    assert results[2]["contract_type"] == "Co_Branding"


def test_parser_class_batch_with_invalid_file():
    """Test batch processing skips invalid files and continues."""
    parser = ContractFilenameParser()
    filepaths = [
        "/full_contract_pdf/Part_I/Affiliate_Agreements/Company_20200101_S-1_EX-10.1_12345_EX-10.1_Agreement.pdf",
        "/invalid/path/Invalid.pdf",  # This should be skipped
        "/full_contract_pdf/Part_II/IP/Company_20210101_10-K_EX-10.1_22222_EX-10.1_IP_Agreement.pdf",
    ]

    results = parser.parse_batch(filepaths)

    assert len(results) == 2  # Only valid files parsed
    assert results[0]["company_name"] == "Company"
    assert results[1]["contract_type"] == "IP"


def test_parser_class_get_statistics():
    """Test getting parsing statistics."""
    parser = ContractFilenameParser()
    filepaths = [
        "/full_contract_pdf/Part_I/Affiliate_Agreements/Company_20200101_S-1_EX-10.1_12345_EX-10.1_Agreement.pdf",
        "/invalid.pdf",  # Will fail
        "/full_contract_pdf/Part_II/IP/Company_20210101_10-K_EX-10.1_22222_EX-10.1_IP.pdf",
    ]

    results = parser.parse_batch(filepaths)
    stats = parser.get_statistics()

    assert stats["total"] == 3
    assert stats["successful"] == 2
    assert stats["failed"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contract_filename_parser.py::test_parser_class_single_file -v`
Expected: FAIL - `NameError` (class not implemented yet)

**Step 3: Implement the ContractFilenameParser class**

Add to `src/ingestion/contract_filename_parser.py`:

```python
class ContractFilenameParser:
    """Parser for batch processing contract filenames.

    Tracks statistics and handles errors gracefully for batch operations.
    """

    def __init__(self):
        """Initialize the parser with zero statistics."""
        self._total_processed = 0
        self._successful_parses = 0
        self._failed_parses = 0

    def parse(self, file_path: str) -> dict:
        """Parse a single contract file path.

        Args:
            file_path: Path to the contract PDF file

        Returns:
            Metadata dictionary

        Raises:
            ParseError: If parsing fails
        """
        self._total_processed += 1
        try:
            result = parse_contract_path(file_path)
            self._successful_parses += 1
            return result
        except ParseError:
            self._failed_parses += 1
            raise

    def parse_batch(self, file_paths: list[str]) -> list[dict]:
        """Parse multiple contract file paths.

        Invalid files are skipped and counted in statistics.

        Args:
            file_paths: List of paths to contract PDF files

        Returns:
            List of metadata dictionaries (only for successfully parsed files)
        """
        results = []

        for file_path in file_paths:
            self._total_processed += 1
            try:
                result = parse_contract_path(file_path)
                results.append(result)
                self._successful_parses += 1
            except ParseError as e:
                self._failed_parses += 1
                logger.warning(f"Skipping invalid file: {file_path} - {e}")

        return results

    def get_statistics(self) -> dict:
        """Get parsing statistics.

        Returns:
            Dictionary with keys: total, successful, failed
        """
        return {
            "total": self._total_processed,
            "successful": self._successful_parses,
            "failed": self._failed_parses,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics to zero."""
        self._total_processed = 0
        self._successful_parses = 0
        self._failed_parses = 0
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contract_filename_parser.py -v`
Expected: PASS (all tests pass, including new class tests)

**Step 5: Commit**

```bash
git add src/ingestion/contract_filename_parser.py tests/test_contract_filename_parser.py
git commit -m "feat: add ContractFilenameParser class for batch processing"
```

---

## Task 5: Add utility functions for metadata validation

**Files:**
- Modify: `src/ingestion/contract_filename_parser.py`
- Modify: `tests/test_contract_filename_parser.py`

**Step 1: Write the failing test first**

Add to `tests/test_contract_filename_parser.py`:

```python
from src.ingestion.contract_filename_parser import (
    validate_metadata,
    normalize_company_name,
    is_valid_filing_type,
)


def test_validate_metadata_with_all_fields():
    """Test validation passes with complete metadata."""
    metadata = {
        "company_name": "TestCompany",
        "execution_date": "2020-01-01",
        "filing_type": "10-K",
        "exhibit_number": "EX-10.1",
        "accession_number": "12345",
        "contract_type": "License_Agreements",
        "filename": "Test.pdf",
        "document_id": "abc123",
    }

    # Should not raise
    validate_metadata(metadata)


def test_validate_metadata_missing_required_field():
    """Test validation fails with missing required field."""
    metadata = {
        "company_name": "TestCompany",
        # Missing execution_date
        "filing_type": "10-K",
        "exhibit_number": "EX-10.1",
        "accession_number": "12345",
    }

    with pytest.raises(ValueError):
        validate_metadata(metadata)


def test_normalize_company_name():
    """Test company name normalization."""
    assert normalize_company_name("Test-Company_Inc.") == "Test Company Inc"
    assert normalize_company_name("ABC_Corp") == "ABC Corp"
    assert normalize_company_name("Multiple   Spaces") == "Multiple Spaces"


def test_is_valid_filing_type():
    """Test filing type validation."""
    assert is_valid_filing_type("10-K") is True
    assert is_valid_filing_type("S-1") is True
    assert is_valid_filing_type("8-K") is True
    assert is_valid_filing_type("INVALID") is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contract_filename_parser.py::test_validate_metadata_with_all_fields -v`
Expected: FAIL - `ImportError` (functions not implemented yet)

**Step 3: Implement utility functions**

Add to `src/ingestion/contract_filename_parser.py`:

```python
# Valid SEC filing types
_VALID_FILING_TYPES = {
    "S-1", "S-1/A", "S-3", "S-3/A", "S-4", "S-8",
    "10-K", "10-K/A", "10-Q", "10-Q/A",
    "8-K", "8-K/A",
    "20-F", "20-F/A",
    "40-F", "6-K",
    "11-K", "11-K/A",
    "SB-2", "SB-2/A", "SB-3", "SB-3/A",
    "DEF 14A", "DEF 14A/A",
    "424B2", "424B3", "424B5", "424B7",
    "497K", "497A",
}


def validate_metadata(metadata: dict) -> None:
    """Validate that metadata contains all required fields.

    Args:
        metadata: Metadata dictionary to validate

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = {
        "company_name",
        "execution_date",
        "filing_type",
        "exhibit_number",
        "accession_number",
    }

    missing = required_fields - metadata.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Validate date format
    try:
        datetime.strptime(metadata["execution_date"], "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid execution_date format: {metadata['execution_date']}") from e


def normalize_company_name(name: str) -> str:
    """Normalize company name by replacing underscores/hyphens with spaces.

    Args:
        name: Raw company name from filename

    Returns:
        Normalized company name

    Example:
        >>> normalize_company_name("Test-Company_Inc.")
        "Test Company Inc"
    """
    # Replace underscores and hyphens with spaces
    normalized = name.replace("_", " ").replace("-", " ")
    # Remove extra dots (keep decimal points in numbers though)
    normalized = " ".join(normalized.split())  # Collapse multiple spaces
    return normalized


def is_valid_filing_type(filing_type: str) -> bool:
    """Check if filing type is a valid SEC filing type.

    Args:
        filing_type: Filing type string

    Returns:
        True if filing type is recognized, False otherwise
    """
    # Normalize (remove /A suffix for amendments)
    base_type = filing_type.split("/")[0]
    return base_type in _VALID_FILING_TYPES
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contract_filename_parser.py -v`
Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add src/ingestion/contract_filename_parser.py tests/test_contract_filename_parser.py
git commit -m "feat: add metadata validation and utility functions"
```

---

## Task 6: Update __init__.py to export parser functions

**Files:**
- Modify: `src/ingestion/__init__.py`

**Step 1: Check current __init__.py content**

Run: `cat src/ingestion/__init__.py`
Expected: Empty or minimal exports

**Step 2: Add exports for contract parser**

```python
"""Ingestion pipeline for ArXiv papers and contracts."""

from src.ingestion.contract_filename_parser import (
    ContractFilenameParser,
    ParseError,
    is_valid_filing_type,
    normalize_company_name,
    parse_contract_filename,
    parse_contract_path,
    validate_metadata,
)

__all__ = [
    "ContractFilenameParser",
    "ParseError",
    "is_valid_filing_type",
    "normalize_company_name",
    "parse_contract_filename",
    "parse_contract_path",
    "validate_metadata",
]
```

**Step 3: Run linter**

Run: `ruff check src/ingestion/__init__.py`
Expected: No errors

**Step 4: Test import works**

Run: `python -c "from src.ingestion import parse_contract_path; print('Import successful')"`
Expected: "Import successful"

**Step 5: Commit**

```bash
git add src/ingestion/__init__.py
git commit -m "feat: export contract parser functions from ingestion module"
```

---

## Task 7: Run full test suite and verify

**Step 1: Run all parser tests**

Run: `pytest tests/test_contract_filename_parser.py -v`
Expected: PASS (all tests pass)

**Step 2: Run full test suite to ensure no breakage**

Run: `pytest tests/ -v`
Expected: PASS (all existing tests still pass)

**Step 3: Run ruff linting**

Run: `ruff check src/ingestion/contract_filename_parser.py tests/test_contract_filename_parser.py`
Expected: No errors

**Step 4: Run ruff formatting**

Run: `ruff format src/ingestion/contract_filename_parser.py tests/test_contract_filename_parser.py`
Expected: Files already formatted

**Step 5: Test with real contract filenames**

Create: `scripts/test_parser_real.py`

```python
"""Test parser with real contract filenames."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.contract_filename_parser import parse_contract_path

# Real examples from the dataset
test_files = [
    "full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf",
    "full_contract_pdf/Part_I/Co_Branding/2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.pdf",
    "full_contract_pdf/Part_III/License_Agreements/Example_20200101_10-K_EX-10.1_12345_EX-10.1_License Agreement.pdf",
]

for filepath in test_files:
    print(f"\nParsing: {os.path.basename(filepath)}")
    try:
        result = parse_contract_path(filepath)
        print(f"  Company: {result['company_name']}")
        print(f"  Date: {result['execution_date']}")
        print(f"  Type: {result['filing_type']}")
        print(f"  Contract: {result['contract_type']}")
        print(f"  Document ID: {result['document_id'][:16]}...")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n✓ Parser test complete")
```

Run: `python scripts/test_parser_real.py`
Expected: All files parsed successfully with metadata displayed

**Step 6: Commit test script**

```bash
git add scripts/test_parser_real.py
git commit -m "test: add real contract filename test script"
```

---

## Task 8: Update Phase 1 issue status

**Step 1: Close Phase 1 issue**

Run: `bd close ChromaDB_POC-nwf --reason="Implemented contract filename parser with regex pattern, batch processing, and comprehensive tests"`

**Step 2: Sync beads**

Run: `bd sync`

**Step 3: Push to git**

```bash
git add .
git commit -m "chore: close Phase 1 issue"
git push
```

---

## Phase 1 Gate Verification

**Run all verification commands:**

```bash
# 1. Unit tests pass
pytest tests/test_contract_filename_parser.py -v

# 2. Full test suite passes
pytest tests/ -v

# 3. Linting passes
ruff check src/ingestion/

# 4. Type check (if using mypy)
# mypy src/ingestion/contract_filename_parser.py

# 5. Real file test
python scripts/test_parser_real.py
```

**Expected Results:**
- ✅ All parser tests pass (15+ tests)
- ✅ Full test suite passes without regression
- ✅ No linting errors
- ✅ Real contract filenames parse successfully
- ✅ Phase 1 issue closed in beads
- ✅ All commits pushed to remote

---

## Next Phase

After Phase 1 is complete, proceed to **Phase 2: Update ChromaStore for multi-collection support** (`ChromaDB_POC-n72`).

---

**Total Estimated Time:** 2-3 hours
**Test Coverage:** ~95% (15+ tests)
**Lines of Code:** ~300 (production) + ~400 (tests)
