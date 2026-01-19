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

logger = logging.getLogger(__name__)


__all__ = [
    "parse_contract_filename",
    "parse_contract_path",
    "ParseError",
    "ContractFilenameParser",
    "validate_metadata",
    "normalize_company_name",
    "is_valid_filing_type",
]


class ParseError(Exception):
    """Raised when filename parsing fails."""

    pass


# Valid SEC filing types
_VALID_FILING_TYPES = {
    "S-1",
    "S-1/A",
    "S-3",
    "S-3/A",
    "S-4",
    "S-8",
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "8-K",
    "8-K/A",
    "20-F",
    "20-F/A",
    "40-F",
    "6-K",
    "11-K",
    "11-K/A",
    "SB-2",
    "SB-2/A",
    "SB-3",
    "SB-3/A",
    "DEF 14A",
    "DEF 14A/A",
    "424B2",
    "424B3",
    "424B5",
    "424B7",
    "497K",
    "497A",
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
        raise ValueError(
            f"Invalid execution_date format: {metadata['execution_date']}"
        ) from e


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
    # Remove trailing dots (like "Inc." -> "Inc")
    normalized = normalized.rstrip(".")
    # Collapse multiple spaces
    normalized = " ".join(normalized.split())
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
