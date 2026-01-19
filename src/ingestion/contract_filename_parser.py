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
import re
from datetime import datetime

logger = logging.getLogger(__name__)


__all__ = [
    "parse_contract_filename",
    "ParseError",
]


class ParseError(Exception):
    """Raised when filename parsing fails."""

    pass


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
