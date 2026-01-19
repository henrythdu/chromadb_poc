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
