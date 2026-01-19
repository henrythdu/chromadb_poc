"""Test contract filename parser."""

import os
import sys

import pytest

from src.ingestion.contract_filename_parser import ParseError, parse_contract_filename

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
