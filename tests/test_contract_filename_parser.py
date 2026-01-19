"""Test contract filename parser."""

import os
import sys

import pytest

from src.ingestion.contract_filename_parser import (
    ContractFilenameParser,
    ParseError,
    is_valid_filing_type,
    normalize_company_name,
    parse_contract_filename,
    parse_contract_path,
    validate_metadata,
)

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
    assert (
        result["filename"]
        == "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
    )


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
