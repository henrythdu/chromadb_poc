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
