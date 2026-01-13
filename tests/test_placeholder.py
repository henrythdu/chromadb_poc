"""Placeholder tests to verify pytest works."""


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
