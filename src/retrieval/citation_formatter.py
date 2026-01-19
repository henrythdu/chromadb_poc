"""Format citations for different collection types."""

import logging

logger = logging.getLogger(__name__)


class CitationFormatter:
    """Format citations for different collection types.

    Detects collection type from metadata fields and formats appropriately:
    - ArXiv papers: arxiv_id field present
    - SEC contracts: document_id field present
    """

    def format_citation(self, metadata: dict) -> str:
        """Format citation based on collection type detection.

        Args:
            metadata: Chunk metadata dictionary

        Returns:
            Formatted citation string (markdown format for papers, plain text for contracts)
        """
        if "arxiv_id" in metadata:
            return self._format_arxiv_paper(metadata)
        elif "document_id" in metadata:
            return self._format_contract(metadata)
        else:
            logger.warning(f"Unknown metadata type: {list(metadata.keys())}")
            return f"Unknown source: {metadata}"

    def _format_arxiv_paper(self, metadata: dict) -> str:
        """Format ArXiv paper citation with clickable link.

        Args:
            metadata: Must contain arxiv_id, may contain title and authors

        Returns:
            Markdown link format: [Title - Authors (arxiv:id)](url)
        """
        title = metadata.get("title", "Unknown Title")
        authors = metadata.get("authors", "Unknown Authors")
        arxiv_id = metadata.get("arxiv_id", "")

        citation_text = f"{title} - {authors} (arxiv:{arxiv_id})"
        url = f"https://arxiv.org/abs/{arxiv_id}"

        return f"[{citation_text}]({url})"

    def _format_contract(self, metadata: dict) -> str:
        """Format SEC contract citation with descriptive narrative.

        Args:
            metadata: Must contain document_id, may contain company_name,
                     contract_type, execution_date, filing_type, exhibit_number

        Returns:
            Plain text format: Company [Contract Type] executed on Date (Filing: X Exhibit: Y)
        """
        company = metadata.get("company_name", "Unknown Company")
        contract_type = metadata.get("contract_type", "Unknown Agreement")
        date = metadata.get("execution_date", "Unknown Date")
        filing = metadata.get("filing_type", "N/A")
        exhibit = metadata.get("exhibit_number", "N/A")

        return f"{company} [{contract_type}] executed on {date} (Filing: {filing} Exhibit: {exhibit})"
