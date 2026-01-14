"""PDF parser module supporting Docling (local) and LlamaParse (cloud)."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DoclingWrapper:
    """Wrapper for Docling to convert PDFs to Markdown locally.

    Docling is an open-source PDF parsing library by IBM Research
    that provides high-quality text extraction without API costs.
    """

    def __init__(self, result_type: str = "markdown", enable_ocr: bool = False) -> None:
        """Initialize the Docling wrapper.

        Args:
            result_type: Output format (only "markdown" supported for now)
            enable_ocr: Whether to enable OCR for image-based PDFs (default: False for POC)
        """
        self.result_type = result_type
        self.enable_ocr = enable_ocr
        self._converter: Any = None
        logger.info(f"Initialized DoclingWrapper with result_type={result_type}, enable_ocr={enable_ocr}")

    def _get_converter(self) -> Any:
        """Lazy load the Docling converter.

        Returns:
            DocumentConverter instance

        Raises:
            ImportError: If docling is not installed
        """
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                # Configure pipeline options for maximum speed
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = self.enable_ocr
                pipeline_options.do_table_structure = False

                self._converter = DocumentConverter(
                    format_options={"application/pdf": pipeline_options}
                )
                logger.info(
                    f"Docling converter initialized "
                    f"(OCR: {'enabled' if self.enable_ocr else 'disabled'}, table structure: disabled)"
                )
            except ImportError as e:
                logger.error(f"Failed to import docling: {e}")
                raise
        return self._converter

    def parse_pdf(
        self, pdf_path: str | Path, max_retries: int = 2
    ) -> dict[str, Any]:
        """Parse a PDF file to Markdown format.

        Args:
            pdf_path: Path to the PDF file to parse
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            Dictionary with keys:
                - markdown: Parsed markdown content (empty if error)
                - pages: Number of pages parsed (0 if error)
                - error: Error message if parsing failed, None otherwise
        """
        import time

        pdf_path = Path(pdf_path)

        # Check if file exists
        if not pdf_path.exists():
            error_msg = f"File not found: {pdf_path}"
            logger.error(error_msg)
            return {"markdown": "", "pages": 0, "error": error_msg}

        # Parse with retry logic
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Parsing PDF {pdf_path} (attempt {attempt + 1}/{max_retries})"
                )

                # Get converter (lazy loading)
                converter = self._get_converter()

                # Convert PDF to markdown
                result = converter.convert(str(pdf_path))

                # Export to markdown
                markdown_content = result.document.export_to_markdown()

                # Get page count
                page_count = len(result.document.pages)

                logger.info(
                    f"Successfully parsed {pdf_path}: {page_count} pages, "
                    f"{len(markdown_content)} characters"
                )

                return {
                    "markdown": markdown_content.strip(),
                    "pages": page_count,
                    "error": None,
                }

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                # Wait before retry to give system time to recover
                if attempt < max_retries - 1:
                    time.sleep(1)

        # All retries exhausted
        error_msg = f"Failed to parse PDF after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        return {"markdown": "", "pages": 0, "error": error_msg}


class LlamaParserWrapper:
    """Wrapper for LlamaParse to convert PDFs to Markdown.

    This class provides a lazy-loading wrapper around LlamaParse with
    retry logic and error handling for robust PDF parsing.

    Note: LlamaParse is a cloud service with API costs.
    For local processing, use DoclingWrapper instead.
    """

    def __init__(self, api_key: str, result_type: str = "markdown") -> None:
        """Initialize the LlamaParse wrapper.

        Args:
            api_key: LlamaParse API key for authentication
            result_type: Output format ("markdown" or "text")
        """
        self.api_key = api_key
        self.result_type = result_type
        self._client: Any = None
        logger.info(f"Initialized LlamaParserWrapper with result_type={result_type}")

    def _get_client(self) -> Any:
        """Lazy load the LlamaParse client.

        Returns:
            LlamaParse client instance

        Raises:
            ImportError: If llama_parse is not installed
        """
        if self._client is None:
            try:
                from llama_parse import LlamaParse

                self._client = LlamaParse(
                    api_key=self.api_key,
                    result_type=self.result_type,
                    verbose=True,
                )
                logger.info("LlamaParse client initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import llama_parse: {e}")
                raise
        return self._client

    def parse_pdf(
        self, pdf_path: str | Path, max_retries: int = 3
    ) -> dict[str, Any]:
        """Parse a PDF file to Markdown format with retry logic.

        Args:
            pdf_path: Path to the PDF file to parse
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Dictionary with keys:
                - markdown: Parsed markdown content (empty if error)
                - pages: Number of pages parsed (0 if error)
                - error: Error message if parsing failed, None otherwise
        """
        import time

        pdf_path = Path(pdf_path)

        # Check if file exists
        if not pdf_path.exists():
            error_msg = f"File not found: {pdf_path}"
            logger.error(error_msg)
            return {"markdown": "", "pages": 0, "error": error_msg}

        # Get client (lazy loading)
        try:
            client = self._get_client()
        except ImportError as e:
            error_msg = f"Failed to initialize LlamaParse client: {e}"
            logger.error(error_msg)
            return {"markdown": "", "pages": 0, "error": error_msg}

        # Parse with exponential backoff retry
        last_error = None
        for attempt in range(max_retries):
            try:
                # Simple throttle to avoid hitting API limits too fast
                if attempt == 0:
                    time.sleep(0.5)  # 500ms delay before each API call

                logger.info(
                    f"Parsing PDF {pdf_path} (attempt {attempt + 1}/{max_retries})"
                )

                # Parse the PDF
                documents = client.load_data(str(pdf_path))

                # Extract markdown content
                markdown_content = ""
                page_count = 0

                if documents:
                    for doc in documents:
                        markdown_content += doc.text + "\n\n"
                        page_count += 1

                logger.info(
                    f"Successfully parsed {pdf_path}: {page_count} pages, "
                    f"{len(markdown_content)} characters"
                )

                return {
                    "markdown": markdown_content.strip(),
                    "pages": page_count,
                    "error": None,
                }

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}"
                )

                # Exponential backoff: wait before retry
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s, etc.
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        # All retries exhausted
        error_msg = f"Failed to parse PDF after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        return {"markdown": "", "pages": 0, "error": error_msg}


# For backward compatibility, default to Docling (free, local)
DefaultParser = DoclingWrapper
