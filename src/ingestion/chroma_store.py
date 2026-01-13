"""ChromaDB Cloud connection and store management."""

import logging
import os

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manage ChromaDB Cloud connection and collection operations."""

    def __init__(
        self,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        collection_name: str = "papers",
    ):
        """Initialize ChromaDB Cloud client.

        Args:
            api_key: ChromaDB Cloud API key (defaults to CHROMA_CLOUD_API_KEY env var)
            tenant: ChromaDB tenant ID (defaults to CHROMA_TENANT env var)
            database: ChromaDB database name (defaults to CHROMA_DATABASE env var)
            collection_name: Name of the collection to use

        Raises:
            ImportError: If chromadb is not installed
            ValueError: If required credentials are not provided
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Install it with: pip install chromadb"
            )

        # Load credentials from parameters or environment variables
        self.api_key = api_key or os.getenv("CHROMA_CLOUD_API_KEY")
        self.tenant = tenant or os.getenv("CHROMA_TENANT")
        self.database = database or os.getenv("CHROMA_DATABASE")
        self.collection_name = collection_name

        # Validate required credentials
        if not all([self.api_key, self.tenant, self.database]):
            raise ValueError(
                "ChromaDB credentials not found. "
                "Set CHROMA_CLOUD_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE "
                "environment variables, or pass them as parameters."
            )

        # Initialize ChromaDB Cloud client
        self.client = chromadb.CloudClient(
            api_key=self.api_key,
            tenant=self.tenant,
            database=self.database,
        )

        logger.info(
            f"Initialized ChromaDB Cloud client for tenant={self.tenant}, "
            f"database={self.database}"
        )

    def _get_or_create_collection(self):
        """Get existing collection or create new one.

        Returns:
            ChromaDB collection object
        """
        collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(f"Accessed collection: {self.collection_name}")
        return collection

    def add_documents(
        self, documents: list[str], metadatas: list[dict], ids: list[str]
    ):
        """Add documents to the collection.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: List of unique identifiers for each document
        """
        collection = self._get_or_create_collection()
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(documents)} documents to collection")

    def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Number of documents in the collection
        """
        collection = self._get_or_create_collection()
        count = collection.count()
        logger.info(f"Collection {self.collection_name} has {count} documents")
        return count

    def test_connection(self) -> bool:
        """Test the connection to ChromaDB Cloud.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.heartbeat()
            logger.info("Successfully connected to ChromaDB Cloud")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB Cloud: {e}")
            return False
