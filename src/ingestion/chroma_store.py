"""ChromaDB Cloud connection and store management."""

import logging

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manage ChromaDB Cloud connection and collection operations."""

    def __init__(
        self,
        api_key: str,
        tenant: str,
        database: str,
        collection_name: str = "papers",
    ):
        """Initialize ChromaDB Cloud client.

        Args:
            api_key: ChromaDB Cloud API key (required)
            tenant: ChromaDB tenant ID (required)
            database: ChromaDB database name (required)
            collection_name: Name of the collection to use

        Raises:
            ImportError: If chromadb is not installed
            ValueError: If required credentials are not provided
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Install it with: pip install chromadb"
            )

        # Validate required credentials
        if not all([api_key, tenant, database]):
            raise ValueError(
                "ChromaDB credentials must be provided as parameters."
            )

        self.collection_name = collection_name

        # Initialize ChromaDB Cloud client (credentials not stored after use)
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )

        logger.info(
            f"Initialized ChromaDB Cloud client for tenant={tenant}, "
            f"database={database}"
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
