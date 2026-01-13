"""ChromaDB Cloud connection and store management."""

import logging

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manage ChromaDB Cloud connection and collection operations."""

    def __init__(self, host: str, api_key: str, collection_name: str):
        """Initialize ChromaDB client.

        Args:
            host: ChromaDB Cloud host URL
            api_key: ChromaDB API key for authentication
            collection_name: Name of the collection to use

        Raises:
            ImportError: If chromadb is not installed
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Install it with: pip install chromadb"
            )

        self.host = host
        self.api_key = api_key
        self.collection_name = collection_name

        # Initialize ChromaDB client with authentication
        self.client = chromadb.HttpClient(
            host=host,
            headers={"X-Chroma-Token": api_key},
        )

        logger.info(f"Initialized ChromaDB client for {host}")

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
