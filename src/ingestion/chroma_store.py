"""ChromaDB Cloud connection and store management."""

import logging
from typing import Any

try:
    import chromadb
except ImportError:
    chromadb = None

from src.config import settings

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
            raise ValueError("ChromaDB credentials must be provided as parameters.")

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

    def get_collection(self, name: str | None = None) -> Any:
        """Get a collection by name, or default collection if no name provided.

        This method enables dynamic collection access for multi-collection scenarios
        while maintaining backward compatibility with existing single-collection usage.

        Args:
            name: Collection name to retrieve. If None, returns the default collection
                    specified during initialization.

        Returns:
            ChromaDB collection object

        Raises:
            ValueError: If name is an empty string

        Example:
            >>> store = ChromaStore(api_key="...", tenant="...", database="...", collection_name="papers")
            >>> papers_coll = store.get_collection()  # Gets "papers" (default)
            >>> contracts_coll = store.get_collection("contracts")  # Gets "contracts" (dynamic)
        """
        if name is None:
            # Use default collection from instance
            return self._get_or_create_collection()

        if not name or not isinstance(name, str):
            raise ValueError("Collection name must be a non-empty string")

        collection = self.client.get_or_create_collection(name=name)
        logger.info(f"Accessed collection: {name}")
        return collection

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        collection_name: str | None = None,
    ):
        """Add or update documents in the collection using upsert.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: List of unique identifiers for each document
            collection_name: Optional collection name. If None, uses the default
                           collection specified during initialization.

        Note:
            Uses upsert instead of add to enable resumable ingestion.
            Existing documents with the same ID will be updated,
            new documents will be added. No duplicate errors.

            When collection_name is specified, documents are added to that collection
            instead of the default. This enables multi-collection scenarios.
        """
        if collection_name is not None:
            collection = self.get_collection(collection_name)
        else:
            collection = self._get_or_create_collection()

        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        coll_name = collection_name if collection_name else self.collection_name
        logger.info(f"Upserted {len(documents)} documents to collection '{coll_name}'")

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

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if any chunks exist for the given arxiv_id.

        Args:
            arxiv_id: The arxiv ID to check (e.g., "2601.03764v1")

        Returns:
            True if any chunks with matching arxiv_id are found
        """
        collection = self._get_or_create_collection()

        # Query for documents with this arxiv_id in metadata
        results = collection.get(
            where={"arxiv_id": arxiv_id},
            limit=1,  # Only need to know if at least one exists
        )

        exists = len(results.get("ids", [])) > 0
        if exists:
            logger.debug(f"Paper {arxiv_id} already exists in collection")
        return exists

    def get_existing_paper_ids(self, arxiv_ids: list[str]) -> set[str]:
        """Given a list of arxiv_ids, return a set of those that already exist.

        Args:
            arxiv_ids: List of arxiv IDs to check

        Returns:
            Set of arxiv_ids that exist in the collection
        """
        if not arxiv_ids:
            return set()

        collection = self._get_or_create_collection()

        # Use the $in operator for batch metadata query
        results = collection.get(
            where={"arxiv_id": {"$in": arxiv_ids}},
            include=["metadatas"],  # Only fetch metadata for efficiency
        )

        # Extract unique arxiv_ids from the metadata of found chunks
        return {
            meta["arxiv_id"]
            for meta in results.get("metadatas", [])
            if "arxiv_id" in meta
        }

    def update_paper_metadata(self, arxiv_id: str, new_metadata: dict) -> int:
        """Update metadata for all chunks belonging to a paper.

        Args:
            arxiv_id: Paper ID whose chunks should be updated
            new_metadata: New metadata dict (will be merged with existing)

        Returns:
            Number of chunks updated
        """
        collection = self._get_or_create_collection()

        # Get all chunk IDs for this paper
        results = collection.get(
            where={"arxiv_id": arxiv_id}, include=["metadatas", "documents"]
        )

        if not results["ids"]:
            logger.warning(f"No chunks found for {arxiv_id}")
            return 0

        # Update each chunk with new metadata
        ids = results["ids"]
        metadatas = []

        for old_meta in results["metadatas"]:
            # Merge old metadata with new metadata
            merged = {**old_meta, **new_metadata}
            metadatas.append(merged)

        collection.update(ids=ids, metadatas=metadatas)
        logger.info(f"Updated {len(ids)} chunks for {arxiv_id}")
        return len(ids)

    def get_all_arxiv_ids(self) -> list[str]:
        """Get all unique arxiv_ids from the collection.

        Returns:
            Sorted list of unique arxiv_ids
        """
        collection = self._get_or_create_collection()
        arxiv_ids = set()

        # Use pagination to avoid quota limit (max 300 per request)
        offset = 0
        batch_size = settings.chroma_batch_size

        while True:
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )

            # Extract arxiv_ids from this batch
            batch_ids = results.get("ids", [])
            if not batch_ids:
                break

            for meta in results.get("metadatas", []):
                if "arxiv_id" in meta:
                    arxiv_ids.add(meta["arxiv_id"])

            logger.info(
                f"Fetched {len(batch_ids)} chunks (offset={offset}, total unique={len(arxiv_ids)})"
            )

            # If we got fewer than requested, we've reached the end
            if len(batch_ids) < batch_size:
                break

            offset += batch_size

        return sorted(list(arxiv_ids))

    def get_chunks_by_arxiv_id(
        self, arxiv_id: str, include: list[str] | None = None
    ) -> dict[str, Any]:
        """Retrieve all chunks for a specific document by arxiv_id.

        Args:
            arxiv_id: The arxiv ID to retrieve chunks for (e.g., "2601.03764v1")
            include: What to include in results. Options: ["documents", "metadatas", "embeddings"]
                     Default: ["documents", "metadatas"]

        Returns:
            Dict with keys:
                - ids: List of chunk IDs
                - documents: List of chunk texts (if included)
                - metadatas: List of metadata dicts (if included)
                - embeddings: List of embeddings (if included)
            Returns empty dict if no chunks found.

        Example:
            >>> chunks = store.get_chunks_by_arxiv_id("2601.03764v1")
            >>> for i, (text, meta) in enumerate(zip(chunks["documents"], chunks["metadatas"])):
            ...     print(f"Chunk {meta['chunk_index']}: {text[:50]}...")
        """
        if include is None:
            include = ["documents", "metadatas"]

        collection = self._get_or_create_collection()

        results = collection.get(
            where={"arxiv_id": arxiv_id},
            include=include,
        )

        if not results.get("ids"):
            logger.warning(f"No chunks found for arxiv_id: {arxiv_id}")
            return {}

        # Sort results by chunk_index if metadata is included
        if "metadatas" in include and results.get("metadatas"):
            # Create list of indices sorted by chunk_index
            sorted_indices = sorted(
                range(len(results["metadatas"])),
                key=lambda i: results["metadatas"][i].get("chunk_index", 0),
            )

            # Reorder all lists based on sorted indices
            for key in ["ids", "documents", "metadatas", "embeddings"]:
                if key in results and results[key] is not None:
                    results[key] = [results[key][i] for i in sorted_indices]

            logger.info(
                f"Retrieved {len(results['ids'])} chunks for {arxiv_id} "
                f"(sorted by chunk_index)"
            )
        else:
            logger.info(f"Retrieved {len(results['ids'])} chunks for {arxiv_id}")

        return results
