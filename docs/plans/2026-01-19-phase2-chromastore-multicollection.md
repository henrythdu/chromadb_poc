# Phase 2: ChromaStore Multi-Collection Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update ChromaStore to support multiple collections (arxiv_papers and contracts) while maintaining backward compatibility.

**Architecture:** Add dynamic collection access via `get_collection(name: str)` method while preserving existing single-collection behavior through default parameter.

**Tech Stack:** Python 3.10+, pytest, unittest.mock, chromadb

---

## Task 1: Add get_collection method to ChromaStore

**Files:**
- Modify: `src/ingestion/chroma_store.py`
- Test: `tests/test_chroma_store.py`

**Step 1: Write the failing test**

Add to `tests/test_chroma_store.py`:

```python
def test_get_collection_dynamic():
    """Test getting a collection by name dynamically."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock two different collections
        mock_arxiv_collection = MagicMock()
        mock_contracts_collection = MagicMock()

        def mock_get_or_create(name):
            if name == "arxiv_papers":
                return mock_arxiv_collection
            elif name == "contracts":
                return mock_contracts_collection
            return MagicMock()

        mock_instance.get_or_create_collection = mock_get_or_create

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="arxiv_papers",
        )

        # Get arxiv_papers collection (default)
        arxiv_coll = store.get_collection("arxiv_papers")
        assert arxiv_coll == mock_arxiv_collection

        # Get contracts collection (dynamic)
        contracts_coll = store.get_collection("contracts")
        assert contracts_coll == mock_contracts_collection


def test_get_collection_default():
    """Test that get_collection with no args returns default collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        mock_default_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_default_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="default_collection",
        )

        # Get default collection without argument
        default_coll = store.get_collection()
        assert default_coll == mock_default_collection
        assert default_coll == store._get_or_create_collection()  # Same as default
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chroma_store.py::test_get_collection_dynamic -v`
Expected: FAIL - `AttributeError: 'ChromaStore' object has no attribute 'get_collection'`

**Step 3: Implement the get_collection method**

Add to `src/ingestion/chroma_store.py` after the `_get_or_create_collection` method (around line 70):

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chroma_store.py::test_get_collection_dynamic -v`
Expected: PASS (both new tests pass)

**Step 5: Commit**

```bash
git add src/ingestion/chroma_store.py tests/test_chroma_store.py
git commit -m "feat: add get_collection method for dynamic collection access"
```

---

## Task 2: Update add_documents to support optional collection parameter

**Files:**
- Modify: `src/ingestion/chroma_store.py`
- Test: `tests/test_chroma_store.py`

**Step 1: Write the failing test**

Add to `tests/test_chroma_store.py`:

```python
def test_add_documents_to_specific_collection():
    """Test adding documents to a specific collection by name."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock collections
        mock_default_collection = MagicMock()
        mock_contracts_collection = MagicMock()

        collections_called = []

        def mock_get_or_create(name):
            collections_called.append(name)
            if name == "papers":
                return mock_default_collection
            elif name == "contracts":
                return mock_contracts_collection
            return MagicMock()

        mock_instance.get_or_create_collection = mock_get_or_create

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="papers",
        )

        # Add to default collection (no collection_name specified)
        store.add_documents(
            documents=["doc1"],
            metadatas=[{"key": "value1"}],
            ids=["id1"]
        )

        # Add to contracts collection (specific collection)
        store.add_documents(
            documents=["doc2"],
            metadatas=[{"key": "value2"}],
            ids=["id2"],
            collection_name="contracts"
        )

        # Verify both collections were accessed
        assert "papers" in collections_called
        assert "contracts" in collections_called

        # Verify upsert was called on both collections
        assert mock_default_collection.upsert.called
        assert mock_contracts_collection.upsert.called


def test_add_documents_backward_compatible():
    """Test that existing add_documents usage (without collection_name) still works."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="papers",
        )

        # Call without collection_name parameter (existing usage)
        store.add_documents(
            documents=["doc1"],
            metadatas=[{"key": "value"}],
            ids=["id1"]
        )

        # Verify upsert was called
        mock_collection.upsert.assert_called_once_with(
            documents=["doc1"],
            metadatas=[{"key": "value"}],
            ids=["id1"]
        )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chroma_store.py::test_add_documents_to_specific_collection -v`
Expected: FAIL - `TypeError: add_documents() got an unexpected keyword argument 'collection_name'`

**Step 3: Update the add_documents method**

Replace the existing `add_documents` method in `src/ingestion/chroma_store.py` with:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chroma_store.py::test_add_documents_to_specific_collection -v`
Expected: PASS (both new tests pass)

**Step 5: Verify backward compatibility - run existing tests**

Run: `pytest tests/test_chroma_store.py::test_add_documents -v`
Expected: PASS (existing test still works)

**Step 6: Commit**

```bash
git add src/ingestion/chroma_store.py tests/test_chroma_store.py
git commit -m "feat: add optional collection_name parameter to add_documents"
```

---

## Task 3: Update count method to support optional collection parameter

**Files:**
- Modify: `src/ingestion/chroma_store.py`
- Test: `tests/test_chroma_store.py`

**Step 1: Write the failing test**

Add to `tests/test_chroma_store.py`:

```python
def test_count_specific_collection():
    """Test counting documents in a specific collection."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")
        return

    from src.ingestion.chroma_store import ChromaStore

    if chromadb is None:
        pytest.skip("chromadb not installed - requires Python 3.12 or earlier")

    with patch("src.ingestion.chroma_store.chromadb.CloudClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Mock collections with different counts
        mock_papers_collection = MagicMock()
        mock_papers_collection.count.return_value = 100

        mock_contracts_collection = MagicMock()
        mock_contracts_collection.count.return_value = 50

        def mock_get_or_create(name):
            if name == "papers":
                return mock_papers_collection
            elif name == "contracts":
                return mock_contracts_collection
            return MagicMock()

        mock_instance.get_or_create_collection = mock_get_or_create

        store = ChromaStore(
            api_key="test_key",
            tenant="test-tenant",
            database="test_database",
            collection_name="papers",
        )

        # Count default collection
        papers_count = store.count()
        assert papers_count == 100

        # Count specific collection
        contracts_count = store.count(collection_name="contracts")
        assert contracts_count == 50

        # Verify count was called on correct collection
        mock_contracts_collection.count.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chroma_store.py::test_count_specific_collection -v`
Expected: FAIL - `TypeError: count() got an unexpected keyword argument 'collection_name'`

**Step 3: Update the count method**

Replace the existing `count` method in `src/ingestion/chroma_store.py` with:

```python
    def count(self, collection_name: str | None = None) -> int:
        """Get the number of documents in the collection.

        Args:
            collection_name: Optional collection name to count. If None, uses the default
                           collection specified during initialization.

        Returns:
            Number of documents in the collection
        """
        if collection_name is not None:
            collection = self.get_collection(collection_name)
        else:
            collection = self._get_or_create_collection()

        count = collection.count()

        coll_name = collection_name if collection_name else self.collection_name
        logger.info(f"Collection '{coll_name}' has {count} documents")
        return count
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chroma_store.py::test_count_specific_collection -v`
Expected: PASS

**Step 5: Verify backward compatibility**

Run: `pytest tests/test_chroma_store.py -k "test_count" -v`
Expected: All count tests pass

**Step 6: Commit**

```bash
git add src/ingestion/chroma_store.py tests/test_chroma_store.py
git commit -m "feat: add optional collection_name parameter to count method"
```

---

## Task 4: Run full ChromaStore test suite to verify backward compatibility

**Step 1: Run all ChromaStore tests**

Run: `pytest tests/test_chroma_store.py -v`

Expected: All tests pass (including existing tests)

**Step 2: Check for any test failures**

If any tests fail, investigate and fix:
- Check if tests rely on default collection behavior
- Ensure backward compatibility is maintained

**Step 3: Run ruff checks**

Run: `ruff check src/ingestion/chroma_store.py tests/test_chroma_store.py`

Expected: No errors

**Step 4: Format with ruff**

Run: `ruff format src/ingestion/chroma_store.py tests/test_chroma_store.py`

**Step 5: Commit any formatting changes**

```bash
git add src/ingestion/chroma_store.py tests/test_chroma_store.py
git commit -m "style: format ChromaStore code with ruff"
```

---

## Task 5: Update Phase 2 beads issue and push

**Step 1: Close Phase 2 issue**

Run: `bd close ChromaDB_POC-n72 --reason="Implemented multi-collection support in ChromaStore with get_collection() method and optional collection_name parameters. All tests passing. Backward compatible."`

**Step 2: Sync beads**

Run: `bd sync`

**Step 3: Push to git**

```bash
git push
```

**Step 4: Verify push succeeded**

Run: `git status`

Expected: "Your branch is up to date with 'origin/master'"

---

## Phase 2 Gate Verification

**Run all verification commands:**

```bash
# 1. ChromaStore unit tests pass
pytest tests/test_chroma_store.py -v

# 2. Full test suite passes (no regressions)
pytest tests/ -v

# 3. Linting passes
ruff check src/ingestion/chroma_store.py

# 4. Can create stores with different collections
python -c "
from src.ingestion.chroma_store import ChromaStore
store = ChromaStore(api_key='test', tenant='test', database='test', collection_name='contracts')
print('✓ Can create ChromaStore with contracts collection')
coll = store.get_collection('contracts')
print('✓ Can get contracts collection dynamically')
"

# 5. Backward compatibility verified
python -c "
from src.ingestion.chroma_store import ChromaStore
store = ChromaStore(api_key='test', tenant='test', database='test')
print('✓ Can create ChromaStore with default collection')
coll = store.get_collection()
print('✓ Can get default collection without argument')
"
```

**Expected Results:**
- ✅ All ChromaStore tests pass (new + existing)
- ✅ Full test suite passes without regression
- ✅ No linting errors
- ✅ Can create stores for different collections
- ✅ Backward compatibility maintained
- ✅ Phase 2 issue closed in beads
- ✅ All commits pushed to remote

---

## Next Phase

After Phase 2 is complete, proceed to **Phase 3: Build contract ingestion pipeline** (`ChromaDB_POC-bsn`).

---

**Total Estimated Time:** 1-2 hours
**Test Coverage:** Existing tests + 6 new tests for multi-collection support
**Lines of Code:** ~100 (modifications to existing class) + ~150 (tests)
