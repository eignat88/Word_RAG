from word_rag.storage import KnowledgeBaseStore


class DummyCursor:
    def __init__(self):
        self.executed_query = None
        self.executed_params = None

    def execute(self, query, params):
        self.executed_query = str(query)
        self.executed_params = params

    def fetchall(self):
        return [
            (1, "doc-old", "FD-1", "sec", "old text", [1.0, 0.0]),
            (2, "doc-new", "FD-1", "sec", "new text", [0.0, 1.0]),
        ]


def test_python_cosine_fallback_scans_all_filtered_rows():
    store = KnowledgeBaseStore("postgresql://unused")
    store._doc_columns = {"document_name", "fd_number"}
    store._chunk_columns = {"section"}

    cur = DummyCursor()
    results = store._search_python_cosine(cur, [1.0, 0.0], top_k=1, fd_number=None, section=None)

    assert len(results) == 1
    assert results[0].document_name == "doc-old"
    assert "LIMIT" not in cur.executed_query.upper()
    assert cur.executed_params == []


def test_search_prefers_pgvector_path_when_embedding_column_is_vector():
    store = KnowledgeBaseStore("postgresql://unused")
    store._schema_loaded = True
    store._embedding_is_vector = True

    class _DummyConnectionCtx:
        def __init__(self, cursor):
            self._cursor = cursor

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return self._cursor

    class _DummyCursorCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    vector_called = {"value": False}
    cosine_called = {"value": False}

    def _vector(*args, **kwargs):
        vector_called["value"] = True
        return ["vector"]

    def _cosine(*args, **kwargs):
        cosine_called["value"] = True
        return ["cosine"]

    store.connection = lambda: _DummyConnectionCtx(_DummyCursorCtx())  # type: ignore[method-assign]
    store._search_vector = _vector  # type: ignore[method-assign]
    store._search_python_cosine = _cosine  # type: ignore[method-assign]

    results = store.search([0.1, 0.2], top_k=3)

    assert results == ["vector"]
    assert vector_called["value"] is True
    assert cosine_called["value"] is False
