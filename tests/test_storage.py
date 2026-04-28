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
