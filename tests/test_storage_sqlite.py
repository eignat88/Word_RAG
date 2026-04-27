from pathlib import Path

from word_rag.models import ChunkRecord
from word_rag.storage_sqlite import SQLiteKnowledgeBaseStore


def test_sqlite_search_and_filters(tmp_path: Path):
    db = tmp_path / "kb.db"
    store = SQLiteKnowledgeBaseStore(str(db))

    chunks = [
        ChunkRecord(document_name="DAX-1.docx", fd_number="DAX-1", section="Алгоритм", chunk_text="alpha"),
        ChunkRecord(document_name="DAX-2.docx", fd_number="DAX-2", section="Интерфейс", chunk_text="beta"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.upsert_chunks(chunks, embeddings)

    result = store.search([1.0, 0.0], top_k=2)
    assert result[0].document_name == "DAX-1.docx"

    filtered = store.search([1.0, 0.0], top_k=2, fd_number="DAX-2")
    assert len(filtered) == 1
    assert filtered[0].fd_number == "DAX-2"
