from pathlib import Path
from types import SimpleNamespace

import pytest

from word_rag.models import ChunkRecord
from word_rag.rag_service import RagService


class DummyStore:
    def __init__(self):
        self.replaced = False
        self.deleted = False
        self.upserted = False
        self.text_search_called = False

    def replace_document_chunks_with_ids(self, document_name, chunks, embeddings):
        self.replaced = True
        return [{"chunk_id": 1, "chunk_text": "chunk", "document_id": 1}]

    def delete_document(self, document_name):
        self.deleted = True

    def upsert_chunks_with_ids(self, chunks, embeddings):
        self.upserted = True
        return [{"chunk_id": 1, "chunk_text": "chunk", "document_id": 1}]

    def search(self, query_embedding, top_k, fd_number=None, section=None):
        return []

    def search_by_text(self, query_text, top_k, fd_number=None, section=None):
        self.text_search_called = True
        return [SimpleNamespace(id=1, document_name="doc1.docx", fd_number="FD-1", section="S", chunk_text="chunk", distance=1.0)]


class DummyOllama:
    def __init__(self, fallback_works: bool = False):
        self.batch_calls = 0
        self.single_calls = 0
        self.fallback_works = fallback_works

    def embed_many(self, texts):
        self.batch_calls += 1
        if self.fallback_works:
            raise RuntimeError("batch failed")
        return [[0.1, 0.2] for _ in texts]

    def embed(self, text):
        self.single_calls += 1
        return [0.1, 0.2]



def _service_with(store, ollama):
    service = RagService.__new__(RagService)
    service.settings = SimpleNamespace(
        chunk_min_chars=1,
        chunk_max_chars=100,
        index_min_chars=1,
    )
    service.store = store
    service.ollama = ollama
    return service


def test_replace_uses_atomic_replace_path(tmp_path: Path, monkeypatch):
    (tmp_path / "doc1.docx").write_text("x")

    monkeypatch.setattr("word_rag.rag_service.parse_docx_sections", lambda path: [SimpleNamespace(section="S", text="T")])
    monkeypatch.setattr("word_rag.rag_service.extract_fd_number", lambda name: "FD-1")
    monkeypatch.setattr(
        "word_rag.rag_service.build_chunks",
        lambda **kwargs: [ChunkRecord(document_name=kwargs["document_name"], fd_number="FD-1", section="S", chunk_text="chunk")],
    )
    monkeypatch.setattr("word_rag.rag_service.should_skip_chunk", lambda *args, **kwargs: False)

    store = DummyStore()
    service = _service_with(store, SimpleNamespace(embed=lambda text: [0.1, 0.2], embed_many=lambda texts: [[0.1, 0.2] for _ in texts]))

    result = service.ingest_directory(str(tmp_path), replace=True)

    assert result["documents"] == 1
    assert store.replaced is True
    assert store.deleted is False
    assert store.upserted is False


def test_replace_does_not_delete_if_embedding_fails(tmp_path: Path, monkeypatch):
    (tmp_path / "doc1.docx").write_text("x")

    monkeypatch.setattr("word_rag.rag_service.parse_docx_sections", lambda path: [SimpleNamespace(section="S", text="T")])
    monkeypatch.setattr("word_rag.rag_service.extract_fd_number", lambda name: "FD-1")
    monkeypatch.setattr(
        "word_rag.rag_service.build_chunks",
        lambda **kwargs: [ChunkRecord(document_name=kwargs["document_name"], fd_number="FD-1", section="S", chunk_text="chunk")],
    )
    monkeypatch.setattr("word_rag.rag_service.should_skip_chunk", lambda *args, **kwargs: False)

    store = DummyStore()

    def _explode(_text):
        raise RuntimeError("embed failed")

    service = _service_with(store, SimpleNamespace(embed=_explode, embed_many=lambda texts: [_explode(t) for t in texts]))

    with pytest.raises(RuntimeError):
        service.ingest_directory(str(tmp_path), replace=True)

    assert store.replaced is False
    assert store.deleted is False
    assert store.upserted is False


def test_search_uses_text_fallback_when_semantic_results_empty():
    store = DummyStore()
    service = _service_with(
        store,
        SimpleNamespace(embed=lambda text: [0.1, 0.2], embed_many=lambda texts: [[0.1, 0.2] for _ in texts]),
    )
    service.settings.top_k = 3

    results = service.search("негабарит им")

    assert len(results) == 1
    assert store.text_search_called is True


def test_ingest_uses_batch_embeddings(tmp_path: Path, monkeypatch):
    (tmp_path / "doc1.docx").write_text("x")

    monkeypatch.setattr("word_rag.rag_service.parse_docx_sections", lambda path: [SimpleNamespace(section="S", text="T")])
    monkeypatch.setattr("word_rag.rag_service.extract_fd_number", lambda name: "FD-1")
    monkeypatch.setattr(
        "word_rag.rag_service.build_chunks",
        lambda **kwargs: [
            ChunkRecord(document_name=kwargs["document_name"], fd_number="FD-1", section="S", chunk_text="chunk1"),
            ChunkRecord(document_name=kwargs["document_name"], fd_number="FD-1", section="S", chunk_text="chunk2"),
        ],
    )
    monkeypatch.setattr("word_rag.rag_service.should_skip_chunk", lambda *args, **kwargs: False)

    store = DummyStore()
    ollama = DummyOllama()
    service = _service_with(store, ollama)

    service.ingest_directory(str(tmp_path), replace=True)

    assert ollama.batch_calls == 1
    assert ollama.single_calls == 0


def test_ingest_falls_back_to_single_embeddings_when_batch_fails(tmp_path: Path, monkeypatch):
    (tmp_path / "doc1.docx").write_text("x")

    monkeypatch.setattr("word_rag.rag_service.parse_docx_sections", lambda path: [SimpleNamespace(section="S", text="T")])
    monkeypatch.setattr("word_rag.rag_service.extract_fd_number", lambda name: "FD-1")
    monkeypatch.setattr(
        "word_rag.rag_service.build_chunks",
        lambda **kwargs: [
            ChunkRecord(document_name=kwargs["document_name"], fd_number="FD-1", section="S", chunk_text="chunk1"),
            ChunkRecord(document_name=kwargs["document_name"], fd_number="FD-1", section="S", chunk_text="chunk2"),
        ],
    )
    monkeypatch.setattr("word_rag.rag_service.should_skip_chunk", lambda *args, **kwargs: False)

    store = DummyStore()
    ollama = DummyOllama(fallback_works=True)
    service = _service_with(store, ollama)

    service.ingest_directory(str(tmp_path), replace=True)

    assert ollama.batch_calls == 1
    assert ollama.single_calls == 2
