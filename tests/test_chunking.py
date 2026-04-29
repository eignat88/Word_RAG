from word_rag.chunking import build_chunks, chunk_text
from word_rag.models import SectionText


def test_chunk_text_respects_max_len():
    text = "\n".join(["a" * 180 for _ in range(8)])
    chunks = chunk_text(text, min_chars=100, max_chars=400, overlap_chars=150)
    assert chunks
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_chunk_text_has_overlap_between_neighbors():
    text = "\n".join(["A" * 220, "B" * 220, "C" * 220, "D" * 220])
    chunks = chunk_text(text, min_chars=100, max_chars=500, overlap_chars=150)
    assert len(chunks) >= 2
    for prev, cur in zip(chunks, chunks[1:]):
        shared = prev[-120:]
        assert shared in cur


def test_chunk_text_single_short_paragraph():
    text = "Короткий абзац."
    chunks = chunk_text(text, min_chars=100, max_chars=400, overlap_chars=150)
    assert chunks == [text]


def test_build_chunks_keeps_section_boundaries():
    sections = [
        SectionText(section="Алгоритм", text="x" * 500),
        SectionText(section="Интерфейс", text="y" * 500),
    ]
    chunks = build_chunks("DAX-1.docx", "DAX-1", sections, min_chars=100, max_chars=400, overlap_chars=150)
    assert any(c.section == "Алгоритм" for c in chunks)
    assert any(c.section == "Интерфейс" for c in chunks)
    assert all(("x" in c.chunk_text and c.section == "Алгоритм") or ("y" in c.chunk_text and c.section == "Интерфейс") for c in chunks)
