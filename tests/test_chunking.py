from word_rag.chunking import chunk_text, build_chunks
from word_rag.models import SectionText


def test_chunk_text_respects_max_len():
    text = "\n".join(["a" * 180 for _ in range(8)])
    chunks = chunk_text(text, min_chars=100, max_chars=400)
    assert chunks
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_build_chunks_keeps_section_boundaries():
    sections = [
        SectionText(section="Алгоритм", text="x" * 500),
        SectionText(section="Интерфейс", text="y" * 500),
    ]
    chunks = build_chunks("DAX-1.docx", "DAX-1", sections, min_chars=100, max_chars=400)
    assert any(c.section == "Алгоритм" for c in chunks)
    assert any(c.section == "Интерфейс" for c in chunks)
    assert all(("x" in c.chunk_text and c.section == "Алгоритм") or ("y" in c.chunk_text and c.section == "Интерфейс") for c in chunks)
