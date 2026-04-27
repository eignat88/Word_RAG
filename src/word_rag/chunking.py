from __future__ import annotations

from .models import ChunkRecord, SectionText


def chunk_text(text: str, min_chars: int = 300, max_chars: int = 1000) -> list[str]:
    """Split text by paragraphs while keeping chunk sizes bounded."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current and (current_len + para_len + 1 > max_chars):
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len + 1

    if current:
        chunks.append("\n".join(current))

    # Merge very short tail chunks into previous one when possible.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars and len(chunks[-2]) + len(chunks[-1]) + 1 <= max_chars:
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
        chunks.pop()

    return chunks


def build_chunks(document_name: str, fd_number: str | None, sections: list[SectionText], min_chars: int, max_chars: int) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    for section in sections:
        for part in chunk_text(section.text, min_chars=min_chars, max_chars=max_chars):
            out.append(
                ChunkRecord(
                    document_name=document_name,
                    fd_number=fd_number,
                    section=section.section,
                    chunk_text=part,
                )
            )
    return out
