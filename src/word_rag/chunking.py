from __future__ import annotations

from .models import ChunkRecord, SectionText


def _tail_with_paragraph_preference(chunk: str, overlap: int) -> str:
    if overlap <= 0 or not chunk:
        return ""
    if len(chunk) <= overlap:
        return chunk
    window = chunk[-min(len(chunk), overlap * 2):]
    boundary = window.find("\n")
    if boundary != -1 and (len(window) - boundary - 1) >= overlap:
        return window[boundary + 1:]
    return chunk[-overlap:]


def chunk_text(text: str, min_chars: int = 300, max_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    """Split text by paragraphs while keeping chunk sizes bounded and adding overlap."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    overlap = max(0, min(overlap_chars, max_chars // 2))
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_chars:
            if current:
                ready = "\n".join(current)
                chunks.append(ready)
                tail = _tail_with_paragraph_preference(ready, overlap)
                current = [tail] if tail else []
                current_len = len(tail)
            step = max(1, max_chars - overlap)
            start = 0
            while start < para_len:
                end = min(start + max_chars, para_len)
                part = para[start:end]
                chunks.append(part)
                if end >= para_len:
                    tail = _tail_with_paragraph_preference(part, overlap)
                    current = [tail] if tail else []
                    current_len = len(tail)
                    break
                start += step
            continue

        extra = para_len + (1 if current else 0)
        if current and (current_len + extra > max_chars):
            ready = "\n".join(current)
            chunks.append(ready)
            tail = _tail_with_paragraph_preference(ready, overlap)
            current = [tail] if tail else []
            current_len = len(tail)
        if current:
            current_len += 1
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n".join(current))

    # Merge very short tail chunks into previous one when possible.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars and len(chunks[-2]) + len(chunks[-1]) + 1 <= max_chars:
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
        chunks.pop()

    return chunks


def build_chunks(
    document_name: str,
    fd_number: str | None,
    sections: list[SectionText],
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    for section in sections:
        for part in chunk_text(section.text, min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars):
            out.append(
                ChunkRecord(
                    document_name=document_name,
                    fd_number=fd_number,
                    section=section.section,
                    chunk_text=part,
                )
            )
    return out
