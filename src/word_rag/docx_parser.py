from __future__ import annotations

import re
from pathlib import Path

from .models import SectionText

SECTION_HEADERS = {
    "назначение",
    "алгоритм",
    "интерфейс",
}

FD_PATTERN = re.compile(r"(DAX-\d+)", re.IGNORECASE)


def extract_fd_number(file_name: str) -> str | None:
    match = FD_PATTERN.search(file_name)
    return match.group(1).upper() if match else None


def parse_docx_sections(path: Path) -> list[SectionText]:
    """Parse .docx into logical sections by heading-like paragraphs."""
    from docx import Document

    doc = Document(path)
    sections: list[SectionText] = []
    current_section = "Прочее"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        joined = "\n".join(x.strip() for x in buffer if x.strip()).strip()
        if joined:
            sections.append(SectionText(section=current_section, text=joined))
        buffer = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        lower = text.lower()
        style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
        is_heading = lower in SECTION_HEADERS or "heading" in style_name or "заголовок" in style_name

        if is_heading:
            flush()
            current_section = text
            continue

        buffer.append(text)

    flush()
    return sections
