from dataclasses import dataclass


@dataclass
class SectionText:
    section: str
    text: str


@dataclass
class ChunkRecord:
    document_name: str
    fd_number: str | None
    section: str
    chunk_text: str


@dataclass
class SearchResult:
    id: int
    document_name: str
    fd_number: str | None
    section: str
    chunk_text: str
    distance: float
