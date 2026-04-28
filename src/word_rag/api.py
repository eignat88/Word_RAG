from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import Settings
from .rag_service import RagService


app = FastAPI(title="Word RAG API", version="0.1.0")
settings = Settings()
service = RagService(settings)


class IngestRequest(BaseModel):
    directory: str = Field(..., description="Path to directory with .docx files")
    replace: bool = True


class SearchRequest(BaseModel):
    question: str
    top_k: int | None = None
    fd_number: str | None = None
    section: str | None = None


class AskRequest(SearchRequest):
    pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, int]:
    return service.ingest_directory(payload.directory, replace=payload.replace)


@app.post("/search")
def search(payload: SearchRequest) -> dict:
    results = service.search(
        question=payload.question,
        top_k=payload.top_k,
        fd_number=payload.fd_number,
        section=payload.section,
    )
    return {
        "results": [
            {
                "id": r.id,
                "document_name": r.document_name,
                "fd_number": r.fd_number,
                "section": r.section,
                "chunk_text": r.chunk_text,
                "distance": r.distance,
            }
            for r in results
        ]
    }


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    return service.answer(
        question=payload.question,
        top_k=payload.top_k,
        fd_number=payload.fd_number,
        section=payload.section,
    )
