from __future__ import annotations

from pathlib import Path

from .chunking import build_chunks
from .config import Settings
from .docx_parser import extract_fd_number, parse_docx_sections
from .embeddings import OllamaClient
from .filtering import should_skip_chunk
from .models import SearchResult
from .storage import KnowledgeBaseStore
from .storage_sqlite import SQLiteKnowledgeBaseStore


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama = OllamaClient(
            base_url=settings.ollama_base_url,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
        )
        if settings.storage_backend.lower() == "postgres":
            self.store = KnowledgeBaseStore(database_url=settings.database_url)
        else:
            self.store = SQLiteKnowledgeBaseStore(sqlite_path=settings.sqlite_path)

    def ingest_directory(self, directory: str, replace: bool = True) -> dict[str, int]:
        base = Path(directory)
        processed_docs = 0
        inserted_chunks = 0
        skipped_chunks = 0

        for path in sorted(base.glob("*.docx")):
            sections = parse_docx_sections(path)
            fd_number = extract_fd_number(path.name)

            chunks = build_chunks(
                document_name=path.name,
                fd_number=fd_number,
                sections=sections,
                min_chars=self.settings.chunk_min_chars,
                max_chars=self.settings.chunk_max_chars,
            )

            filtered_chunks = []
            for chunk in chunks:
                if should_skip_chunk(chunk.chunk_text, chunk.section, min_chars=self.settings.index_min_chars):
                    skipped_chunks += 1
                    continue
                filtered_chunks.append(chunk)

            if replace:
                self.store.delete_document(path.name)

            embeddings = [self.ollama.embed(c.chunk_text) for c in filtered_chunks]
            inserted_chunks += self.store.upsert_chunks(filtered_chunks, embeddings)
            processed_docs += 1

        return {"documents": processed_docs, "chunks": inserted_chunks, "skipped_chunks": skipped_chunks}

    def search(self, question: str, fd_number: str | None = None, section: str | None = None, top_k: int | None = None) -> list[SearchResult]:
        query_emb = self.ollama.embed(question)
        return self.store.search(
            query_embedding=query_emb,
            top_k=top_k or self.settings.top_k,
            fd_number=fd_number,
            section=section,
        )

    def answer(self, question: str, fd_number: str | None = None, section: str | None = None, top_k: int | None = None) -> dict:
        results = self.search(question=question, fd_number=fd_number, section=section, top_k=top_k)
        context = "\n\n".join(
            f"[{r.document_name} | {r.section}]\n{r.chunk_text}"
            for r in results
        )
        prompt = (
            "Контекст:\n"
            f"{context if context else 'Контекст не найден.'}\n\n"
            "Вопрос:\n"
            f"{question}\n\n"
            "Ответь строго на основе контекста. Если данных недостаточно — так и скажи."
        )
        response = self.ollama.answer(prompt)
        return {
            "answer": response,
            "sources": [
                {
                    "document_name": r.document_name,
                    "fd_number": r.fd_number,
                    "section": r.section,
                    "distance": r.distance,
                    "chunk_text": r.chunk_text,
                }
                for r in results
            ],
        }
