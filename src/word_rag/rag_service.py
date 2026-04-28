from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

from .chunking import build_chunks
from .config import Settings
from .docx_parser import extract_fd_number, parse_docx_sections
from .embeddings import OllamaClient
from .entity_extractor import extract_dax_numbers, extract_entities
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
            embed_timeout_sec=settings.embed_timeout_sec,
            llm_timeout_sec=settings.llm_timeout_sec,
        )
        if settings.storage_backend.lower() == "postgres":
            self.store = KnowledgeBaseStore(
                database_url=settings.database_url,
                schema=settings.pg_schema,
                documents_table=settings.pg_documents_table,
                chunks_table=settings.pg_chunks_table,
            )
        else:
            self.store = SQLiteKnowledgeBaseStore(sqlite_path=settings.sqlite_path)

    def ingest_directory(
        self,
        directory: str,
        replace: bool = True,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict[str, int | float]:
        base = Path(directory)
        files = sorted(base.glob("*.docx"))
        processed_docs = 0
        inserted_chunks = 0
        skipped_chunks = 0
        start_ts = perf_counter()

        if progress_callback:
            progress_callback({"event": "start", "total_documents": len(files), "directory": str(base)})

        for idx, path in enumerate(files, start=1):
            doc_start_ts = perf_counter()
            if progress_callback:
                progress_callback({"event": "document_start", "index": idx, "total_documents": len(files), "document_name": path.name})
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
            doc_skipped = 0
            for chunk in chunks:
                if should_skip_chunk(chunk.chunk_text, chunk.section, min_chars=self.settings.index_min_chars):
                    skipped_chunks += 1
                    doc_skipped += 1
                    continue
                filtered_chunks.append(chunk)

            if replace:
                self.store.delete_document(path.name)

            embeddings = [self.ollama.embed(c.chunk_text) for c in filtered_chunks]
            saved_chunks = self.store.upsert_chunks_with_ids(filtered_chunks, embeddings)
            inserted_for_doc = len(saved_chunks)
            inserted_chunks += inserted_for_doc
            processed_docs += 1

            if saved_chunks and hasattr(self.store, "upsert_entity"):
                full_document_text = "\n".join(chunk["chunk_text"] for chunk in saved_chunks)
                source_document_id = saved_chunks[0]["document_id"]

                with self.store.connection() as conn:
                    for chunk in saved_chunks:
                        entities = extract_entities(chunk["chunk_text"])
                        for entity in entities:
                            entity_id = self.store.upsert_entity(conn, entity["type"], entity["value"])
                            self.store.link_chunk_entity(conn, chunk["chunk_id"], entity_id)

                    for dax_number in extract_dax_numbers(full_document_text):
                        target_document_id = self.store.find_document_id_by_fd_number(conn, dax_number)
                        if target_document_id and target_document_id != source_document_id:
                            self.store.insert_document_link(
                                conn,
                                source_document_id=source_document_id,
                                link_type="referenced",
                                target_document_id=target_document_id,
                            )
                    conn.commit()
            if progress_callback:
                progress_callback(
                    {
                        "event": "document_done",
                        "index": idx,
                        "total_documents": len(files),
                        "document_name": path.name,
                        "inserted_chunks": inserted_for_doc,
                        "skipped_chunks": doc_skipped,
                        "elapsed_sec": round(perf_counter() - doc_start_ts, 2),
                    }
                )

        elapsed = round(perf_counter() - start_ts, 2)
        if progress_callback:
            progress_callback(
                {
                    "event": "done",
                    "documents": processed_docs,
                    "chunks": inserted_chunks,
                    "skipped_chunks": skipped_chunks,
                    "elapsed_sec": elapsed,
                }
            )
        return {"documents": processed_docs, "chunks": inserted_chunks, "skipped_chunks": skipped_chunks, "elapsed_sec": elapsed}

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
