from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import ChunkRecord, SearchResult


class SQLiteKnowledgeBaseStore:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        self._init_db()

    def _init_db(self) -> None:
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    fd_number TEXT,
                    section TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS knowledge_base_fd_number_idx ON knowledge_base(fd_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS knowledge_base_section_idx ON knowledge_base(section)")

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], embeddings: Iterable[list[float]]) -> int:
        return len(self.upsert_chunks_with_ids(chunks, embeddings))

    def upsert_chunks_with_ids(self, chunks: Iterable[ChunkRecord], embeddings: Iterable[list[float]]) -> list[dict]:
        rows = list(zip(chunks, embeddings, strict=True))
        if not rows:
            return []

        saved: list[dict] = []
        with sqlite3.connect(self.sqlite_path) as conn:
            for chunk, emb in rows:
                cur = conn.execute(
                    """
                    INSERT INTO knowledge_base (document_name, fd_number, section, chunk_text, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.document_name,
                        chunk.fd_number,
                        chunk.section,
                        chunk.chunk_text,
                        json.dumps(emb),
                    ),
                )
                saved.append({"chunk_id": int(cur.lastrowid), "chunk_text": chunk.chunk_text, "document_id": None})
        return saved

    def delete_document(self, document_name: str) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM knowledge_base WHERE document_name = ?", (document_name,))

    def replace_document_chunks_with_ids(
        self,
        document_name: str,
        chunks: Iterable[ChunkRecord],
        embeddings: Iterable[list[float]],
    ) -> list[dict]:
        rows = list(zip(chunks, embeddings, strict=True))
        if not rows:
            return []

        saved: list[dict] = []
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM knowledge_base WHERE document_name = ?", (document_name,))
            for chunk, emb in rows:
                cur = conn.execute(
                    """
                    INSERT INTO knowledge_base (document_name, fd_number, section, chunk_text, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.document_name,
                        chunk.fd_number,
                        chunk.section,
                        chunk.chunk_text,
                        json.dumps(emb),
                    ),
                )
                saved.append({"chunk_id": int(cur.lastrowid), "chunk_text": chunk.chunk_text, "document_id": None})
        return saved

    def search(self, query_embedding: list[float], top_k: int, fd_number: str | None = None, section: str | None = None) -> list[SearchResult]:
        clauses = []
        params: list[str] = []
        if fd_number:
            clauses.append("fd_number = ?")
            params.append(fd_number)
        if section:
            clauses.append("section = ?")
            params.append(section)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT id, document_name, fd_number, section, chunk_text, embedding FROM knowledge_base {where}"

        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(sql, params).fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row[5])
            dist = _cosine_distance(query_embedding, emb)
            scored.append((dist, row))

        scored.sort(key=lambda x: x[0])
        top_rows = scored[:top_k]

        return [
            SearchResult(
                id=row[0],
                document_name=row[1],
                fd_number=row[2],
                section=row[3],
                chunk_text=row[4],
                distance=float(dist),
            )
            for dist, row in top_rows
        ]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1 - (dot / (na * nb))
