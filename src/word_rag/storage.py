from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import psycopg

from .models import ChunkRecord, SearchResult


class KnowledgeBaseStore:
    def __init__(self, database_url: str, embedding_dim: int = 768) -> None:
        self.database_url = database_url
        self.embedding_dim = embedding_dim

    @contextmanager
    def connection(self):
        with psycopg.connect(self.database_url) as conn:
            yield conn

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], embeddings: Iterable[list[float]]) -> int:
        rows = list(zip(chunks, embeddings, strict=True))
        if not rows:
            return 0

        with self.connection() as conn, conn.cursor() as cur:
            for chunk, emb in rows:
                emb_sql = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                cur.execute(
                    """
                    INSERT INTO knowledge_base (document_name, fd_number, section, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (chunk.document_name, chunk.fd_number, chunk.section, chunk.chunk_text, emb_sql),
                )
            conn.commit()
        return len(rows)

    def delete_document(self, document_name: str) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_base WHERE document_name = %s", (document_name,))
            conn.commit()

    def search(self, query_embedding: list[float], top_k: int, fd_number: str | None = None, section: str | None = None) -> list[SearchResult]:
        emb_sql = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

        conditions = []
        filter_params: list[object] = []
        if fd_number:
            conditions.append("fd_number = %s")
            filter_params.append(fd_number)
        if section:
            conditions.append("section = %s")
            filter_params.append(section)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params: list[object] = [emb_sql, *filter_params, emb_sql, top_k]

        sql = f"""
            SELECT id, document_name, fd_number, section, chunk_text,
                   embedding <-> %s::vector AS distance
            FROM knowledge_base
            {where_clause}
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """

        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            SearchResult(
                id=r[0],
                document_name=r[1],
                fd_number=r[2],
                section=r[3],
                chunk_text=r[4],
                distance=float(r[5]),
            )
            for r in rows
        ]
