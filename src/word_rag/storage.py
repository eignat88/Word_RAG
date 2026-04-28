from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import psycopg
from psycopg import sql

from .models import ChunkRecord, SearchResult


class KnowledgeBaseStore:
    def __init__(
        self,
        database_url: str,
        embedding_dim: int = 768,
        schema: str = "ai",
        documents_table: str = "fd_documents",
        chunks_table: str = "fd_chunks",
    ) -> None:
        self.database_url = database_url
        self.embedding_dim = embedding_dim
        self.schema = schema
        self.documents_table = documents_table
        self.chunks_table = chunks_table

    @contextmanager
    def connection(self):
        with psycopg.connect(self.database_url) as conn:
            yield conn

    def _docs_ident(self):
        return sql.Identifier(self.schema, self.documents_table)

    def _chunks_ident(self):
        return sql.Identifier(self.schema, self.chunks_table)

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], embeddings: Iterable[list[float]]) -> int:
        rows = list(zip(chunks, embeddings, strict=True))
        if not rows:
            return 0

        upsert_doc_sql = sql.SQL(
            """
            INSERT INTO {docs} (document_name, fd_number, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (document_name)
            DO UPDATE SET
                fd_number = EXCLUDED.fd_number,
                updated_at = now()
            RETURNING id
            """
        ).format(docs=self._docs_ident())

        insert_chunk_sql = sql.SQL(
            """
            INSERT INTO {chunks} (document_id, section, chunk_text, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """
        ).format(chunks=self._chunks_ident())

        with self.connection() as conn, conn.cursor() as cur:
            for chunk, emb in rows:
                cur.execute(upsert_doc_sql, (chunk.document_name, chunk.fd_number))
                document_id = cur.fetchone()[0]

                emb_sql = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                cur.execute(insert_chunk_sql, (document_id, chunk.section, chunk.chunk_text, emb_sql))
            conn.commit()
        return len(rows)

    def delete_document(self, document_name: str) -> None:
        delete_sql = sql.SQL("DELETE FROM {docs} WHERE document_name = %s").format(docs=self._docs_ident())
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(delete_sql, (document_name,))
            conn.commit()

    def search(self, query_embedding: list[float], top_k: int, fd_number: str | None = None, section: str | None = None) -> list[SearchResult]:
        emb_sql = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

        conditions = []
        filter_params: list[object] = []
        if fd_number:
            conditions.append(sql.SQL("d.fd_number = %s"))
            filter_params.append(fd_number)
        if section:
            conditions.append(sql.SQL("c.section = %s"))
            filter_params.append(section)

        where_sql = sql.SQL("")
        if conditions:
            where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

        query_sql = sql.SQL(
            """
            SELECT c.id,
                   d.document_name,
                   d.fd_number,
                   c.section,
                   c.chunk_text,
                   c.embedding <-> %s::vector AS distance
            FROM {chunks} c
            JOIN {docs} d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.embedding <-> %s::vector
            LIMIT %s
            """
        ).format(chunks=self._chunks_ident(), docs=self._docs_ident(), where_clause=where_sql)

        params: list[object] = [emb_sql, *filter_params, emb_sql, top_k]
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, params)
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
