from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable
import math

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
        self._schema_loaded = False
        self._doc_columns: set[str] = set()
        self._chunk_columns: set[str] = set()
        self._embedding_is_vector = False

    def _fd_col(self) -> str:
        return "dax_code" if "dax_code" in self._doc_columns else "fd_number"

    @contextmanager
    def connection(self):
        with psycopg.connect(self.database_url) as conn:
            yield conn

    def _docs_ident(self):
        return sql.Identifier(self.schema, self.documents_table)

    def _chunks_ident(self):
        return sql.Identifier(self.schema, self.chunks_table)

    def _load_schema_info(self, cur) -> None:
        if self._schema_loaded:
            return

        cur.execute(
            """
            SELECT table_name, column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name IN (%s, %s)
            """,
            (self.schema, self.documents_table, self.chunks_table),
        )
        rows = cur.fetchall()

        for table_name, column_name, data_type, udt_name in rows:
            if table_name == self.documents_table:
                self._doc_columns.add(column_name)
            elif table_name == self.chunks_table:
                self._chunk_columns.add(column_name)
                if column_name == "embedding":
                    self._embedding_is_vector = data_type == "USER-DEFINED" and udt_name == "vector"

        self._schema_loaded = True

    def _build_document_upsert_sql(self):
        if {"title", "source_file", "dax_code"}.issubset(self._doc_columns):
            return sql.SQL(
                """
                INSERT INTO {docs} (title, source_file, dax_code, created_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (source_file)
                DO UPDATE SET
                    dax_code = EXCLUDED.dax_code,
                    created_at = now()
                RETURNING id
                """
            ).format(docs=self._docs_ident()), "new"

        return sql.SQL(
            """
            INSERT INTO {docs} (document_name, fd_number, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (document_name)
            DO UPDATE SET
                fd_number = EXCLUDED.fd_number,
                updated_at = now()
            RETURNING id
            """
        ).format(docs=self._docs_ident()), "legacy"

    def _build_chunk_insert_sql(self):
        if "section_title" in self._chunk_columns:
            cols = ["document_id", "section_title", "chunk_text", "embedding"]
            values = ["%s", "%s", "%s", "%s::vector" if self._embedding_is_vector else "%s"]

            if "section_number" in self._chunk_columns:
                cols.append("section_number")
                values.append("%s")
            if "chunk_type" in self._chunk_columns:
                cols.append("chunk_type")
                values.append("%s")

            query = sql.SQL("INSERT INTO {chunks} ({cols}) VALUES ({vals})").format(
                chunks=self._chunks_ident(),
                cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols),
                vals=sql.SQL(", ").join(sql.SQL(v) for v in values),
            )
            return query, "new"

        query = sql.SQL(
            """
            INSERT INTO {chunks} (document_id, section, chunk_text, embedding)
            VALUES (%s, %s, %s, {emb})
            """
        ).format(chunks=self._chunks_ident(), emb=sql.SQL("%s::vector") if self._embedding_is_vector else sql.SQL("%s"))
        return query, "legacy"

    def _upsert_chunks_with_ids_in_tx(self, cur, rows: list[tuple[ChunkRecord, list[float]]]) -> list[dict]:
        self._load_schema_info(cur)
        upsert_doc_sql, doc_mode = self._build_document_upsert_sql()
        insert_chunk_sql, chunk_mode = self._build_chunk_insert_sql()

        saved: list[dict] = []
        for chunk, emb in rows:
            if doc_mode == "new":
                cur.execute(upsert_doc_sql, (chunk.document_name, chunk.document_name, chunk.fd_number))
            else:
                cur.execute(upsert_doc_sql, (chunk.document_name, chunk.fd_number))
            document_id = cur.fetchone()[0]

            embedding_value = emb if not self._embedding_is_vector else "[" + ",".join(f"{x:.8f}" for x in emb) + "]"

            if chunk_mode == "new":
                params: list[object] = [document_id, chunk.section, chunk.chunk_text, embedding_value]
                if "section_number" in self._chunk_columns:
                    params.append(0)
                if "chunk_type" in self._chunk_columns:
                    params.append("section")
                cur.execute(insert_chunk_sql + sql.SQL(" RETURNING id"), params)
            else:
                cur.execute(insert_chunk_sql + sql.SQL(" RETURNING id"), (document_id, chunk.section, chunk.chunk_text, embedding_value))

            chunk_id = cur.fetchone()[0]
            saved.append({"chunk_id": chunk_id, "chunk_text": chunk.chunk_text, "document_id": document_id})

        return saved

    def upsert_chunks_with_ids(self, chunks: Iterable[ChunkRecord], embeddings: Iterable[list[float]]) -> list[dict]:
        rows = list(zip(chunks, embeddings, strict=True))
        if not rows:
            return []

        with self.connection() as conn, conn.cursor() as cur:
            saved = self._upsert_chunks_with_ids_in_tx(cur, rows)
            conn.commit()
            return saved

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], embeddings: Iterable[list[float]]) -> int:
        return len(self.upsert_chunks_with_ids(chunks, embeddings))

    def delete_document(self, document_name: str) -> None:
        with self.connection() as conn, conn.cursor() as cur:
            self._load_schema_info(cur)
            if "source_file" in self._doc_columns:
                delete_sql = sql.SQL("DELETE FROM {docs} WHERE source_file = %s").format(docs=self._docs_ident())
            else:
                delete_sql = sql.SQL("DELETE FROM {docs} WHERE document_name = %s").format(docs=self._docs_ident())

            cur.execute(delete_sql, (document_name,))
            conn.commit()

    def replace_document_chunks_with_ids(
        self,
        document_name: str,
        chunks: Iterable[ChunkRecord],
        embeddings: Iterable[list[float]],
    ) -> list[dict]:
        rows = list(zip(chunks, embeddings, strict=True))
        if not rows:
            return []

        with self.connection() as conn, conn.cursor() as cur:
            self._load_schema_info(cur)
            if "source_file" in self._doc_columns:
                delete_sql = sql.SQL("DELETE FROM {docs} WHERE source_file = %s").format(docs=self._docs_ident())
            else:
                delete_sql = sql.SQL("DELETE FROM {docs} WHERE document_name = %s").format(docs=self._docs_ident())

            cur.execute(delete_sql, (document_name,))
            saved = self._upsert_chunks_with_ids_in_tx(cur, rows)
            conn.commit()
            return saved

    def search(self, query_embedding: list[float], top_k: int, fd_number: str | None = None, section: str | None = None) -> list[SearchResult]:
        with self.connection() as conn, conn.cursor() as cur:
            self._load_schema_info(cur)
            return self._search_python_cosine(cur, query_embedding, top_k, fd_number, section)

    def search_by_text(self, query_text: str, top_k: int, fd_number: str | None = None, section: str | None = None) -> list[SearchResult]:
        with self.connection() as conn, conn.cursor() as cur:
            self._load_schema_info(cur)
            return self._search_text(cur, query_text, top_k, fd_number, section)

    def _search_vector(self, cur, query_embedding: list[float], top_k: int, fd_number: str | None, section: str | None) -> list[SearchResult]:
        emb_sql = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

        doc_name_col = "title" if "title" in self._doc_columns else "document_name"
        fd_col = "dax_code" if "dax_code" in self._doc_columns else "fd_number"
        section_col = "section_title" if "section_title" in self._chunk_columns else "section"

        conditions = []
        filter_params: list[object] = []
        if fd_number:
            conditions.append(sql.SQL("d.{fd} = %s").format(fd=sql.Identifier(fd_col)))
            filter_params.append(fd_number)
        if section:
            conditions.append(sql.SQL("c.{section} = %s").format(section=sql.Identifier(section_col)))
            filter_params.append(section)

        where_sql = sql.SQL("")
        if conditions:
            where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

        query_sql = sql.SQL(
            """
            SELECT c.id,
                   d.{doc_name},
                   d.{fd_col},
                   c.{section_col},
                   c.chunk_text,
                   c.embedding <-> %s::vector AS distance
            FROM {chunks} c
            JOIN {docs} d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.embedding <-> %s::vector
            LIMIT %s
            """
        ).format(
            doc_name=sql.Identifier(doc_name_col),
            fd_col=sql.Identifier(fd_col),
            section_col=sql.Identifier(section_col),
            chunks=self._chunks_ident(),
            docs=self._docs_ident(),
            where_clause=where_sql,
        )

        params: list[object] = [emb_sql, *filter_params, emb_sql, top_k]
        cur.execute(query_sql, params)
        rows = cur.fetchall()

        return [
            SearchResult(id=r[0], document_name=r[1], fd_number=r[2], section=r[3], chunk_text=r[4], distance=float(r[5]))
            for r in rows
        ]

    def _search_python_cosine(self, cur, query_embedding: list[float], top_k: int, fd_number: str | None, section: str | None) -> list[SearchResult]:
        doc_name_col = "title" if "title" in self._doc_columns else "document_name"
        fd_col = "dax_code" if "dax_code" in self._doc_columns else "fd_number"
        section_col = "section_title" if "section_title" in self._chunk_columns else "section"

        conditions = []
        params: list[object] = []
        if fd_number:
            conditions.append(sql.SQL("d.{fd} = %s").format(fd=sql.Identifier(fd_col)))
            params.append(fd_number)
        if section:
            conditions.append(sql.SQL("c.{section} = %s").format(section=sql.Identifier(section_col)))
            params.append(section)

        where_sql = sql.SQL("")
        if conditions:
            where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

        query_sql = sql.SQL(
            """
            SELECT c.id,
                   d.{doc_name},
                   d.{fd_col},
                   c.{section_col},
                   c.chunk_text,
                   c.embedding
            FROM {chunks} c
            JOIN {docs} d ON d.id = c.document_id
            {where_clause}
            """
        ).format(
            doc_name=sql.Identifier(doc_name_col),
            fd_col=sql.Identifier(fd_col),
            section_col=sql.Identifier(section_col),
            chunks=self._chunks_ident(),
            docs=self._docs_ident(),
            where_clause=where_sql,
        )

        cur.execute(query_sql, params)
        rows = cur.fetchall()

        scored = []
        for row in rows:
            emb = row[5] or []
            distance = _cosine_distance(query_embedding, emb)
            scored.append((distance, row))

        scored.sort(key=lambda x: x[0])
        top_rows = scored[:top_k]

        return [
            SearchResult(id=row[0], document_name=row[1], fd_number=row[2], section=row[3], chunk_text=row[4], distance=float(distance))
            for distance, row in top_rows
        ]

    def _search_text(self, cur, query_text: str, top_k: int, fd_number: str | None, section: str | None) -> list[SearchResult]:
        doc_name_col = "title" if "title" in self._doc_columns else "document_name"
        fd_col = "dax_code" if "dax_code" in self._doc_columns else "fd_number"
        section_col = "section_title" if "section_title" in self._chunk_columns else "section"

        conditions = [sql.SQL("c.chunk_text ILIKE %s")]
        params: list[object] = [f"%{query_text.strip()}%"]
        if fd_number:
            conditions.append(sql.SQL("d.{fd} = %s").format(fd=sql.Identifier(fd_col)))
            params.append(fd_number)
        if section:
            conditions.append(sql.SQL("c.{section} = %s").format(section=sql.Identifier(section_col)))
            params.append(section)

        where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
        query_sql = sql.SQL(
            """
            SELECT c.id,
                   d.{doc_name},
                   d.{fd_col},
                   c.{section_col},
                   c.chunk_text
            FROM {chunks} c
            JOIN {docs} d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.id DESC
            LIMIT %s
            """
        ).format(
            doc_name=sql.Identifier(doc_name_col),
            fd_col=sql.Identifier(fd_col),
            section_col=sql.Identifier(section_col),
            chunks=self._chunks_ident(),
            docs=self._docs_ident(),
            where_clause=where_sql,
        )
        params.append(top_k)
        cur.execute(query_sql, params)
        rows = cur.fetchall()
        return [
            SearchResult(id=row[0], document_name=row[1], fd_number=row[2], section=row[3], chunk_text=row[4], distance=1.0)
            for row in rows
        ]

    def upsert_entity(self, conn, entity_type: str, entity_value: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai.fd_entities (entity_type, entity_value)
                VALUES (%s, %s)
                ON CONFLICT (entity_type, entity_value)
                DO UPDATE SET entity_value = EXCLUDED.entity_value
                RETURNING id
                """,
                (entity_type, entity_value),
            )
            return cur.fetchone()[0]

    def link_chunk_entity(self, conn, chunk_id: int, entity_id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai.fd_chunk_entities (chunk_id, entity_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (chunk_id, entity_id),
            )

    def find_document_id_by_fd_number(self, conn, dax_number: str) -> int | None:
        fd_col = self._fd_col()
        with conn.cursor() as cur:
            query = sql.SQL("SELECT id FROM {docs} WHERE {fd} = %s LIMIT 1").format(
                docs=self._docs_ident(),
                fd=sql.Identifier(fd_col),
            )
            cur.execute(query, (dax_number,))
            row = cur.fetchone()
            return int(row[0]) if row else None

    def insert_document_link(self, conn, source_document_id: int, link_type: str, target_document_id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai.fd_document_links (
                    source_document_id,
                    link_type,
                    target_document_id
                )
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (source_document_id, link_type, target_document_id),
            )


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1 - (dot / (na * nb))
