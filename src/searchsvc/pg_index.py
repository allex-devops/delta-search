"""Postgres + pgvector version of SearchIndex, same shape, for scale and concurrent writers.

Chroma's PersistentClient is a single embedded process: two servers pointed at the same folder can corrupt it.
A real database with row locks lets several API workers write at once.
"""
import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from .chunking import chunk_paragraphs
from .filters import FilterError
from .index import Doc, EmbeddingMismatch, Result, SyncResult, UpsertResult, clean_metadata

SCHEMA = """
create extension if not exists vector;
create table if not exists chunks (
    id text primary key,
    doc_id text not null,
    text text not null,
    metadata jsonb not null,
    embedding vector({dim})
);
create index if not exists chunks_doc_id on chunks (doc_id);
create index if not exists chunks_metadata on chunks using gin (metadata);
create table if not exists index_meta (key text primary key, value text not null);
"""


def _pg_filter(filt: dict | None) -> tuple[str, list]:
    """A tiny subset of the same filter shape, as SQL over the jsonb metadata column."""
    if not filt:
        return "", []
    clauses, params = [], []
    for field, cond in filt.items():
        if not field.replace("_", "").isalnum():
            raise FilterError(f"can't filter on '{field}'")
        if isinstance(cond, dict):
            ops = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            for op, val in cond.items():
                if op in ops:
                    clauses.append(f"(metadata->>{field!r})::numeric {ops[op]} %s")
                    params.append(val)
                elif op == "eq":
                    clauses.append(f"metadata->>{field!r} = %s")
                    params.append(str(val))
                elif op == "in":
                    if not isinstance(val, list) or not val:
                        raise FilterError("'in' needs a non-empty list")
                    clauses.append(f"metadata->>{field!r} = ANY(%s)")
                    params.append([str(v) for v in val])
                else:
                    raise FilterError(f"unknown operator '{op}' for '{field}'")
        elif isinstance(cond, list):
            if not cond:
                raise FilterError(f"empty condition for '{field}'")
            clauses.append(f"metadata->>{field!r} = ANY(%s)")
            params.append([str(v) for v in cond])
        else:
            clauses.append(f"metadata->>{field!r} = %s")
            params.append(str(cond))
    return " and " + " and ".join(clauses) if clauses else "", params


@dataclass
class PgSearchIndex:
    dsn: str
    embed: Callable[[list[str], str], np.ndarray]
    embed_model: str
    dim: int = 1536  # must match your embedding model

    def __post_init__(self):
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(SCHEMA.format(dim=self.dim))
            row = conn.execute("select value from index_meta where key = 'embed_model'").fetchone()
            if row and row[0] != self.embed_model:
                raise EmbeddingMismatch(f"index was built with '{row[0]}' but this is '{self.embed_model}'; re-index, or switch back")
            if not row:
                conn.execute("insert into index_meta values ('embed_model', %s)", (self.embed_model,))

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def upsert(self, doc_id: str, text: str, metadata: dict | None = None) -> UpsertResult:
        if not text.strip():
            raise ValueError("document has no text")
        meta = clean_metadata(metadata)
        new = {f"{doc_id}#{hashlib.sha1(c.encode()).hexdigest()[:16]}": c for c in chunk_paragraphs(text)}

        with self._connect() as conn:
            existing = conn.execute("select id, metadata from chunks where doc_id = %s", (doc_id,)).fetchall()
            old_meta = {r["id"]: r["metadata"] for r in existing}
            to_add = [i for i in new if i not in old_meta]
            to_remove = [i for i in old_meta if i not in new]
            kept = [i for i in new if i in old_meta]
            meta_changed = [i for i in kept if old_meta[i] != meta]

            with conn.cursor() as cur:
                if to_add:
                    vectors = self.embed([new[i] for i in to_add], "document")
                    cur.executemany(
                        "insert into chunks (id, doc_id, text, metadata, embedding) values (%s, %s, %s, %s, %s::vector)",
                        [(i, doc_id, new[i], Json(meta), str(list(map(float, v)))) for i, v in zip(to_add, vectors)],
                    )
                if to_remove:
                    cur.execute("delete from chunks where id = ANY(%s)", (to_remove,))
                if meta_changed:
                    cur.executemany("update chunks set metadata = %s where id = %s", [(Json(meta), i) for i in meta_changed])
            conn.commit()

        status = "added" if not old_meta else "updated" if (to_add or to_remove or meta_changed) else "unchanged"
        return UpsertResult(status, len(to_add), len(to_remove), len(kept))

    def delete(self, doc_id: str) -> bool:
        with self._connect() as conn:
            n = conn.execute("delete from chunks where doc_id = %s", (doc_id,)).rowcount
            conn.commit()
        return n > 0

    def sync(self, docs: list[Doc]) -> SyncResult:
        result, seen = SyncResult(), set()
        for d in docs:
            r = self.upsert(d.id, d.text, d.metadata)
            seen.add(d.id)
            result.chunks_embedded += r.chunks_embedded
            result.added += r.status == "added"
            result.updated += r.status == "updated"
            result.unchanged += r.status == "unchanged"
        for doc_id in self.doc_ids() - seen:
            self.delete(doc_id)
            result.removed += 1
        return result

    def doc_ids(self) -> set[str]:
        with self._connect() as conn:
            return {r["doc_id"] for r in conn.execute("select distinct doc_id from chunks")}

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("select count(*) as chunks, count(distinct doc_id) as documents from chunks").fetchone()
        return dict(row)

    def search(self, query: str, k: int = 5, filter: dict | None = None) -> list[Result]:
        return self.search_vector(self.embed([query], "query")[0], k, filter)

    def search_vector(self, vector: np.ndarray, k: int = 5, filter: dict | None = None) -> list[Result]:
        where, params = _pg_filter(filter)
        sql = f"""
            select distinct on (doc_id) id, doc_id, text, metadata, embedding <=> %s::vector as dist
            from chunks where true {where}
            order by doc_id, dist
            limit %s
        """
        vec = str(list(map(float, vector)))  # psycopg sends a Python list as an array; pgvector wants "[..]" text
        # cosine distance ranks each document by its best chunk; the outer sort then ranks documents themselves
        with self._connect() as conn:
            rows = conn.execute(
                f"select * from ({sql}) d order by dist limit %s", [vec, *params, k * 20, k]
            ).fetchall()
        return [Result(r["doc_id"], round(1 - r["dist"], 4), r["text"][:300], r["id"], r["metadata"]) for r in rows]
