import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chromadb
import numpy as np

from .chunking import chunk_paragraphs
from .filters import RESERVED, to_where

# (texts, kind) -> one row per text; kind is "document" or "query"
Embedder = Callable[[list[str], str], np.ndarray]
SNIPPET = 300
MAX_TEXT = 500_000


class EmbeddingMismatch(RuntimeError):
    """The index was built with a different embedding model than the one in use now."""


@dataclass
class Doc:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class UpsertResult:
    status: str  # "added", "updated" or "unchanged"
    chunks_embedded: int = 0
    chunks_removed: int = 0
    chunks_kept: int = 0


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks_embedded: int = 0


@dataclass
class Result:
    doc_id: str
    score: float
    snippet: str
    chunk_id: str
    metadata: dict


def clean_metadata(meta: dict | None) -> dict:
    out = {}
    for key, value in (meta or {}).items():
        if key in RESERVED:
            raise ValueError(f"'{key}' is reserved")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"metadata '{key}' must be a string, number or boolean")
        out[key] = value
    return out


class QueryCache:
    """Remembers the vector for recent query strings, because embedding is the slow part of a search."""

    def __init__(self, size: int = 2048):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()
        self.hits = self.misses = 0

    def get_or_compute(self, key: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                self.hits += 1
                return self.items[key]
            self.misses += 1
        value = compute()  # outside the lock so one slow embedding doesn't block every other search
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return value


class SearchIndex:
    def __init__(self, path: Path, embed: Embedder, embed_model: str, cache_size: int = 2048, name: str = "docs"):
        path = Path(path)
        self.embed = embed
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self.write_lock = threading.Lock()
        self.cache = QueryCache(cache_size)
        self._check_model(path / f"{name}.embedding.json", embed_model)

    @staticmethod
    def _check_model(meta_file: Path, embed_model: str) -> None:
        # vectors from two models can share a size and still mean nothing to each other, so record which built it
        if meta_file.exists():
            built_with = json.loads(meta_file.read_text())["embed_model"]
            if built_with != embed_model:
                raise EmbeddingMismatch(f"index was built with '{built_with}' but this is '{embed_model}'; re-index, or switch back")
        else:
            meta_file.write_text(json.dumps({"embed_model": embed_model}))

    # --- writing

    def upsert(self, doc_id: str, text: str, metadata: dict | None = None) -> UpsertResult:
        """Bring one document up to date, embedding only chunks whose text is new.

        A chunk's id contains a hash of its text, so unchanged chunks are recognised without being compared
        or embedded again, and chunks that disappeared from the text are deleted.
        """
        if not text.strip():
            raise ValueError("document has no text")
        if len(text) > MAX_TEXT:
            raise ValueError(f"document is longer than {MAX_TEXT} characters")
        meta = {"doc_id": doc_id, **clean_metadata(metadata)}

        new: dict[str, str] = {}
        for chunk in chunk_paragraphs(text):
            new[f"{doc_id}#{hashlib.sha1(chunk.encode()).hexdigest()[:16]}"] = chunk  # repeated paragraphs collapse

        with self.write_lock:
            old = self.col.get(where={"doc_id": doc_id}, include=["metadatas"])
            old_meta = dict(zip(old["ids"], old["metadatas"]))
            to_add = [i for i in new if i not in old_meta]
            to_remove = [i for i in old_meta if i not in new]
            kept = [i for i in new if i in old_meta]
            meta_changed = [i for i in kept if old_meta[i] != meta]

            if to_add:
                texts = [new[i] for i in to_add]
                self.col.add(ids=to_add, embeddings=self.embed(texts, "document").astype(float).tolist(), documents=texts, metadatas=[meta] * len(to_add))
            if to_remove:
                self.col.delete(ids=to_remove)
            if meta_changed:
                self._rewrite_metadata(meta_changed, meta)

        if not old_meta:
            status = "added"
        elif to_add or to_remove or meta_changed:
            status = "updated"
        else:
            status = "unchanged"
        return UpsertResult(status, len(to_add), len(to_remove), len(kept))

    def _rewrite_metadata(self, ids: list[str], meta: dict) -> None:
        # Chroma merges metadata on update, so a key removed by the caller would linger. Deleting and re-adding
        # with the stored vectors gives exact metadata without paying for the embedding again.
        rows = self.col.get(ids=ids, include=["embeddings", "documents"])
        self.col.delete(ids=ids)
        self.col.add(ids=ids, embeddings=rows["embeddings"], documents=rows["documents"], metadatas=[meta] * len(ids))

    def delete(self, doc_id: str) -> bool:
        with self.write_lock:
            ids = self.col.get(where={"doc_id": doc_id})["ids"]
            if ids:
                self.col.delete(ids=ids)
        return bool(ids)

    def sync(self, docs: list[Doc]) -> SyncResult:
        """Make the index match exactly this set of documents, touching only what differs."""
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

    # --- reading

    def doc_ids(self) -> set[str]:
        return {m["doc_id"] for m in self.col.get(include=["metadatas"])["metadatas"]}

    def get(self, doc_id: str) -> dict | None:
        rows = self.col.get(where={"doc_id": doc_id}, include=["metadatas"])
        if not rows["ids"]:
            return None
        return {"id": doc_id, "chunks": len(rows["ids"]), "metadata": {k: v for k, v in rows["metadatas"][0].items() if k != "doc_id"}}

    def stats(self) -> dict:
        return {"chunks": self.col.count(), "documents": len(self.doc_ids())}

    def query_vector(self, query: str) -> np.ndarray:
        return self.cache.get_or_compute(query, lambda: self.embed([query], "query")[0])

    def search(self, query: str, k: int = 5, filter: dict | None = None) -> list[Result]:
        to_where(filter)  # a bad filter is the caller's mistake; don't spend an embedding call finding that out
        return self.search_vector(self.query_vector(query), k, filter)

    def search_vector(self, vector: np.ndarray, k: int = 5, filter: dict | None = None) -> list[Result]:
        where = to_where(filter)  # checked first, so a bad filter is an error whether or not anything is indexed
        total = self.col.count()
        if total == 0:
            return []
        # ask for more chunks than documents wanted, since several chunks of one document can rank together
        res = self.col.query(
            query_embeddings=[np.asarray(vector, dtype=float).tolist()],
            n_results=min(max(k * 4, 20), total),
            where=where,
        )
        best: dict[str, Result] = {}
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            doc_id = meta["doc_id"]
            if doc_id not in best:  # results arrive best first, so the first chunk seen is the document's best
                user_meta = {k2: v for k2, v in meta.items() if k2 != "doc_id"}
                best[doc_id] = Result(doc_id, round(1.0 - dist, 4), doc[:SNIPPET], cid, user_meta)
        return list(best.values())[:k]
