import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Path as PathParam
from pydantic import BaseModel, Field
from shared.llm import LLMError

from .filters import FilterError
from .index import Doc, EmbeddingMismatch, SearchIndex

log = logging.getLogger("searchsvc")
DOC_ID = r"^[A-Za-z0-9._:-]{1,200}$"
Scalar = str | int | float | bool


class DocIn(BaseModel):
    text: str = Field(min_length=1, max_length=500_000)
    metadata: dict[str, Scalar] = Field(default_factory=dict)


class SyncDoc(DocIn):
    id: str = Field(pattern=DOC_ID)


class SyncIn(BaseModel):
    documents: list[SyncDoc] = Field(max_length=5000)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=50)
    filter: dict | None = None


def create_app(index: SearchIndex) -> FastAPI:
    app = FastAPI(title="delta-search")

    def guarded(fn):
        """Run an index call and turn its failures into the right HTTP status."""
        try:
            return fn()
        except (FilterError, ValueError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except EmbeddingMismatch as e:
            raise HTTPException(status_code=409, detail=str(e))
        except LLMError as e:
            log.warning("embedding service failed: %s", e)  # detail may hold upstream text, keep it out of the response
            raise HTTPException(status_code=502, detail="the embedding service is unavailable")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stats")
    def stats():
        return {**index.stats(), "query_cache": {"hits": index.cache.hits, "misses": index.cache.misses}}

    @app.put("/documents/{doc_id}")
    def put_document(body: DocIn, doc_id: str = PathParam(pattern=DOC_ID)):
        r = guarded(lambda: index.upsert(doc_id, body.text, body.metadata))
        return {"id": doc_id, "status": r.status, "chunks_embedded": r.chunks_embedded, "chunks_removed": r.chunks_removed, "chunks_kept": r.chunks_kept}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str = PathParam(pattern=DOC_ID)):
        found = index.get(doc_id)
        if not found:
            raise HTTPException(status_code=404, detail="no such document")
        return found

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str = PathParam(pattern=DOC_ID)):
        if not index.delete(doc_id):
            raise HTTPException(status_code=404, detail="no such document")
        return {"deleted": doc_id}

    @app.post("/sync")
    def sync(body: SyncIn):
        ids = [d.id for d in body.documents]
        if len(set(ids)) != len(ids):
            raise HTTPException(status_code=422, detail="document ids must be unique")
        r = guarded(lambda: index.sync([Doc(d.id, d.text, d.metadata) for d in body.documents]))
        return vars(r)

    @app.post("/search")
    def search(body: SearchIn):
        results = guarded(lambda: index.search(body.query, body.k, body.filter))
        return {"results": [vars(r) for r in results]}

    return app


REQUIRED_ENV = ("LLM_BASE_URL", "LLM_API_KEY", "EMBED_MODEL")


def app_from_env() -> FastAPI:
    from shared.embed import embed_texts
    from shared.llm import LLM

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} in your environment or .env file (see .env.example)")
    llm = LLM(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], embed_model=os.environ["EMBED_MODEL"], timeout=60)
    embed = lambda texts, kind: embed_texts(texts, kind=kind, llm=llm)
    return create_app(SearchIndex(Path(os.environ.get("SEARCHSVC_DATA", "data")), embed, llm.embed_model))
