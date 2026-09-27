# Delta Search

A semantic search API that only re-embeds what changed. Send it documents, search them by meaning with metadata filters, and when a document is edited, only the paragraphs that actually changed go back to the embedding model. It also measures its own relevance, benchmarks itself at scale and under load, and comes with a Postgres + pgvector backend for when one embedded index isn't enough.

## What it does

- **Incremental indexing.** Documents are chunked by paragraph and each chunk's id is a hash of its text, so editing one paragraph re-embeds just that paragraph. Unchanged chunks are recognised without being compared or embedded again, and removed ones are deleted.
- **Metadata filters.** `{"year": {"gte": 2023}, "lang": ["en", "fr"]}` style filters, validated before they reach the index.
- **Query cache.** Recent query embeddings are kept in memory, since embedding is the slow part of a search.
- **Relevance metrics.** recall@k, MRR and nDCG over a labelled set of queries.
- **Scale and load benchmark.** Synthetic vectors at growing sizes, plus real HTTP load against the running API. No model or key needed.
- **Postgres backend.** `PgSearchIndex` has the same upsert/search/sync shape as the default Chroma-backed index, backed by pgvector, so several API workers can write at once.
- **Safe errors.** Bad filters return 422, an index built with a different embedding model returns 409, and a failing embedding provider returns 502 without leaking its error text.

## Requirements

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- An API key for OpenAI or Google Gemini (any OpenAI-compatible endpoint works), used for embeddings. Your document text is sent to the provider you choose.
- Docker, only for the Postgres backend or the container

## Setup

```bash
uv sync
cp .env.example .env    # uncomment one provider block and paste your key
```

## Run

```bash
uv run --env-file .env uvicorn searchsvc.api:app_from_env --factory --port 8002

curl -X PUT localhost:8002/documents/paper1 -H 'content-type: application/json' \
  -d '{"text": "...", "metadata": {"year": 2024}}'
curl -X POST localhost:8002/search -H 'content-type: application/json' \
  -d '{"query": "reranking", "k": 5, "filter": {"year": {"gte": 2023}}}'

uv run python -m searchsvc bench --sizes 5000 50000 200000   # scale + load test, no key needed

uv run pytest                                   # offline, no key needed
uv run --env-file .env pytest -m live -s        # against your provider and 112 real arXiv papers
```

| Endpoint | What it does |
|---|---|
| `PUT /documents/{id}` | add or update one document; the reply says how many chunks were embedded, kept and removed |
| `GET /documents/{id}` | chunk count and metadata |
| `DELETE /documents/{id}` | remove a document |
| `POST /sync` | make the index match exactly this set of documents, adding, updating and removing as needed |
| `POST /search` | `query`, `k` and an optional `filter`; one result per document, best chunk first |
| `GET /stats` | chunk and document counts, query-cache hits and misses |
| `GET /health` | liveness |

**Postgres backend:**

```bash
docker compose up -d
uv run pytest -m pg          # same behaviour as the Chroma-backed index, over a real database
docker compose down
```

Pass `dim` to `PgSearchIndex` to match your embedding model (default 1536).

**Container:**

```bash
docker build -t delta-search .
docker run --env-file .env -p 8002:8002 delta-search
```

## Configuration

Set these in `.env`. The first three are required and have no defaults.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | the provider's OpenAI-compatible endpoint |
| `LLM_API_KEY` | your provider key |
| `EMBED_MODEL` | embedding model name |
| `SEARCHSVC_DATA` | where the index is kept (default `./data`) |

Changing `EMBED_MODEL` after indexing is refused rather than silently mixing vectors: delete the index folder and add the documents again.

## Results

Relevance and cache numbers were measured with a small open embedding model run locally; a hosted provider will differ, so run the live tests against yours.

| Check | Result |
|---|---|
| Offline tests | 71 pass, plus 6 Postgres tests run separately with `docker compose up` |
| Deliberately broken to confirm tests notice | skip-unchanged, delete-on-edit, metadata merge-vs-replace, per-document dedup, short-paragraph merging, filter validation on an empty index, Postgres delete-on-edit: each made a test fail |
| Docker image | builds and runs; the non-root user originally had no permission to create the data directory, found and fixed |
| Incremental indexing on real papers | adding a paragraph to one paper embedded exactly 1 chunk out of ~95; repeating the identical edit embedded 0 |
| Relevance on 112 real papers, 72 questions | recall@5 0.88, MRR 0.83, nDCG@5 0.84 |
| Query cache | a fresh query took 26 ms median, the same query again 1.7 ms |

**Scale**, synthetic vectors, single embedded Chroma process:

| Corpus size | Build rate | p50 / p95 search | recall@10 |
|---|---|---|---|
| 5,000 | 4,430/s | 1.5 / 1.9 ms | 0.96 |
| 50,000 | 3,400/s | 4.1 / 6.0 ms | 0.55 |
| 200,000 | 1,680/s | 14.3 / 23.5 ms | 0.25 |

Recall@10 (exact document-id overlap with brute-force nearest neighbours) drops as the corpus grows. It isn't a benchmark bug: raising Chroma's `hnsw:search_ef` to 200 changed nothing. Comparing actual scores shows two things. Most "misses" are near-ties (a 10th result scoring 0.110 against a true 10th of 0.112), which don't matter. A minority are real misses (one query's true top 10 scored 0.138–0.153 while Chroma returned 0.120–0.127). This points at Chroma's default HNSW build parameters, not search-time `ef`, being untuned for a growing corpus. Raising `hnsw:construction_ef` and `hnsw:M` would likely help and hasn't been tried.

**Concurrent load**, same 200,000-vector index, real HTTP requests to the API with synthetic query embeddings:

| Concurrency | Throughput | p50 / p95 / p99 |
|---|---|---|
| 1 | 52 req/s | 17 / 26 / 32 ms |
| 8 | 72 req/s | 106 / 172 / 220 ms |
| 32 | 72 req/s | 447 / 655 / 751 ms |

Throughput stops growing between 8 and 32 clients while latency keeps climbing: the single embedded Chroma process is the bottleneck, which is what the Postgres backend is for. The same load test against `PgSearchIndex` hasn't been run yet.

These are single-laptop numbers (Apple M3 Max), not a claim about any particular deployment.
