"""How does the index behave as it grows and as more clients hit it? Synthetic vectors, so no model is needed.

    python -m searchsvc bench --sizes 10000 50000 200000
"""
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import httpx
import numpy as np

from .index import SearchIndex

DIM = 768  # a common embedding size; the benchmark never calls a model


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def synthetic_vectors(n: int, dim: int = DIM, seed: int = 0, clusters: int = 50) -> np.ndarray:
    """Vectors that clump around a few centres, the way real embeddings do, unlike uniform random ones."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim))
    which = rng.integers(0, clusters, size=n)
    return unit(centres[which] + 0.6 * rng.normal(size=(n, dim)))


def text_vector(text: str, dim: int = DIM) -> np.ndarray:
    """A stand-in for a query embedding: the same text always gives the same vector, and it costs nothing."""
    return unit(np.random.default_rng(zlib.crc32(text.encode())).normal(size=dim))


def fake_embed(texts: list[str], kind: str = "query") -> np.ndarray:
    return np.vstack([text_vector(t) for t in texts])


def percentiles(ms: list[float]) -> dict:
    ordered = sorted(ms)
    at = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"p50": round(at(0.50), 2), "p95": round(at(0.95), 2), "p99": round(at(0.99), 2)}


def build_synthetic(path: Path, n: int, dim: int = DIM, batch: int = 4000) -> tuple[SearchIndex, np.ndarray, float]:
    """Load n vectors straight into the index, skipping chunking and embedding, and return them for checking."""
    index = SearchIndex(path, fake_embed, "synthetic")
    vectors = synthetic_vectors(n, dim)
    started = time.perf_counter()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        ids = [f"d{i}#0" for i in range(start, end)]
        index.col.add(
            ids=ids,
            embeddings=vectors[start:end].tolist(),
            documents=[f"synthetic document {i}" for i in range(start, end)],
            metadatas=[{"doc_id": f"d{i}", "group": i % 20} for i in range(start, end)],
        )
    return index, vectors, time.perf_counter() - started


def exact_top(vectors: np.ndarray, queries: np.ndarray, k: int) -> list[set[str]]:
    """The true nearest neighbours by brute force, to see how many an approximate index misses."""
    out = []
    for q in queries:
        top = np.argpartition(-(vectors @ q), k)[:k]
        out.append({f"d{i}" for i in top})
    return out


def single_thread(index: SearchIndex, vectors: np.ndarray, n_queries: int = 200, k: int = 10) -> dict:
    rng = np.random.default_rng(1)
    picks = rng.integers(0, len(vectors), size=n_queries)
    queries = unit(vectors[picks] + 0.3 * rng.normal(size=(n_queries, vectors.shape[1])))  # near real data
    times, found = [], []
    for q in queries:
        t0 = time.perf_counter()
        hits = index.search_vector(q, k=k)
        times.append((time.perf_counter() - t0) * 1000)
        found.append({h.doc_id for h in hits})
    truth = exact_top(vectors, queries, k)
    recall = statistics.mean(len(f & t) / k for f, t in zip(found, truth))
    return {**percentiles(times), "recall_at_10": round(recall, 3)}


async def http_load(base_url: str, concurrency: int, seconds: float, client: httpx.AsyncClient | None = None) -> dict:
    """`concurrency` clients each send searches back to back for `seconds`."""
    own = client is None
    client = client or httpx.AsyncClient(base_url=base_url, timeout=30, limits=httpx.Limits(max_connections=concurrency + 4))
    latencies: list[float] = []
    errors = 0
    deadline = time.perf_counter() + seconds

    async def worker(wid: int):
        nonlocal errors
        i = 0
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            try:
                r = await client.post("/search", json={"query": f"query {wid}-{i}", "k": 10})
                errors += r.status_code != 200
            except httpx.HTTPError:
                errors += 1
            latencies.append((time.perf_counter() - t0) * 1000)
            i += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    elapsed = time.perf_counter() - started
    if own:
        await client.aclose()
    return {"concurrency": concurrency, "requests": len(latencies), "qps": round(len(latencies) / elapsed, 1), "errors": errors, **percentiles(latencies)}


def serve(path: Path, port: int) -> None:
    """What the child process runs: the real API on top of a prebuilt index, with free query embeddings."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(SearchIndex(path, fake_embed, "synthetic")), host="127.0.0.1", port=port, log_level="warning")


def load_test_server(path: Path, levels: list[int], seconds: float, port: int = 8765) -> list[dict]:
    child = subprocess.Popen([sys.executable, "-m", "searchsvc", "serve-bench", str(path), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise RuntimeError("the benchmark server did not start")
        return [asyncio.run(http_load(f"http://127.0.0.1:{port}", c, seconds)) for c in levels]
    finally:
        child.terminate()  # the server must never outlive the benchmark
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


def run(sizes: list[int], seconds: float = 8.0, concurrency: list[int] | None = None) -> dict:
    out: dict = {"sizes": {}}
    workdir = Path(tempfile.mkdtemp(prefix="searchsvc-bench-"))
    for n in sizes:
        path = workdir / str(n)
        path.mkdir()
        index, vectors, build_s = build_synthetic(path, n)
        stats = single_thread(index, vectors)
        disk_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        out["sizes"][n] = {"build_seconds": round(build_s, 1), "vectors_per_second": round(n / build_s), "disk_mb": round(disk_mb), **stats}
        print(n, out["sizes"][n], flush=True)
        if n == max(sizes):
            del index  # release the files before another process opens them
            out["http"] = load_test_server(path, concurrency or [1, 8, 32], seconds)
            for row in out["http"]:
                print(row, flush=True)
    return out


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="searchsvc bench")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 200_000])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=Path("data/bench.json"))
    args = ap.parse_args(argv)
    result = run(args.sizes, args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print("wrote", args.out)
