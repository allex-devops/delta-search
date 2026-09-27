import asyncio

import httpx
import numpy as np

from searchsvc.bench import build_synthetic, exact_top, http_load, percentiles, single_thread, synthetic_vectors, text_vector


def test_synthetic_vectors_are_unit_length_and_reproducible():
    a, b = synthetic_vectors(200, dim=32, seed=3), synthetic_vectors(200, dim=32, seed=3)
    assert np.array_equal(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert not np.array_equal(a, synthetic_vectors(200, dim=32, seed=4))


def test_the_vectors_really_are_clumped_not_uniform():
    v = synthetic_vectors(2000, dim=32, clusters=5)
    uniform = np.random.default_rng(0).normal(size=(2000, 32))
    uniform /= np.linalg.norm(uniform, axis=1, keepdims=True)
    nearest = lambda x: np.sort(x @ x[0])[-2]  # similarity of the closest other vector to the first one
    assert nearest(v) > nearest(uniform)


def test_a_query_text_always_gives_the_same_vector():
    assert np.array_equal(text_vector("hello", 16), text_vector("hello", 16))
    assert not np.array_equal(text_vector("hello", 16), text_vector("hellp", 16))


def test_exact_search_finds_the_true_neighbours():
    v = synthetic_vectors(300, dim=16)
    assert exact_top(v, v[7:8], 1) == [{"d7"}]  # a vector's nearest neighbour is itself


def test_percentiles():
    assert percentiles([float(i) for i in range(1, 101)]) == {"p50": 51.0, "p95": 96.0, "p99": 100.0}


def test_a_small_synthetic_index_answers_queries_with_high_recall(tmp_path):
    index, vectors, seconds = build_synthetic(tmp_path, 600, dim=32, batch=250)
    assert index.col.count() == 600 and seconds > 0
    stats = single_thread(index, vectors, n_queries=30, k=10)
    assert stats["recall_at_10"] >= 0.8 and stats["p50"] > 0


def test_the_load_generator_counts_requests_errors_and_latency():
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        return httpx.Response(200 if hits["n"] % 5 else 500, json={"results": []})  # every fifth request fails

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x") as client:
            return await http_load("http://x", concurrency=4, seconds=0.3, client=client)

    r = asyncio.run(run())
    assert r["requests"] == hits["n"] > 10 and r["concurrency"] == 4
    assert 0 < r["errors"] < r["requests"] and r["qps"] > 0
