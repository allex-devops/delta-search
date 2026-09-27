"""Calls your embedding provider (LLM_BASE_URL, LLM_API_KEY, EMBED_MODEL) over 112 real arXiv papers:

    uv run --env-file .env pytest -m live -s

The first run downloads the papers into data/live-papers (about 10 minutes, pausing between requests as arXiv asks).
"""
import json
import os
import time
from pathlib import Path

import httpx
import pytest
from shared.docs import read_pages
from shared.embed import embed_texts
from shared.llm import LLM

from searchsvc.index import SearchIndex
from searchsvc.relevance import evaluate

pytestmark = pytest.mark.live
DATA = Path(__file__).parent / "data"
# questions written from single passages of the papers, each with the paper it came from
QUESTIONS = DATA / "questions.jsonl"
# the papers those questions came from, then 40 more as distractors
PAPERS = DATA / "papers.txt"
PDF_DIR = Path("data") / "live-papers"


def pdf_path(paper_id: str) -> Path:
    return PDF_DIR / (paper_id.replace("/", "_") + ".pdf")


def fetch(paper_ids: list[str]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": "delta-search/0.1"}, timeout=60, follow_redirects=True) as client:
        for pid in paper_ids:
            if pdf_path(pid).exists():
                continue
            resp = client.get(f"https://arxiv.org/pdf/{pid}")
            if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
                pdf_path(pid).write_bytes(resp.content)
            time.sleep(5)


def paper_text(paper_id: str) -> str:
    # pages are separated by blank lines so the paragraph chunker treats them as separate blocks
    return "\n\n".join(text for _, text in read_pages(pdf_path(paper_id)))


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    missing = [n for n in ("LLM_BASE_URL", "LLM_API_KEY", "EMBED_MODEL") if not os.environ.get(n)]
    if missing:
        pytest.skip(f"set {', '.join(missing)}")
    llm = LLM(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], embed_model=os.environ["EMBED_MODEL"], timeout=120)
    embed = lambda texts, kind: embed_texts(texts, kind=kind, llm=llm)
    index = SearchIndex(tmp_path_factory.mktemp("live") / "idx", embed, llm.embed_model)
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
    ids = PAPERS.read_text().split()
    fetch(ids)
    papers = [p for p in ids if pdf_path(p).exists()]
    rows = [r for r in rows if r["paper_id"] in papers]
    return index, rows, papers


def test_relevance_on_real_papers_and_incremental_updates(stack):
    index, rows, papers = stack
    started = time.perf_counter()
    for pid in papers:
        index.upsert(pid, paper_text(pid), {"kind": "paper"})
    built = time.perf_counter() - started
    stats = index.stats()
    print(f"\nindexed {stats['documents']} papers as {stats['chunks']} chunks in {built:.0f}s")

    m = evaluate(lambda q, k: [h.doc_id for h in index.search(q, k)], [(r["question"], {r["paper_id"]}) for r in rows], k=5)
    print(m.report())
    assert m.recall >= 0.5

    # editing one real paper by adding a paragraph should embed one chunk, not the whole paper
    pid = papers[0]
    r = index.upsert(pid, paper_text(pid) + "\n\n" + "A late addition about retrieval evaluation. " * 12, {"kind": "paper"})
    print(f"appended one paragraph to {pid}: {r.status}, embedded {r.chunks_embedded}, kept {r.chunks_kept}")
    same = index.upsert(pid, paper_text(pid) + "\n\n" + "A late addition about retrieval evaluation. " * 12, {"kind": "paper"})
    assert r.chunks_embedded == 1 and same.status == "unchanged" and same.chunks_embedded == 0

    # filters on real data
    only = index.search("retrieval augmented generation", k=5, filter={"kind": "paper"})
    assert len(only) == 5 and index.search("retrieval", filter={"kind": "video"}) == []


def test_query_embedding_is_the_slow_part_and_the_cache_removes_it(stack):
    index, _, _ = stack
    index.cache.items.clear()
    queries = [f"how do systems handle {t}" for t in ("reranking", "chunking", "hallucination", "latency", "evaluation")]
    cold = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q)
        cold.append((time.perf_counter() - t0) * 1000)
    warm = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q)
        warm.append((time.perf_counter() - t0) * 1000)
    print(f"\nsearch with a fresh query: {sorted(cold)[len(cold) // 2]:.1f} ms median; the same query again: {sorted(warm)[len(warm) // 2]:.1f} ms median")
    assert sorted(warm)[len(warm) // 2] < sorted(cold)[len(cold) // 2]
