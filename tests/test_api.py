import pytest
from conftest import document
from fastapi.testclient import TestClient
from shared.llm import LLMError

from searchsvc.api import create_app
from searchsvc.index import SearchIndex


@pytest.fixture
def client(index):
    return TestClient(create_app(index))


def put(client, doc_id, text, **metadata):
    return client.put(f"/documents/{doc_id}", json={"text": text, "metadata": metadata})


def test_health_and_stats(client):
    assert client.get("/health").json() == {"status": "ok"}
    put(client, "a", document("volcano", "glacier"))
    s = client.get("/stats").json()
    assert (s["documents"], s["chunks"]) == (1, 2)


def test_put_reports_what_it_did_and_is_idempotent(client):
    first = put(client, "a", document("volcano", "glacier")).json()
    assert (first["status"], first["chunks_embedded"]) == ("added", 2)
    again = put(client, "a", document("volcano", "glacier")).json()
    assert (again["status"], again["chunks_embedded"], again["chunks_kept"]) == ("unchanged", 0, 2)
    edited = put(client, "a", document("volcano", "harbour")).json()
    assert (edited["status"], edited["chunks_embedded"], edited["chunks_removed"]) == ("updated", 1, 1)


def test_search_returns_scored_snippets_with_metadata(client):
    put(client, "geo", document("volcano"), year=2020)
    put(client, "food", document("noodle"), year=2021)
    r = client.post("/search", json={"query": "volcano", "k": 2}).json()["results"]
    assert r[0]["doc_id"] == "geo" and r[0]["metadata"] == {"year": 2020} and "volcano" in r[0]["snippet"]
    assert r[0]["score"] > r[1]["score"]


def test_search_filters(client):
    put(client, "old", document("volcano"), year=2015)
    put(client, "new", document("volcano"), year=2024)
    r = client.post("/search", json={"query": "volcano", "filter": {"year": {"gte": 2020}}}).json()["results"]
    assert [x["doc_id"] for x in r] == ["new"]


def test_get_and_delete_a_document(client):
    put(client, "a", document("volcano"), year=1)
    assert client.get("/documents/a").json() == {"id": "a", "chunks": 1, "metadata": {"year": 1}}
    assert client.delete("/documents/a").json() == {"deleted": "a"}
    assert client.get("/documents/a").status_code == 404 and client.delete("/documents/a").status_code == 404


def test_sync_reports_counts(client):
    put(client, "gone", document("market"))
    docs = [{"id": "a", "text": document("volcano")}, {"id": "b", "text": document("glacier"), "metadata": {"year": 1}}]
    r = client.post("/sync", json={"documents": docs}).json()
    assert (r["added"], r["removed"]) == (2, 1)
    assert client.post("/sync", json={"documents": docs}).json()["unchanged"] == 2


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.put("/documents/a", json={"text": "", "metadata": {}}),
        lambda c: c.put("/documents/a", json={"text": "x" * 500_001}),
        lambda c: c.put("/documents/a", json={"text": "hello world", "metadata": {"tags": ["a"]}}),
        lambda c: c.put("/documents/a", json={"text": "hello", "metadata": {"doc_id": "x"}}),
        lambda c: c.put("/documents/has%20space", json={"text": "hello"}),
        lambda c: c.post("/search", json={"query": "", "k": 5}),
        lambda c: c.post("/search", json={"query": "x", "k": 0}),
        lambda c: c.post("/search", json={"query": "x", "k": 51}),
        lambda c: c.post("/search", json={"query": "x", "filter": {"doc_id": "a"}}),
        lambda c: c.post("/search", json={"query": "x", "filter": {"year": {"between": 1}}}),
        lambda c: c.post("/sync", json={"documents": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]}),
    ],
)
def test_bad_input_is_a_422_and_never_reaches_the_index(client, call):
    assert call(client).status_code == 422


def test_an_embedding_failure_is_a_502_with_nothing_leaked(tmp_path):
    def broken(texts, kind="document"):
        raise LLMError("upstream said the key sk-live-123 is bad")

    c = TestClient(create_app(SearchIndex(tmp_path / "b", broken, "m")))
    r = c.put("/documents/a", json={"text": document("volcano")})
    assert r.status_code == 502 and "sk-live-123" not in r.text


def test_the_stats_show_the_query_cache_working(client):
    put(client, "a", document("volcano"))
    for _ in range(3):
        client.post("/search", json={"query": "volcano"})
    assert client.get("/stats").json()["query_cache"] == {"hits": 2, "misses": 1}
