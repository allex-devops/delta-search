"""Needs Postgres from docker-compose: docker compose up -d, then uv run pytest -m pg"""
import os

import pytest
from conftest import CountingEmbed, document

from searchsvc.filters import FilterError
from searchsvc.index import EmbeddingMismatch
from searchsvc.pg_index import PgSearchIndex, _pg_filter

pytestmark = pytest.mark.pg
DSN = os.environ.get("SEARCHSVC_TEST_DSN", "postgresql://postgres:searchsvc@localhost:55432/searchsvc")


@pytest.fixture
def pg_index():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("drop table if exists chunks, index_meta")
    return PgSearchIndex(DSN, CountingEmbed(), "fake-model", dim=256)


def test_pg_filter_translation():
    where, params = _pg_filter({"year": 2024})
    assert where == " and metadata->>'year' = %s" and params == ["2024"]
    where, params = _pg_filter({"year": {"gte": 2020}, "lang": ["en", "fr"]})
    assert "gte" not in where and params == [2020, ["en", "fr"]]
    with pytest.raises(FilterError):
        _pg_filter({"a b": 1})
    with pytest.raises(FilterError):
        _pg_filter({"year": {"in": []}})


def test_add_search_update_delete_round_trip(pg_index):
    r = pg_index.upsert("geo", document("volcano", "glacier"), {"year": 2020})
    assert r.status == "added" and pg_index.stats() == {"chunks": 2, "documents": 1}
    hits = pg_index.search("volcano", k=5)
    assert hits[0].doc_id == "geo" and hits[0].metadata == {"year": 2020}

    unchanged = pg_index.upsert("geo", document("volcano", "glacier"), {"year": 2020})
    assert unchanged.status == "unchanged"

    updated = pg_index.upsert("geo", document("volcano", "harbour"), {"year": 2020})
    assert (updated.status, updated.chunks_embedded, updated.chunks_removed) == ("updated", 1, 1)
    assert all("glacier" not in h.snippet for h in pg_index.search("glacier", k=5))

    assert pg_index.delete("geo") is True and pg_index.delete("geo") is False
    assert pg_index.search("volcano") == []


def test_metadata_filters_over_a_real_database(pg_index):
    pg_index.upsert("old", document("volcano"), {"year": 2015, "lang": "en"})
    pg_index.upsert("new", document("volcano"), {"year": 2024, "lang": "fr"})
    assert {h.doc_id for h in pg_index.search("volcano", filter={"year": {"gte": 2020}})} == {"new"}
    assert {h.doc_id for h in pg_index.search("volcano", filter={"lang": ["en"]})} == {"old"}


def test_a_document_appears_once_even_with_several_matching_chunks(pg_index):
    pg_index.upsert("geo", document("volcano", "volcano2", "volcano3"))
    assert [h.doc_id for h in pg_index.search("volcano", k=5)] == ["geo"]


def test_sync_matches_the_document_set(pg_index):
    from searchsvc.index import Doc

    pg_index.sync([Doc("a", document("volcano")), Doc("b", document("glacier"))])
    r = pg_index.sync([Doc("a", document("volcano")), Doc("c", document("harbour"))])
    assert (r.added, r.unchanged, r.removed) == (1, 1, 1)
    assert pg_index.doc_ids() == {"a", "c"}


def test_the_index_refuses_a_different_embedding_model():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("drop table if exists chunks, index_meta")
    PgSearchIndex(DSN, CountingEmbed(), "model-a", dim=256)
    with pytest.raises(EmbeddingMismatch, match="built with 'model-a'"):
        PgSearchIndex(DSN, CountingEmbed(), "model-b", dim=256)
