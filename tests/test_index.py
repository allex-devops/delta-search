import pytest
from conftest import CountingEmbed, document, paragraph

from searchsvc.chunking import chunk_paragraphs
from searchsvc.filters import FilterError, to_where
from searchsvc.index import Doc, EmbeddingMismatch, SearchIndex


# chunking

def test_each_long_enough_paragraph_is_its_own_chunk():
    assert len(chunk_paragraphs(document("volcano", "glacier", "harbour"))) == 3


def test_short_paragraphs_are_glued_to_the_next_one():
    text = "A heading\n\n" + paragraph("volcano")
    chunks = chunk_paragraphs(text)
    assert len(chunks) == 1 and chunks[0].startswith("A heading The volcano")


def test_a_short_last_paragraph_is_still_indexed():
    assert chunk_paragraphs(paragraph("volcano") + "\n\nThe end.")[-1].endswith("The end.")


def test_a_paragraph_longer_than_a_chunk_is_split():
    assert len(chunk_paragraphs("word " * 1000)) > 1
    assert all(len(c) <= 800 for c in chunk_paragraphs("word " * 1000))


def test_editing_one_paragraph_changes_only_that_chunk():
    before = chunk_paragraphs(document("volcano", "glacier", "harbour", "market"))
    after = chunk_paragraphs(document("volcano", "glacier", "harbour", "market").replace("harbour", "lighthouse"))
    assert [a == b for a, b in zip(before, after)] == [True, True, False, True]


# filters

def test_filters_become_chroma_where_clauses():
    assert to_where({"year": 2024}) == {"year": {"$eq": 2024}}
    assert to_where({"tag": ["a", "b"]}) == {"tag": {"$in": ["a", "b"]}}
    assert to_where({"year": {"gte": 2020, "lt": 2024}}) == {"$and": [{"year": {"$gte": 2020}}, {"year": {"$lt": 2024}}]}
    assert to_where(None) is None and to_where({}) is None


@pytest.mark.parametrize(
    "bad",
    [{"doc_id": "x"}, {"a b": 1}, {"year": {"between": 1}}, {"year": {"gte": "2020"}}, {"tag": []}, {"tag": {"in": "a"}}, {"x": {"eq": [1]}}, {"x": {}}, {"x": None}],
)
def test_bad_filters_are_rejected_instead_of_passed_to_the_database(bad):
    with pytest.raises(FilterError):
        to_where(bad)


# adding, searching

def test_a_new_document_is_found_by_meaning_and_returns_its_best_chunk(index):
    index.upsert("geo", document("volcano", "glacier"), {"year": 2020})
    index.upsert("food", document("noodle", "pepper"), {"year": 2021})
    hits = index.search("volcano", k=2)
    assert [h.doc_id for h in hits][0] == "geo"
    assert "volcano" in hits[0].snippet and hits[0].metadata == {"year": 2020}


def test_a_document_appears_once_even_when_several_of_its_chunks_match(index):
    index.upsert("geo", document("volcano", "volcano", "volcano2"))
    assert [h.doc_id for h in index.search("volcano", k=5)] == ["geo"]


def test_an_empty_index_returns_nothing_rather_than_failing(index):
    assert index.search("anything") == []


def test_documents_without_text_are_refused(index):
    with pytest.raises(ValueError, match="no text"):
        index.upsert("x", "   \n\n ")


def test_metadata_must_be_flat_and_not_use_reserved_names(index):
    with pytest.raises(ValueError, match="reserved"):
        index.upsert("x", paragraph("volcano"), {"doc_id": "hack"})
    with pytest.raises(ValueError, match="string, number or boolean"):
        index.upsert("x", paragraph("volcano"), {"tags": ["a"]})


# filtering

@pytest.fixture
def filled(index):
    for i, (topic, year, lang) in enumerate([("volcano", 2019, "en"), ("volcano", 2022, "en"), ("volcano", 2024, "fr")]):
        index.upsert(f"d{i}", document(topic, f"other{i}"), {"year": year, "lang": lang})
    return index


def test_filter_by_exact_value(filled):
    assert {h.doc_id for h in filled.search("volcano", k=10, filter={"lang": "fr"})} == {"d2"}


def test_filter_by_range_and_by_list_together(filled):
    assert {h.doc_id for h in filled.search("volcano", k=10, filter={"year": {"gte": 2020}, "lang": ["en"]})} == {"d1"}


def test_a_filter_that_matches_nothing_gives_no_results(filled):
    assert filled.search("volcano", filter={"year": {"gt": 3000}}) == []


# incremental updates

def test_writing_the_same_document_again_costs_nothing(index, embed):
    doc = document("volcano", "glacier", "harbour")
    index.upsert("d", doc)
    before = embed.texts_embedded
    r = index.upsert("d", doc)
    assert r.status == "unchanged" and embed.texts_embedded == before and r.chunks_kept == 3


def test_editing_one_paragraph_embeds_only_that_one(index, embed):
    index.upsert("d", document("volcano", "glacier", "harbour", "market"))
    before = embed.texts_embedded
    r = index.upsert("d", document("volcano", "glacier", "harbour", "market").replace("harbour", "lighthouse"))
    assert (r.status, r.chunks_embedded, r.chunks_removed, r.chunks_kept) == ("updated", 1, 1, 3)
    assert embed.texts_embedded == before + 1


def test_the_old_text_is_gone_and_the_new_text_is_findable(index):
    index.upsert("d", document("volcano", "glacier", "harbour"))
    index.upsert("d", document("volcano", "glacier", "harbour").replace("harbour", "lighthouse"))
    assert "lighthouse" in index.search("lighthouse", k=1)[0].snippet
    assert all("harbour" not in h.snippet for h in index.search("harbour", k=5))


def test_removing_a_paragraph_removes_its_chunk_without_embedding_anything(index, embed):
    index.upsert("d", document("volcano", "glacier", "harbour"))
    before = embed.texts_embedded
    r = index.upsert("d", document("volcano", "harbour"))
    assert (r.status, r.chunks_embedded, r.chunks_removed) == ("updated", 0, 1) and embed.texts_embedded == before


def test_changing_only_metadata_reuses_the_stored_vectors(index, embed):
    index.upsert("d", document("volcano", "glacier"), {"year": 2020, "draft": True})
    before = embed.texts_embedded
    r = index.upsert("d", document("volcano", "glacier"), {"year": 2021})  # draft dropped
    assert r.status == "updated" and embed.texts_embedded == before and r.chunks_embedded == 0
    assert index.search("volcano", filter={"year": 2021})[0].doc_id == "d"
    assert index.search("volcano", filter={"year": 2020}) == []
    assert index.search("volcano", filter={"draft": True}) == []  # the removed key really is gone
    assert index.get("d")["metadata"] == {"year": 2021}


def test_inserting_a_paragraph_at_the_start_still_only_embeds_the_new_one(index, embed):
    index.upsert("d", document("volcano", "glacier", "harbour"))
    r = index.upsert("d", document("aardvark", "volcano", "glacier", "harbour"))
    assert (r.chunks_embedded, r.chunks_kept) == (1, 3)  # a fixed-size window would have redone everything


def test_deleting_a_document(index):
    index.upsert("d", document("volcano"))
    assert index.delete("d") is True and index.delete("d") is False
    assert index.search("volcano") == [] and index.get("d") is None


def test_one_documents_edit_never_touches_another(index):
    index.upsert("a", document("volcano"))
    index.upsert("b", document("volcano"))  # the same text in two documents: different ids
    index.upsert("a", document("glacier"))
    # search always returns the top k, so look at what the hits actually say
    assert {h.doc_id for h in index.search("volcano", k=5) if "volcano" in h.snippet} == {"b"}
    assert index.stats() == {"chunks": 2, "documents": 2}


# sync

def test_sync_makes_the_index_match_exactly_and_reports_what_it_did(index, embed):
    index.sync([Doc("a", document("volcano", "glacier")), Doc("b", document("harbour")), Doc("c", document("market"))])
    before = embed.texts_embedded
    r = index.sync([Doc("a", document("volcano", "glacier")), Doc("b", document("lighthouse")), Doc("d", document("noodle"))])
    assert (r.added, r.updated, r.unchanged, r.removed) == (1, 1, 1, 1)  # d added, b changed, a same, c gone
    assert r.chunks_embedded == 2 and embed.texts_embedded == before + 2
    assert index.doc_ids() == {"a", "b", "d"}


# model guard and query cache

def test_the_index_refuses_a_different_embedding_model(tmp_path, embed):
    SearchIndex(tmp_path / "i", embed, "model-a")
    with pytest.raises(EmbeddingMismatch, match="built with 'model-a'"):
        SearchIndex(tmp_path / "i", embed, "model-b")


def test_repeated_queries_are_embedded_once(index, embed):
    index.upsert("d", document("volcano"))
    before = embed.calls
    for _ in range(5):
        index.search("volcano")
    assert embed.calls == before + 1 and (index.cache.hits, index.cache.misses) == (4, 1)


def test_the_query_cache_evicts_the_least_recently_used(tmp_path, embed):
    small = SearchIndex(tmp_path / "s", embed, "m", cache_size=2)
    small.upsert("d", document("volcano"))
    for q in ("one", "two", "one", "three"):  # "two" is the stalest when "three" arrives
        small.search(q)
    before = embed.calls
    small.search("one")
    small.search("two")
    assert embed.calls == before + 1  # "one" survived, "two" had to be embedded again


def test_the_index_survives_reopening(tmp_path, embed):
    SearchIndex(tmp_path / "i", embed, "m").upsert("d", document("volcano"))
    reopened = SearchIndex(tmp_path / "i", CountingEmbed(), "m")
    assert reopened.search("volcano")[0].doc_id == "d"


def test_a_bad_filter_is_an_error_even_when_nothing_is_indexed_yet(index):
    with pytest.raises(FilterError):
        index.search("anything", filter={"year": {"between": 1}})


def test_a_bad_filter_is_rejected_before_the_query_is_embedded(index, embed):
    with pytest.raises(FilterError):
        index.search("volcano", filter={"year": {"bad": 1}})
    assert embed.calls == 0
