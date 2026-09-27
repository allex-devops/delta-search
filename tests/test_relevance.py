import math

import pytest
from conftest import document

from searchsvc.relevance import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_how_many_relevant_documents_made_the_top_k():
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, 4) == 1.0
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_reciprocal_rank_is_one_over_the_first_hit():
    assert reciprocal_rank(["x", "y", "a"], {"a", "z"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_rewards_putting_relevant_documents_first():
    perfect = ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3)
    swapped = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
    assert perfect == pytest.approx(1.0)
    assert swapped == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert perfect > swapped > ndcg_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0


def test_ndcg_does_not_punish_a_short_relevant_list():
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == pytest.approx(1.0)


def test_evaluate_averages_over_queries():
    runs = {"q1": ["a", "x"], "q2": ["x", "b"]}
    m = evaluate(lambda q, k: runs[q][:k], [("q1", {"a"}), ("q2", {"b"})], k=2)
    assert (m.queries, m.recall, m.mrr) == (2, 1.0, 0.75)
    assert "recall@2 1.00" in m.report()


def test_evaluate_with_no_queries_is_all_zeros_not_an_error():
    assert evaluate(lambda q, k: [], []).recall == 0.0


def test_evaluate_a_real_index_end_to_end(index):
    topics = ["volcano", "glacier", "harbour", "market", "noodle", "lantern"]
    for t in topics:
        index.upsert(t, document(t))
    m = evaluate(lambda q, k: [h.doc_id for h in index.search(q, k)], [(t, {t}) for t in topics], k=3)
    assert m.recall == 1.0 and m.mrr == 1.0
