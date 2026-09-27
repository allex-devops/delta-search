"""Does search put the right documents near the top? The usual measures, over a labelled set of queries."""
import math
from dataclasses import dataclass
from typing import Callable


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 means every relevant document is at the very top; documents count as relevant or not, nothing in between."""
    gain = sum(1 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], 1) if doc in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


@dataclass
class Metrics:
    queries: int
    k: int
    recall: float
    mrr: float
    ndcg: float

    def report(self) -> str:
        return f"{self.queries} queries: recall@{self.k} {self.recall:.2f}, MRR {self.mrr:.2f}, nDCG@{self.k} {self.ndcg:.2f}"


def evaluate(search: Callable[[str, int], list[str]], labelled: list[tuple[str, set[str]]], k: int = 5) -> Metrics:
    """`search(query, k)` returns document ids, best first. `labelled` is (query, ids that answer it)."""
    n = len(labelled)
    if n == 0:
        return Metrics(0, k, 0.0, 0.0, 0.0)
    runs = [(search(q, k), rel) for q, rel in labelled]
    return Metrics(
        n,
        k,
        sum(recall_at_k(r, rel, k) for r, rel in runs) / n,
        sum(reciprocal_rank(r, rel) for r, rel in runs) / n,
        sum(ndcg_at_k(r, rel, k) for r, rel in runs) / n,
    )
