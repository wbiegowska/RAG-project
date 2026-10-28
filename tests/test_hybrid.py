"""Tests for the reciprocal rank fusion, the hybrid retriever and the reranker.
Everything is faked (no model, no index) so the tests are fast."""

import pytest

from rag.base.chunk import Chunk
from rag.base.retriever import BaseRetriever
from rag.impl.retriever.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag.impl.retriever.reranker import CrossEncoderReranker


def chunk(chunk_id: str, score: float = 0.0) -> Chunk:
    return Chunk(id=chunk_id, text=f"text {chunk_id}", section_path=chunk_id.upper(), score=score)


class FixedRetriever(BaseRetriever):
    """Always returns the same ranking, and remembers how many chunks were asked."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.asked_k: int | None = None

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        self.asked_k = k
        return [chunk(i, score=1.0) for i in self.ids[:k]]


class FakeCrossEncoder:
    """Gives a higher score to chunks whose id is later in the alphabet."""

    def predict(self, pairs):
        return [float(ord(text.split()[-1])) for _query, text in pairs]


def test_rrf_puts_chunks_found_by_both_lists_first():
    merged = reciprocal_rank_fusion([[chunk("a"), chunk("b")], [chunk("c"), chunk("b")]])
    assert [c.id for c in merged] == ["b", "a", "c"]
    # b is 2nd in both lists: 1/(60+2) + 1/(60+2)
    assert merged[0].score == pytest.approx(2 / 62)


def test_vector_only_keeps_the_vector_order_and_scores():
    vector = FixedRetriever(["a", "b", "c"])
    results = HybridRetriever(vector, candidates=20, rrf_k=60).retrieve("q", k=2)
    assert [c.id for c in results] == ["a", "b"]
    assert results[0].score == 1.0
    assert vector.asked_k == 20  # it asks for all the candidates, not just k


def test_hybrid_merges_vector_and_bm25():
    vector = FixedRetriever(["a", "b", "c"])
    bm25 = FixedRetriever(["b", "d", "e"])
    results = HybridRetriever(vector, bm25=bm25, candidates=20, rrf_k=60).retrieve("q", k=3)
    # b is found by both so it goes first, then the 1st of vector, then the 2nd of BM25
    assert [c.id for c in results] == ["b", "a", "d"]


def test_reranker_has_the_last_word():
    reranker = CrossEncoderReranker(model_name="fake")
    reranker.model = FakeCrossEncoder()  # replaces the real model, nothing is downloaded
    vector = FixedRetriever(["a", "b", "c"])
    results = HybridRetriever(vector, reranker=reranker, candidates=20, rrf_k=60).retrieve("q", k=2)
    assert [c.id for c in results] == ["c", "b"]
    assert results[0].score == float(ord("c"))


def test_reranker_with_no_chunks():
    assert CrossEncoderReranker(model_name="fake").rerank("q", [], k=5) == []
