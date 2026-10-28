"""Hybrid search: vector + BM25 merged with reciprocal rank fusion, then optionally reranked
with the cross-encoder. BM25 and the reranker can be switched off to compare the setups."""

from dataclasses import replace

from rag.base.chunk import Chunk
from rag.base.reranker import Reranker
from rag.base.retriever import BaseRetriever
from rag.settings import settings


def reciprocal_rank_fusion(rankings: list[list[Chunk]], rrf_k: int = 60) -> list[Chunk]:
    """Each chunk gets 1 / (rrf_k + rank) from every list its in. Only the ranks are used
    because cosine and BM25 scores are not on the same scale."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (rrf_k + rank)
            chunks.setdefault(chunk.id, chunk)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [replace(chunks[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector: BaseRetriever,
        bm25: BaseRetriever | None = None,
        reranker: Reranker | None = None,
        candidates: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.vector = vector
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates or settings.retrieval_candidates
        self.rrf_k = rrf_k or settings.rrf_k

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        n = max(self.candidates, k)
        found = self.vector.retrieve(query, n)
        if self.bm25 is not None:
            found = reciprocal_rank_fusion([found, self.bm25.retrieve(query, n)], self.rrf_k)
        if self.reranker is not None:
            return self.reranker.rerank(query, found[:n], k)
        return found[:k]
