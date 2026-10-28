"""Builds the rulebook retriever from the settings. Use get_rulebook_retriever() everywhere,
the models and the index are only loaded once."""

from functools import lru_cache

from rag.impl.embedder.sbert import SbertEmbedder
from rag.impl.retriever.bm25 import BM25Retriever
from rag.impl.retriever.hybrid import HybridRetriever
from rag.impl.retriever.reranker import CrossEncoderReranker
from rag.impl.retriever.vector import VectorRetriever
from rag.settings import settings

# the setups we compare
SEARCH_MODES = {
    "vector": {"use_bm25": False, "use_reranker": False},
    "hybrid": {"use_bm25": True, "use_reranker": False},
    "full": {"use_bm25": True, "use_reranker": True},
}


@lru_cache
def get_rulebook_retriever(
    use_bm25: bool | None = None, use_reranker: bool | None = None
) -> HybridRetriever:
    """Defaults come from the settings, the arguments are only for comparing setups."""
    use_bm25 = settings.use_bm25 if use_bm25 is None else use_bm25
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker
    return HybridRetriever(
        _vector_retriever(),
        bm25=_bm25_retriever() if use_bm25 else None,
        reranker=_reranker() if use_reranker else None,
    )


# created once and shared by all the setups
@lru_cache
def _vector_retriever() -> VectorRetriever:
    return VectorRetriever(SbertEmbedder())


@lru_cache
def _bm25_retriever() -> BM25Retriever:
    return BM25Retriever(_vector_retriever().all_chunks())


@lru_cache
def _reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
