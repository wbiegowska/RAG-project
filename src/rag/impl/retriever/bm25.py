"""BM25 keyword search (rank-bm25). Words are stemmed so "scores", "scoring" and "score"
all match."""

import re
from dataclasses import replace

import snowballstemmer
from rank_bm25 import BM25Okapi

from rag.base.chunk import Chunk
from rag.base.retriever import BaseRetriever
from rag.impl.retriever.vector import embedding_text

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its of on or "
    "that the their them then there these they this to was were what when where which who "
    "will with you your".split()
)
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return _stemmer.stemWords(words)


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise RuntimeError("No chunks to build BM25 from, run `make index` first")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(embedding_text(c)) for c in chunks])

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        scores = self._bm25.get_scores(tokenize(query))
        best = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
        # a score of 0 means none of the question words is in the chunk, so its not a match
        return [replace(self.chunks[i], score=float(scores[i])) for i in best if scores[i] > 0]
