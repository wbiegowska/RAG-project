from abc import ABC, abstractmethod

from rag.base.chunk import Chunk


class Reranker(ABC):
    """Re-orders a short list of candidate chunks, best first."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        """Returns the k best chunks for the query, with their new score."""
