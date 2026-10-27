from abc import ABC, abstractmethod

from rag.base.chunk import Chunk


class BaseRetriever(ABC):
    """Anything that turns a question into a ranked list of rulebook chunks."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the k most relevant chunks, best first."""
