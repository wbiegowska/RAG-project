from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors for the vector search."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per chunk of the rulebook."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Vector for a user question (some models embed questions a bit differently)."""
