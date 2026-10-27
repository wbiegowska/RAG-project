from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    """Wrapper around the LLM backend (Gemini via Google AI Studio)."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the full answer as one string."""

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the answer piece by piece, for the chat interface."""
