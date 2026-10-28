"""Adapters so ragas uses our gemini client and our embedding model instead of OpenAI."""

import asyncio
from typing import Any

from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig

from rag.base.embedder import Embedder
from rag.base.llm import LLMClient


class GeminiRagasLLM(BaseRagasLLM):
    """Goes through our LLMClient, so ragas also uses the throttle and the fallback models."""

    def __init__(self, client: LLMClient) -> None:
        super().__init__()
        self.client = client

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        # the temperature asked by ragas is ignored, we keep gemini's default (see gemini.py)
        texts = [self.client.generate(prompt.to_string()) for _ in range(n)]
        return LLMResult(generations=[[Generation(text=text) for text in texts]])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class EmbedderRagasEmbeddings(BaseRagasEmbeddings):
    """Our embedder for ragas. Questions also use embed_documents (no bge instruction)
    because ragas compares texts of the same kind, like a question with a question."""

    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        self.embedder = embedder
        self.run_config = RunConfig()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
