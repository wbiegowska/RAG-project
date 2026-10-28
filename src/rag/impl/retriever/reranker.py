"""Cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L6-v2 by default), used on the
candidates of the hybrid search. Off by default because it didnt improve our test
questions, see docs/technical_choices.md."""

from dataclasses import replace
from functools import cached_property
from typing import Any

from rag.base.chunk import Chunk
from rag.base.reranker import Reranker
from rag.impl.retriever.vector import embedding_text
from rag.settings import settings


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model

    @cached_property
    def model(self) -> Any:
        # imported here so the tests can use a fake model without loading torch
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        if not chunks:
            return []
        scores = self.model.predict([(query, embedding_text(c)) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:k]]
