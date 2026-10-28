"""Local SBERT embeddings (BAAI/bge-small-en-v1.5 by default, downloaded the first time)."""

from functools import cached_property

from sentence_transformers import SentenceTransformer

from rag.base.embedder import Embedder
from rag.settings import settings

# bge models expect this in front of the questions (not the chunks)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SbertEmbedder(Embedder):
    def __init__(
        self, model_name: str | None = None, query_instruction: str = BGE_QUERY_INSTRUCTION
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.query_instruction = query_instruction

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode_document(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode_query(
            text, prompt=self.query_instruction, normalize_embeddings=True
        )
        return vector.tolist()
