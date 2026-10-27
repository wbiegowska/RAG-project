"""Embeddings with a sentence-transformers (SBERT) model that runs locally.

We use BAAI/bge-small-en-v1.5 (setting `embedding_model`). It's small (33M parameters,
about 130MB) so it runs fine on a laptop CPU, it does well on the retrieval part of the
MTEB leaderboard for its size, and since it runs locally there are no API calls and no
rate limits (Gemini embeddings would eat the same free quota as the chat).
The model is downloaded from HuggingFace the first time it's used.

The BGE models were trained with a short instruction in front of the questions (not in
front of the documents) for retrieval, so we add it in embed_query.
"""

from functools import cached_property

from sentence_transformers import SentenceTransformer

from rag.base.embedder import Embedder
from rag.settings import settings

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SbertEmbedder(Embedder):
    def __init__(
        self, model_name: str | None = None, query_instruction: str = BGE_QUERY_INSTRUCTION
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.query_instruction = query_instruction

    @cached_property
    def model(self) -> SentenceTransformer:
        # only loaded when we need it, it takes a few seconds
        return SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalized vectors, so cosine similarity is just a dot product
        vectors = self.model.encode_document(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode_query(
            text, prompt=self.query_instruction, normalize_embeddings=True
        )
        return vector.tolist()
