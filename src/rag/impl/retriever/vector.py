"""Vector search over the rulebook chunks, stored in ChromaDB.

ChromaDB is saved on disk (settings.chroma_dir, not in git) so we only compute the
embeddings once with `make index`, the chat just loads it. We compute the embeddings
ourselves with our Embedder and give them to chroma, instead of letting chroma use its
default model, so we know exactly which model is used for the chunks and for the questions.
The collection uses cosine distance, chroma uses an HNSW index for the search.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.base.chunk import Chunk
from rag.base.embedder import Embedder
from rag.base.retriever import BaseRetriever
from rag.settings import settings

COLLECTION_NAME = "rulebook"
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def embedding_text(chunk: Chunk) -> str:
    """What we actually embed: the section path + the text. A chunk that only says
    "Roll six D6..." still gets found for "grenades" because its path is STRATAGEMS > GRENADE"""
    return f"{chunk.section_path}\n\n{chunk.text}"


class VectorRetriever(BaseRetriever):
    def __init__(self, embedder: Embedder, persist_dir: Path | None = None) -> None:
        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(persist_dir or settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata=COLLECTION_METADATA, embedding_function=None
        )

    def index(self, chunks: list[Chunk]) -> None:
        """Rebuilds the collection from scratch with these chunks."""
        self._client.delete_collection(COLLECTION_NAME)
        self.collection = self._client.create_collection(
            COLLECTION_NAME, metadata=COLLECTION_METADATA, embedding_function=None
        )
        embeddings = self.embedder.embed_documents([embedding_text(c) for c in chunks])
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            # chroma doesnt accept None in metadata, 0 means "unknown page"
            metadatas=[{"section_path": c.section_path, "page": c.page or 0} for c in chunks],
        )

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self.collection.count() == 0:
            raise RuntimeError("The vector index is empty, run `make index` first")
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
        return [
            Chunk(
                id=chunk_id,
                text=text,
                section_path=meta["section_path"],
                page=meta["page"],
                score=1 - distance,  # cosine distance -> similarity
            )
            for chunk_id, text, meta, distance in rows
        ]
