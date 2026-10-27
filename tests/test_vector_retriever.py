"""Tests for the ChromaDB vector retriever.

They use a fake embedder (word counts instead of a real model) so they dont need to
download anything, and a temporary folder for chroma so the real index isnt touched.
"""

import math
import zlib

import pytest

from rag.base.chunk import Chunk
from rag.base.embedder import Embedder
from rag.impl.retriever.vector import VectorRetriever, embedding_text

DIMENSIONS = 64


class FakeEmbedder(Embedder):
    """Each word adds 1 to one of 64 dimensions, so texts with the same words are close."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


CHUNKS = [
    Chunk(id="a", text="roll six dice each 4+ is a mortal wound", section_path="GRENADE", page=42),
    Chunk(
        id="b", text="units can move up to their move characteristic", section_path="MOVE", page=13
    ),
    Chunk(
        id="c", text="spend command points to use stratagems", section_path="STRATAGEMS", page=41
    ),
]


def test_finds_the_right_chunk_and_keeps_its_metadata(tmp_path):
    retriever = VectorRetriever(FakeEmbedder(), persist_dir=tmp_path)
    retriever.index(CHUNKS)
    results = retriever.retrieve("grenade mortal wound dice", k=2)
    assert len(results) == 2
    best = results[0]
    assert (best.id, best.section_path, best.page) == ("a", "GRENADE", 42)
    assert best.text == CHUNKS[0].text
    assert results[0].score >= results[1].score


def test_index_replaces_the_old_chunks(tmp_path):
    retriever = VectorRetriever(FakeEmbedder(), persist_dir=tmp_path)
    retriever.index(CHUNKS)
    retriever.index(CHUNKS[:1])
    assert retriever.collection.count() == 1


def test_empty_index_gives_a_clear_error(tmp_path):
    retriever = VectorRetriever(FakeEmbedder(), persist_dir=tmp_path)
    with pytest.raises(RuntimeError, match="make index"):
        retriever.retrieve("anything")


def test_section_path_is_embedded_with_the_text():
    assert embedding_text(CHUNKS[0]).startswith("GRENADE\n\n")
