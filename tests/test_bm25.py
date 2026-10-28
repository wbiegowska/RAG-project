"""Tests for the BM25 keyword search."""

import pytest

from rag.base.chunk import Chunk
from rag.impl.retriever.bm25 import BM25Retriever, tokenize

CHUNKS = [
    Chunk(id="a", text="Roll six D6, each 4+ is a mortal wound.", section_path="GRENADE", page=42),
    Chunk(
        id="b", text="Units can move up to their Move characteristic.", section_path="MOVE", page=13
    ),
    Chunk(id="c", text="Scores 1 Victory point per objective.", section_path="SCORING", page=59),
]


def test_tokenize_removes_stopwords_and_stems():
    # "the" and "for" are stopwords, "grenades" and "scoring" lose their ending
    assert tokenize("The rule for Grenades scoring") == ["rule", "grenad", "score"]


def test_finds_chunks_with_the_same_words():
    results = BM25Retriever(CHUNKS).retrieve("What is the rule for grenades?", k=2)
    assert results[0].id == "a"


def test_stemming_matches_different_forms_of_a_word():
    # the chunk says "Scores", the question says "score"
    results = BM25Retriever(CHUNKS).retrieve("How do I score points?", k=1)
    assert results[0].id == "c"


def test_chunks_without_any_question_word_are_not_returned():
    results = BM25Retriever(CHUNKS).retrieve("grenades", k=3)
    assert [c.id for c in results] == ["a"]


def test_no_chunks_gives_a_clear_error():
    with pytest.raises(RuntimeError, match="make index"):
        BM25Retriever([])
