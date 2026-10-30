"""Tests for the evaluation runner and the stand-in pipeline (no gemini, no real model)."""

import pytest

pytest.importorskip("ragas")

from rag.base.chunk import Chunk  # noqa: E402
from rag.base.llm import LLMClient  # noqa: E402
from rag.base.retriever import BaseRetriever  # noqa: E402
from rag.evaluation.baseline import StandInPipeline  # noqa: E402
from rag.evaluation.runner import (  # noqa: E402
    answer_with_retries,
    evaluate_retrieval,
    hit_rate,
    reciprocal_rank,
    short_metric_names,
    summary_table,
)
from rag.impl.retriever.vector import embedding_text  # noqa: E402

GRENADE = Chunk(id="a", text="Roll six D6.", section_path="STRATAGEMS > GRENADE", page=42)
MOVE = Chunk(id="b", text="Move up to M inches.", section_path="MOVEMENT PHASE", page=13)
TESTSET = [
    {
        "user_input": "grenades?",
        "reference_contexts": [embedding_text(GRENADE)],
        "synthesizer_name": "single",
    },
    {
        "user_input": "moving?",
        "reference_contexts": [embedding_text(MOVE)],
        "synthesizer_name": "single",
    },
]


class FixedRetriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return self.chunks[:k]


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return "You roll six D6."

    def stream(self, prompt: str, system: str | None = None):
        yield self.generate(prompt, system)


def test_hit_rate_and_reciprocal_rank():
    assert hit_rate(["x", "ref"], ["ref"]) == 1.0
    assert hit_rate(["x", "y"], ["ref"]) == 0.0
    assert reciprocal_rank(["x", "ref"], ["ref"]) == 0.5
    assert reciprocal_rank(["x"], ["ref"]) == 0.0


def test_evaluate_retrieval_scores_each_setup():
    retrievers = {
        "good": FixedRetriever([GRENADE, MOVE]),  # grenade #1, move #2
        "bad": FixedRetriever([Chunk(id="c", text="nothing", section_path="OTHER")]),
    }
    scores = evaluate_retrieval(TESTSET, retrievers, k=2)
    good = scores[scores["setup"] == "good"]
    assert list(good["hit_rate"]) == [1.0, 1.0]
    assert list(good["mrr"]) == [1.0, 0.5]
    assert list(scores[scores["setup"] == "bad"]["context_recall"]) == [0.0, 0.0]


def test_stand_in_pipeline_gives_the_chunks_to_the_llm():
    llm = FakeLLM()
    result = StandInPipeline(FixedRetriever([GRENADE]), llm, k=1).run("grenades?", history=[])
    prompt, system = llm.calls[0]
    assert "[STRATAGEMS > GRENADE, page 42]" in prompt
    assert "Question: grenades?" in prompt
    assert "don't know" in system
    assert result.answer == "You roll six D6."
    assert result.rule_chunks == [GRENADE]


def test_summary_table():
    import pandas as pd

    scores = pd.DataFrame({"setup": ["a", "a", "b"], "mrr": [1.0, 0.5, 0.0]})
    table = summary_table(scores, "setup", ["mrr"])
    assert table.splitlines()[2] == "| a | 2 | 0.75 |"
    assert table.splitlines()[3] == "| b | 1 | 0.00 |"


class FlakyPipeline:
    """Fails a few times (like a 503) before answering."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def run(self, question, history):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("503 UNAVAILABLE")
        return "answer"


def test_answer_with_retries_waits_and_tries_again():
    waits = []
    pipeline = FlakyPipeline(failures=2)
    assert answer_with_retries(pipeline, "q", attempts=4, wait=30, sleep=waits.append) == "answer"
    assert waits == [30, 30]


def test_answer_with_retries_gives_up_after_the_last_attempt():
    pipeline = FlakyPipeline(failures=10)
    with pytest.raises(RuntimeError):
        answer_with_retries(pipeline, "q", attempts=3, wait=0, sleep=lambda _: None)
    assert pipeline.calls == 3


def test_short_metric_names():
    import pandas as pd

    scores = pd.DataFrame({"faithfulness": [1.0], "factual_correctness(mode=f1)": [0.5]})
    assert list(short_metric_names(scores).columns) == ["faithfulness", "factual_correctness"]
