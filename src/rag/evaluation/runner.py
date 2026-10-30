"""Runs the evaluation on the frozen test set (eval/testset.json).

Retrieval: for each search setup we compare the retrieved chunks with the reference
contexts of the test set. The reference contexts are exactly the text of one of our chunks
(same "section path + text" format) so we can compare the texts, no chunk ids involved.
- hit rate: at least one reference context is in the top k
- MRR: 1 / rank of the first reference context found (0 if none)
- ragas non-LLM context recall and precision (string similarity, no gemini calls)

Answers: the pipeline answers every question and a gemini judge scores them with ragas:
faithfulness (is everything in the answer supported by the retrieved chunks), response
relevancy (does it answer the question) and factual correctness (does it match the
reference answer).
"""

import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig

from rag.base.pipeline import BasePipeline
from rag.base.retriever import BaseRetriever
from rag.impl.retriever.vector import embedding_text

# the new metrics in ragas.metrics.collections need an "instructor" LLM, our gemini adapter
# is a classic BaseRagasLLM (like in the lab) so we use the classic metrics
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        NonLLMContextPrecisionWithReference,
        NonLLMContextRecall,
        ResponseRelevancy,
    )

RUN_CONFIG = RunConfig(max_workers=4, timeout=300, max_retries=3, max_wait=60)
# the free tier often answers 503 "high demand" for a little while, so we wait and try again
# instead of skipping the question (no fallback model here, see scripts/evaluate.py)
ANSWER_ATTEMPTS = 4
ANSWER_RETRY_WAIT = 30  # seconds
RETRIEVAL_SCORES = ["hit_rate", "mrr", "context_recall", "context_precision"]
ANSWER_SCORES = ["faithfulness", "answer_relevancy", "factual_correctness"]


def load_testset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def hit_rate(retrieved: list[str], references: list[str]) -> float:
    return float(any(reference in retrieved for reference in references))


def reciprocal_rank(retrieved: list[str], references: list[str]) -> float:
    for rank, context in enumerate(retrieved, start=1):
        if context in references:
            return 1 / rank
    return 0.0


def evaluate_retrieval(
    testset: list[dict], retrievers: dict[str, BaseRetriever], k: int
) -> pd.DataFrame:
    """One row per (setup, question)."""
    rows = []
    for setup, retriever in retrievers.items():
        samples = []
        for item in testset:
            retrieved = [embedding_text(c) for c in retriever.retrieve(item["user_input"], k=k)]
            references = item["reference_contexts"]
            samples.append(
                SingleTurnSample(
                    user_input=item["user_input"],
                    retrieved_contexts=retrieved,
                    reference_contexts=references,
                )
            )
            rows.append(
                {
                    "setup": setup,
                    "question": item["user_input"],
                    "type": item.get("synthesizer_name", ""),
                    "hit_rate": hit_rate(retrieved, references),
                    "mrr": reciprocal_rank(retrieved, references),
                }
            )
        scores = evaluate(
            EvaluationDataset(samples=samples),
            metrics=[NonLLMContextRecall(), NonLLMContextPrecisionWithReference()],
            show_progress=False,
        ).to_pandas()
        setup_rows = [row for row in rows if row["setup"] == setup]
        for row, (_, score) in zip(setup_rows, scores.iterrows(), strict=True):
            row["context_recall"] = score["non_llm_context_recall"]
            row["context_precision"] = score["non_llm_context_precision_with_reference"]
    return pd.DataFrame(rows)


def evaluate_answers(
    testset: list[dict], pipeline: BasePipeline, judge_llm, embeddings
) -> pd.DataFrame:
    """The pipeline answers each question, then the judge scores the answers."""
    samples, kept = [], []
    for number, item in enumerate(testset, start=1):
        print(f"Answering {number}/{len(testset)}: {item['user_input'][:70]}")
        try:
            result = answer_with_retries(pipeline, item["user_input"])
        except Exception as error:  # still failing, we skip it instead of losing the run
            print(f"  skipped, the pipeline failed: {str(error)[:120]}")
            continue
        samples.append(
            SingleTurnSample(
                user_input=item["user_input"],
                response=result.answer,
                retrieved_contexts=[embedding_text(c) for c in result.rule_chunks],
                reference=item["reference"],
            )
        )
        kept.append(item)

    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), FactualCorrectness()]
    scores = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RUN_CONFIG,
    ).to_pandas()
    scores.insert(1, "type", [item.get("synthesizer_name", "") for item in kept])
    return short_metric_names(scores)


def short_metric_names(scores: pd.DataFrame) -> pd.DataFrame:
    # ragas calls the column "factual_correctness(mode=f1)", we keep only the name
    return scores.rename(columns={c: c.split("(")[0] for c in scores.columns})


def answer_with_retries(
    pipeline: BasePipeline,
    question: str,
    attempts: int = ANSWER_ATTEMPTS,
    wait: float = ANSWER_RETRY_WAIT,
    sleep: Callable[[float], None] = time.sleep,
):
    for attempt in range(1, attempts + 1):
        try:
            return pipeline.run(question, history=[])
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"  failed ({str(error)[:60]}...), trying again in {wait:.0f}s")
            sleep(wait)


def summary_table(scores: pd.DataFrame, group_by: str, columns: list[str]) -> str:
    """Markdown table with the mean of each score per group (and the count)."""
    means = scores.groupby(group_by, sort=False)[columns].mean()
    counts = scores.groupby(group_by, sort=False).size()
    lines = [
        "| " + " | ".join([group_by, "n", *columns]) + " |",
        "|" + "---|" * (len(columns) + 2),
    ]
    for name, row in means.iterrows():
        values = [f"{row[c]:.2f}" for c in columns]
        lines.append("| " + " | ".join([str(name), str(counts[name]), *values]) + " |")
    return "\n".join(lines)
