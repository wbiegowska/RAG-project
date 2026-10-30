"""Evaluates the rulebook search and the answers on the frozen test set (eval/testset.json).

Usage: make eval                    (search + answers)
       make eval part=retrieval     (only the search, no gemini calls)
       make eval part=answers
       make eval part=summary       (only rebuilds the summaries from the saved csv files)

The scores of every question go to eval/results/*.csv and the summaries to *_summary.md.
"""

import argparse
import warnings

import pandas as pd

from rag.evaluation.baseline import StandInPipeline
from rag.evaluation.ragas_adapters import EmbedderRagasEmbeddings, GeminiRagasLLM
from rag.evaluation.runner import (
    ANSWER_SCORES,
    RETRIEVAL_SCORES,
    evaluate_answers,
    evaluate_retrieval,
    load_testset,
    short_metric_names,
    summary_table,
)
from rag.impl.embedder.sbert import SbertEmbedder
from rag.impl.llm.gemini import GeminiClient
from rag.impl.retriever.factory import SEARCH_MODES, get_rulebook_retriever
from rag.settings import settings

# ragas warns that evaluate() will be replaced in its next big version, nothing to do for us
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Run the ragas evaluation.")
    arg_parser.add_argument(
        "--part", choices=["retrieval", "answers", "all", "summary"], default="all"
    )
    args = arg_parser.parse_args()

    testset = load_testset(settings.eval_dir / "testset.json")
    results_dir = settings.eval_dir / "results"
    results_dir.mkdir(exist_ok=True)
    if args.part in ("retrieval", "all"):
        run_retrieval(testset, results_dir)
    if args.part in ("answers", "all"):
        run_answers(testset, results_dir)
    if args.part == "summary":
        rebuild_summaries(testset, results_dir)


def run_retrieval(testset: list[dict], results_dir) -> None:
    retrievers = {name: get_rulebook_retriever(**opts) for name, opts in SEARCH_MODES.items()}
    scores = evaluate_retrieval(testset, retrievers, k=settings.top_k)
    scores.to_csv(results_dir / "retrieval_scores.csv", index=False)
    write_retrieval_summary(scores, testset, results_dir)


def write_retrieval_summary(scores: pd.DataFrame, testset: list[dict], results_dir) -> None:
    default = _default_setup()
    report = "\n\n".join(
        [
            f"# Retrieval ({len(testset)} questions, top {settings.top_k} chunks)",
            summary_table(scores, "setup", RETRIEVAL_SCORES),
            f"By question type, {default} setup (the default):",
            summary_table(scores[scores["setup"] == default], "type", RETRIEVAL_SCORES),
        ]
    )
    _save(report, results_dir / "retrieval_summary.md")


def run_answers(testset: list[dict], results_dir) -> None:
    if settings.judge_model == settings.llm_model:
        print("Warning: the judge is the same model as the one answering")
    # no fallback models here, so all the answers (and all the scores) come from one model
    answer_llm = GeminiClient(model=settings.llm_model, fallback_models=[])
    judge = GeminiRagasLLM(GeminiClient(model=settings.judge_model, fallback_models=[]))
    embeddings = EmbedderRagasEmbeddings(SbertEmbedder())
    pipeline = StandInPipeline(get_rulebook_retriever(), answer_llm)

    scores = evaluate_answers(testset, pipeline, judge, embeddings)
    scores.to_csv(results_dir / "answers_scores.csv", index=False)
    write_answers_summary(scores, testset, results_dir)


def write_answers_summary(scores: pd.DataFrame, testset: list[dict], results_dir) -> None:
    report = "\n\n".join(
        [
            f"# Answers ({len(scores)} of {len(testset)} questions answered)",
            f"Pipeline: stand-in (search + {settings.llm_model}), "
            f"setup {_default_setup()}, judge: {settings.judge_model}",
            summary_table(scores.assign(all="all"), "all", ANSWER_SCORES),
            "By question type:",
            summary_table(scores, "type", ANSWER_SCORES),
        ]
    )
    _save(report, results_dir / "answers_summary.md")


def rebuild_summaries(testset: list[dict], results_dir) -> None:
    """Rebuilds the summaries from the saved scores, without calling gemini again."""
    retrieval_csv = results_dir / "retrieval_scores.csv"
    answers_csv = results_dir / "answers_scores.csv"
    if retrieval_csv.exists():
        write_retrieval_summary(pd.read_csv(retrieval_csv), testset, results_dir)
    if answers_csv.exists():
        scores = short_metric_names(pd.read_csv(answers_csv))
        scores.to_csv(answers_csv, index=False)
        write_answers_summary(scores, testset, results_dir)


def _default_setup() -> str:
    wanted = {"use_bm25": settings.use_bm25, "use_reranker": settings.use_reranker}
    return next(name for name, options in SEARCH_MODES.items() if options == wanted)


def _save(report: str, path) -> None:
    path.write_text(report + "\n", encoding="utf-8")
    print(f"\n{report}\n\nSaved in {path}")


if __name__ == "__main__":
    main()
