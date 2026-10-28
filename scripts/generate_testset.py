"""Generates the ragas test set (draft) from a selection of rulebook sections.

Usage: make testset              (slow, lots of gemini calls)
       make testset size=20      (reuses the saved knowledge graph, only redoes the questions)
       make testset rebuild=1    (builds the knowledge graph again from scratch)

If it stops while building the knowledge graph (quota, crash), just run it again: the
progress is saved in eval/knowledge_graph_partial.json and it continues from there.

The result goes to eval/testset_draft.json. Check the questions by hand, remove the bad
ones and save it as eval/testset.json, that one is what make eval uses.
"""

import argparse
import json

from ragas.testset.graph import KnowledgeGraph

from rag.evaluation.ragas_adapters import EmbedderRagasEmbeddings, GeminiRagasLLM
from rag.evaluation.testset import build_knowledge_graph, generate_testset, select_sections
from rag.impl.embedder.sbert import SbertEmbedder
from rag.impl.llm.gemini import get_llm
from rag.settings import settings


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Generate the ragas test set.")
    arg_parser.add_argument("--size", type=int, default=settings.testset_size)
    arg_parser.add_argument("--rebuild-graph", action="store_true")
    args = arg_parser.parse_args()

    graph_path = settings.eval_dir / "knowledge_graph.json"
    partial_path = settings.eval_dir / "knowledge_graph_partial.json"
    draft_path = settings.eval_dir / "testset_draft.json"
    llm = GeminiRagasLLM(get_llm(settings.judge_model))
    embeddings = EmbedderRagasEmbeddings(SbertEmbedder())

    if args.rebuild_graph:
        graph_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    if graph_path.exists():
        print(f"Using the saved knowledge graph {graph_path}")
        graph = KnowledgeGraph.load(graph_path)
    else:
        documents = select_sections(settings.rulebook_clean_md.read_text(encoding="utf-8"))
        print(f"Building the knowledge graph from {len(documents)} sections (takes a while)")
        try:
            graph = build_knowledge_graph(documents, llm, embeddings, partial_path)
        except Exception as error:
            raise SystemExit(
                f"\nStopped: {error}\nThe progress is saved, run make testset again to "
                "continue (if it's the gemini quota, wait a bit before)."
            ) from error
        graph.save(graph_path)
        partial_path.unlink()

    print(f"Generating {args.size} questions")
    try:
        samples = generate_testset(graph, llm, embeddings, args.size)
    except Exception as error:
        raise SystemExit(
            f"\nStopped: {error}\nThe knowledge graph is saved, run make testset again to "
            "only redo the questions."
        ) from error
    draft_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(samples)} questions to {draft_path}, check them before using them")


if __name__ == "__main__":
    main()
