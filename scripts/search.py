"""Quick manual check of the vector search: prints the chunks found for a question.

Usage: make search q="What is the rule for grenades?"
"""

import argparse

from rag.impl.embedder.sbert import SbertEmbedder
from rag.impl.retriever.vector import VectorRetriever
from rag.settings import settings


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Search the rulebook index.")
    arg_parser.add_argument("question")
    arg_parser.add_argument("-k", type=int, default=settings.top_k)
    args = arg_parser.parse_args()

    retriever = VectorRetriever(SbertEmbedder())
    for rank, chunk in enumerate(retriever.retrieve(args.question, k=args.k), start=1):
        print(f"\n#{rank}  score {chunk.score:.3f}  page {chunk.page}  [{chunk.section_path}]")
        print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))


if __name__ == "__main__":
    main()
