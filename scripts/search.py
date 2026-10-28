"""Prints the chunks found for a question.

Usage: make search q="What is the rule for grenades?"
       make search q="..." mode=vector    (or hybrid, full)
"""

import argparse

from rag.impl.retriever.factory import SEARCH_MODES, get_rulebook_retriever
from rag.settings import settings


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Search the rulebook index.")
    arg_parser.add_argument("question")
    arg_parser.add_argument("-k", type=int, default=settings.top_k)
    arg_parser.add_argument("--mode", choices=SEARCH_MODES, default=None)
    args = arg_parser.parse_args()

    options = SEARCH_MODES[args.mode] if args.mode else {}
    retriever = get_rulebook_retriever(**options)
    for rank, chunk in enumerate(retriever.retrieve(args.question, k=args.k), start=1):
        print(f"\n#{rank}  score {chunk.score:.3f}  page {chunk.page}  [{chunk.section_path}]")
        print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))


if __name__ == "__main__":
    main()
