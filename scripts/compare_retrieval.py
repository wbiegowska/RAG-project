"""Top 3 chunks of a few test questions for each search setup (vector, hybrid, full).

Usage: make compare-retrieval
"""

from rag.impl.retriever.factory import SEARCH_MODES, get_rulebook_retriever

QUESTIONS = [
    "What is the rule for grenades?",
    "How can I use a stratagem?",
    "What are the different ways to score points?",
    "Can my unit shoot after falling back?",
    "What happens when a unit is below half strength?",
]


def main() -> None:
    for question in QUESTIONS:
        print(f"\n=== {question}")
        for mode, options in SEARCH_MODES.items():
            print(f"  {mode}:")
            for chunk in get_rulebook_retriever(**options).retrieve(question, k=3):
                print(f"    p{chunk.page:<3} {chunk.section_path}")


if __name__ == "__main__":
    main()
