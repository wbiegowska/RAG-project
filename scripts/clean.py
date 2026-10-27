"""Runs the cleaner on the parsed rulebook and saves the result.

Usage: make clean-md
"""

import argparse
from pathlib import Path

from rag.impl.parser.cleaner import clean_markdown
from rag.settings import settings


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Clean the parsed rulebook markdown.")
    arg_parser.add_argument("--source_file", type=Path, default=settings.rulebook_md)
    arg_parser.add_argument("--target_file", type=Path, default=settings.rulebook_clean_md)
    args = arg_parser.parse_args()

    cleaned = clean_markdown(args.source_file.read_text(encoding="utf-8"))
    args.target_file.write_text(cleaned, encoding="utf-8")
    print(f"Wrote {args.target_file}")


if __name__ == "__main__":
    main()
