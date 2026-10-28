"""Parses the rulebook PDF into markdown with docling (slow, a few minutes).

Usage: make parse
"""

import argparse
from pathlib import Path

from rag.impl.parser.docling_parser import parse_document
from rag.settings import settings


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Parse the rulebook PDF into markdown.")
    arg_parser.add_argument("--source_file", type=Path, default=settings.rulebook_pdf)
    arg_parser.add_argument("--target_file", type=Path, default=settings.rulebook_md)
    args = arg_parser.parse_args()

    markdown = parse_document(str(args.source_file))
    args.target_file.parent.mkdir(parents=True, exist_ok=True)
    args.target_file.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.target_file}")


if __name__ == "__main__":
    main()
