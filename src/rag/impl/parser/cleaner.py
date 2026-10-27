"""Cleaning step for the parsed rulebook (runs after docling, before the chunking).

Docling gets most of the text right but it leaves some noise behind, this is what we fix:

- Page 4 is dropped. It's the "how to read this book" page: the list of chapters with
  their page ranges and a few short notes (hints and tips, summaries, page numbers).
  The problem is that the PDF also has the text of pages 5-9 sitting underneath that
  panel as a background, cut off at the edges, and docling reads both layers and mixes
  them together. We checked the output: about half of the page is this hidden copy
  (pages 5-9 are parsed fine on their own) and the rest is the overview, which doesn't
  contain any actual rules. Since we cant separate the two layers reliably we drop the
  whole page, so the only thing we lose is the chapter list and those notes.
- stray page numbers and labels from the diagrams ("3  27", "F") and the page headers
  like "CORE RULES | FIGHT PHASE"
- invisible control characters and double bullets ("- ■ ...") coming from the fonts in
  the pdf, and bullet points that ended up stuck together on one line
- heading levels. Docling guesses them from the font sizes and it's mostly ok, but the
  numbered steps ("1. HIT ROLL") end up at the top level. The title becomes "#",
  everything else moves up one level and numbered steps go one level under their section.

The page breaks from docling are turned into "<!-- page: N -->" so that the chunker
knows which page each chunk comes from.
"""

import re

from rag.impl.parser.markers import PAGE_BREAK, PAGE_MARKER

# see the top of the file for why page 4 is skipped
SKIP_PAGES = {4}
TITLE = "Core Rules"

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
STRAY_LINE = re.compile(r"^(\d+(\s+\d+)*|[A-Z])$")  # only numbers, or a single capital letter
PAGE_HEADER = re.compile(r"^CORE RULES\s*\|")
BULLET = re.compile(r"^(-\s*)?■\s*")
INLINE_BULLET = re.compile(r"\s+■\s*")  # 2 bullet points stuck on the same line
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
NUMBERED_STEP = re.compile(r"^\d+\.?\s+\S")  # "1. COMMAND" but also "1 MUSTER ARMIES"
MAX_SECTION_LEVEL = 4  # anything deeper is a single rule, not a section


def clean_markdown(markdown: str) -> str:
    """Cleans the whole rulebook and adds a page marker at the start of every page we keep."""
    kept_pages = []
    for page_no, page in enumerate(markdown.split(PAGE_BREAK), start=1):
        if page_no in SKIP_PAGES:
            continue
        lines = (clean_line(line) for line in page.splitlines())
        body = "\n".join(line for line in lines if line is not None).strip()
        # pages with only pictures end up empty, they dont get a marker
        if body:
            kept_pages.append(f"{PAGE_MARKER.format(page_no)}\n\n{body}")
    text = fix_heading_levels("\n\n".join(kept_pages))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def clean_line(line: str) -> str | None:
    """Cleans one line, returns None if the line should be removed."""
    line = CONTROL_CHARS.sub("", line).rstrip()
    stripped = line.strip()
    if STRAY_LINE.match(stripped) or PAGE_HEADER.match(stripped):
        return None
    line = BULLET.sub("- ", line)
    return INLINE_BULLET.sub("\n- ", line)


def fix_heading_levels(text: str) -> str:
    """Fixes the heading levels so they make sense as a hierarchy (see top of the file)"""
    section_level = 1  # level of the last section heading, numbered steps go under it
    output = []
    for line in text.split("\n"):
        match = HEADING.match(line)
        if not match:
            output.append(line)
            continue
        level, title = len(match.group(1)), match.group(2).strip()
        if title == TITLE:
            new_level = 1
        elif NUMBERED_STEP.match(title):
            new_level = min(section_level + 1, 6)
        else:
            new_level = max(level - 1, 2)
            if new_level <= MAX_SECTION_LEVEL:
                section_level = new_level
        output.append(f"{'#' * new_level} {title}")
    return "\n".join(output)
