"""Cleans the docling markdown before chunking: drops page 4, removes the noise (page
numbers, page headers, control characters, double bullets) and fixes the heading levels.
Page breaks become "<!-- page: N -->" so the chunker knows the page of each chunk.
Why each fix is needed is explained in docs/technical_choices.md.
"""

import re

from rag.impl.parser.markers import PAGE_BREAK, PAGE_MARKER

# overview page, docling mixes it with a hidden copy of pages 5-9
SKIP_PAGES = {4}
TITLE = "Core Rules"

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
STRAY_LINE = re.compile(r"^(\d+(\s+\d+)*|[A-Z])$")  # page numbers and diagram labels
PAGE_HEADER = re.compile(r"^CORE RULES\s*\|")
BULLET = re.compile(r"^(-\s*)?■\s*")
INLINE_BULLET = re.compile(r"\s+■\s*")  # 2 bullet points stuck on the same line
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
NUMBERED_STEP = re.compile(r"^\d+\.?\s+\S")  # "1. COMMAND" but also "1 MUSTER ARMIES"
MAX_SECTION_LEVEL = 4  # anything deeper is a single rule, not a section

# every terrain type has the same sub-headings, they go one level under it
TERRAIN_TYPES = {
    "CRATERS AND RUBBLE",
    "BARRICADES AND FUEL PIPES",
    "BATTLEFIELD DEBRIS AND STATUARY",
    "HILLS, INDUSTRIAL STRUCTURES, SEALED BUILDINGS AND ARMOURED CONTAINERS",
    "WOODS",
    "RUINS",
}
TERRAIN_DETAILS = {
    "FOOTPRINT",
    "PLUNGING FIRE",
    "MOVEMENT",
    "VISIBILITY",
    "ENGAGEMENT RANGE",
    "BENEFIT OF COVER",
}
# docling puts it under EXAMPLE BATTLEFIELDS but its a chapter on its own
CHAPTERS = {"AIRCRAFT"}


def clean_markdown(markdown: str) -> str:
    """Cleans the whole rulebook and adds a page marker at the start of each kept page."""
    kept_pages = []
    for page_no, page in enumerate(markdown.split(PAGE_BREAK), start=1):
        if page_no in SKIP_PAGES:
            continue
        lines = (clean_line(line) for line in page.splitlines())
        body = "\n".join(line for line in lines if line is not None).strip()
        # pages with only pictures end up empty
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
    """Title -> "#", the rest one level up, numbered steps and terrain details under their parent"""
    section_level = 1  # last section level, numbered steps go under it
    terrain_level = None  # None when we're not in a terrain type
    output = []
    for line in text.split("\n"):
        match = HEADING.match(line)
        if not match:
            output.append(line)
            continue
        level, title = len(match.group(1)), match.group(2).strip()
        if title == TITLE:
            new_level = 1
        elif title in CHAPTERS:
            new_level = section_level = 2
        elif NUMBERED_STEP.match(title):
            new_level = min(section_level + 1, 6)
        elif terrain_level is not None and title in TERRAIN_DETAILS:
            new_level = min(terrain_level + 1, 6)
        else:
            new_level = max(level - 1, 2)
            if new_level <= MAX_SECTION_LEVEL:
                section_level = new_level

        if title in TERRAIN_TYPES:
            terrain_level = new_level
        elif title not in TERRAIN_DETAILS:
            terrain_level = None
        output.append(f"{'#' * new_level} {title}")
    return "\n".join(output)
