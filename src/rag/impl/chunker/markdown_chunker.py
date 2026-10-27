"""Splits the cleaned rulebook into chunks for the vector store.

How it works:
- we go through the markdown line by line and keep track of the current headings (the
  section path, e.g "THE BATTLE ROUND > SHOOTING PHASE > MAKING ATTACKS") and of the page
  we are on, using the <!-- page: N --> markers from the cleaner.
- every heading starts a new section, so a chunk never mixes two rules. Most sections of
  the rulebook are short (one rule, one stratagem, one weapon ability) so they become one
  chunk as they are, which is what we want: the median section is about 500 characters.
- sections that are too long (about 1 in 7) are split with langchain's
  RecursiveCharacterTextSplitter. It cuts on blank lines first, then on line breaks, then
  between sentences, so it avoids cutting in the middle of a sentence when it can. Each
  piece gets the page where it starts.

Chunk size: the embedding model (bge-small-en-v1.5) only reads the first 512 tokens of a
text. 1200 characters is roughly 300 tokens, which leaves room for the section path we put
in front of the text when embedding it (see retriever/vector.py).
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.base.chunk import Chunk
from rag.impl.parser.markers import PAGE_MARKER
from rag.settings import settings

# "<!-- page: {} -->" -> regex that finds "<!-- page: 12 -->" and captures the 12
PAGE_RE = re.compile(re.escape(PAGE_MARKER).replace(r"\{\}", r"(\d+)"))
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Section:
    """The text under one heading, before it gets split into chunks."""

    path: str
    start_page: int
    lines: list[str] = field(default_factory=list)


def chunk_markdown(
    markdown: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Turns the cleaned rulebook into chunks with their section path and page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=SEPARATORS,
        add_start_index=True,
    )
    chunks = []
    for section in split_into_sections(markdown):
        for text, page in split_section(section, splitter):
            chunk_id = f"chunk-{len(chunks):04d}"
            chunks.append(Chunk(id=chunk_id, text=text, section_path=section.path, page=page))
    return chunks


def split_into_sections(markdown: str) -> list[Section]:
    """Cuts the markdown at every heading and remembers the path and the page of each part."""
    headings: dict[int, str] = {}  # heading level -> title, for the current position
    page = 1
    current = Section(path="", start_page=page)
    sections = [current]
    for line in markdown.splitlines():
        page_marker = PAGE_RE.fullmatch(line.strip())
        if page_marker:
            page = int(page_marker.group(1))
            # we keep the marker in the text for now, split_section uses it for the pages
            current.lines.append(line)
            continue
        heading = HEADING_RE.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            # a new heading closes every heading of the same level or deeper
            headings = {lvl: t for lvl, t in headings.items() if lvl < level}
            headings[level] = title
            current = Section(path=_section_path(headings), start_page=page)
            sections.append(current)
            continue
        current.lines.append(line)
    # headings directly followed by another heading have no text of their own
    return [s for s in sections if _without_markers("\n".join(s.lines))]


def split_section(
    section: Section, splitter: RecursiveCharacterTextSplitter
) -> list[tuple[str, int]]:
    """Splits one section if its too long and returns (text, page) for each piece."""
    text = "\n".join(section.lines).strip()
    markers = [(m.start(), int(m.group(1))) for m in PAGE_RE.finditer(text)]
    pieces = []
    for doc in splitter.create_documents([text]):
        start = doc.metadata["start_index"]
        # the page of a piece is the last page marker before it starts
        page = section.start_page
        for position, marker_page in markers:
            if position <= start:
                page = marker_page
        piece = _without_markers(doc.page_content)
        if piece:
            pieces.append((piece, page))
    return pieces


def _section_path(headings: dict[int, str]) -> str:
    titles = [headings[level] for level in sorted(headings)]
    # "Core Rules" is the title of the book, it would be at the start of every path
    if len(titles) > 1:
        titles = titles[1:]
    return " > ".join(titles)


def _without_markers(text: str) -> str:
    text = PAGE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
