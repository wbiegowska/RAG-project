"""Splits the cleaned rulebook into chunks. Every heading starts a new section (so a chunk
never mixes two rules) and sections longer than chunk_size are split again with langchain's
RecursiveCharacterTextSplitter. Each chunk keeps its section path and the page it starts on.
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.base.chunk import Chunk
from rag.impl.parser.markers import PAGE_MARKER
from rag.settings import settings

# regex built from PAGE_MARKER, captures the page number
PAGE_RE = re.compile(re.escape(PAGE_MARKER).replace(r"\{\}", r"(\d+)"))
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Section:
    """The text under one heading, before it gets split into chunks."""

    path: str
    start_page: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _without_markers("\n".join(self.lines))


def chunk_markdown(
    markdown: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Turns the cleaned rulebook into chunks with section path and page number."""
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
    """Cuts the markdown at every heading, keeping the path and start page of each part."""
    headings: dict[int, str] = {}  # level -> title
    page = 1
    current = Section(path="", start_page=page)
    sections = [current]
    for line in markdown.splitlines():
        page_marker = PAGE_RE.fullmatch(line.strip())
        if page_marker:
            page = int(page_marker.group(1))
            # kept for now, split_section needs it to find the page of each piece
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
    # a heading directly followed by another one has no text
    return [s for s in sections if s.text]


def split_section(
    section: Section, splitter: RecursiveCharacterTextSplitter
) -> list[tuple[str, int]]:
    """Splits a section if its too long, returns (text, page) for each piece."""
    text = "\n".join(section.lines).strip()
    markers = [(m.start(), int(m.group(1))) for m in PAGE_RE.finditer(text)]
    pieces = []
    for doc in splitter.create_documents([text]):
        start = doc.metadata["start_index"]
        # page of the piece = last page marker before its start
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
    # "Core Rules" is the book title, no need to have it in every path
    if len(titles) > 1:
        titles = titles[1:]
    return " > ".join(titles)


def _without_markers(text: str) -> str:
    text = PAGE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
