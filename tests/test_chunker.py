"""Tests for the chunker, on a small fake rulebook that looks like the cleaned one."""

from rag.impl.chunker.markdown_chunker import chunk_markdown

SAMPLE = """<!-- page: 1 -->

# Core Rules

Intro text of the book.

<!-- page: 11 -->

## THE BATTLE ROUND

### COMMAND PHASE

Commanders gauge the flow of the battle.

#### 1. COMMAND

Both players gain 1CP.

<!-- page: 12 -->

More text about the command step.

## STRATAGEMS

##### GRENADE

Roll six D6: for each 4+, that enemy unit suffers 1 mortal wound.
"""


def test_section_paths_and_pages():
    chunks = chunk_markdown(SAMPLE)
    assert [(c.section_path, c.page) for c in chunks] == [
        ("Core Rules", 1),
        ("THE BATTLE ROUND > COMMAND PHASE", 11),
        ("THE BATTLE ROUND > COMMAND PHASE > 1. COMMAND", 11),
        ("STRATAGEMS > GRENADE", 12),
    ]


def test_page_markers_are_removed_from_the_text():
    chunks = chunk_markdown(SAMPLE)
    assert all("<!--" not in c.text for c in chunks)
    assert chunks[2].text == "Both players gain 1CP.\n\nMore text about the command step."


def test_ids_are_unique():
    ids = [c.id for c in chunk_markdown(SAMPLE)]
    assert ids == ["chunk-0000", "chunk-0001", "chunk-0002", "chunk-0003"]


def test_long_sections_are_split_and_keep_their_page():
    paragraph = "This is a sentence about moving units over terrain features. " * 5
    long_section = "\n\n".join(
        ["<!-- page: 20 -->", "## MOVEMENT PHASE", paragraph, paragraph]
        + ["<!-- page: 21 -->", paragraph, paragraph]
    )
    chunks = chunk_markdown(long_section, chunk_size=400, chunk_overlap=0)
    assert len(chunks) == 4
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.section_path == "MOVEMENT PHASE" for c in chunks)
    assert [c.page for c in chunks] == [20, 20, 21, 21]
