"""Tests for the cleaner, mostly based on the noise we actually found in the parsed rulebook."""

from rag.impl.parser.cleaner import clean_line, clean_markdown, fix_heading_levels
from rag.impl.parser.markers import PAGE_BREAK


def test_drops_page_numbers_labels_and_page_headers():
    assert clean_line("3  27") is None
    assert clean_line("F") is None
    assert clean_line("CORE RULES | FIGHT PHASE | FIGHT") is None


def test_keeps_rule_text_and_stratagem_costs():
    # "1CP" is short but its the cost of a stratagem, so it has to stay
    assert clean_line("1CP") == "1CP"
    assert clean_line("WHEN: Your Shooting phase.") == "WHEN: Your Shooting phase."


def test_removes_control_characters_and_double_bullets():
    assert clean_line("- ■ \x07Both players gain 1CP.") == "- Both players gain 1CP."
    assert clean_line("■ Army: All of the models.") == "- Army: All of the models."
    assert clean_line("- Always successful. ■ A roll of 1 fails.") == (
        "- Always successful.\n- A roll of 1 fails."
    )


def test_heading_levels():
    text = "\n".join(
        [
            "## Core Rules",
            "#### COMMAND PHASE",
            "## 1. COMMAND",
            "###### GAINING COMMAND POINTS",
            "## 2. BATTLE-SHOCK",
        ]
    )
    assert fix_heading_levels(text).split("\n") == [
        "# Core Rules",
        "### COMMAND PHASE",
        "#### 1. COMMAND",
        "##### GAINING COMMAND POINTS",
        "#### 2. BATTLE-SHOCK",
    ]


def test_skips_overview_page_and_adds_page_markers():
    pages = ["## Core Rules", "", "## Introduction", "scrambled overview", "### CORE CONCEPTS"]
    cleaned = clean_markdown(PAGE_BREAK.join(pages))
    assert "scrambled overview" not in cleaned
    assert "<!-- page: 5 -->" in cleaned
    assert "<!-- page: 2 -->" not in cleaned  # empty page, no marker
