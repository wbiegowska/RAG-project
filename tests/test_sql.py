"""Text-to-SQL on a tiny database made in the test, gemini is mocked (no API key needed)."""

import sqlite3
from contextlib import closing
from unittest.mock import MagicMock

import pytest

from rag.impl.sql import handler
from rag.impl.sql.handler import MAX_ROWS, clean_sql, run_text_to_sql, validate_sql
from rag.settings import settings

N_SQUADS = MAX_ROWS + 10


@pytest.fixture
def unit_db(tmp_path, monkeypatch):
    path = tmp_path / "units.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE datasheets (Id TEXT, name TEXT)")
        conn.executemany(
            "INSERT INTO datasheets VALUES (?, ?)",
            [(str(i), f"Squad {i}") for i in range(N_SQUADS)],
        )
        conn.commit()
    monkeypatch.setattr(settings, "sqlite_path", path)
    return path


def use_fake_llm(monkeypatch, *answers: str) -> MagicMock:
    llm = MagicMock()
    llm.generate.side_effect = list(answers)
    monkeypatch.setattr(handler, "get_llm", lambda: llm)
    return llm


def test_sql_security_validation():
    assert validate_sql("SELECT * FROM units LIMIT 10") is True
    assert validate_sql("select name from units where faction = 'Space Marines'") is True
    assert validate_sql("SELECT * FROM units UNION SELECT * FROM weapons") is True
    assert validate_sql("DROP TABLE units;") is False
    assert validate_sql("DELETE FROM units WHERE id = 1;") is False
    assert validate_sql("SELECT * FROM units; SELECT * FROM weapons;") is False


def test_validation_looks_at_whole_words_and_the_first_word():
    assert validate_sql("SELECT LastUpdate FROM last_update;") is True
    assert validate_sql("WITH s AS (SELECT name FROM datasheets) SELECT * FROM s") is True
    assert validate_sql("PRAGMA table_info(datasheets)") is False
    assert validate_sql("SELECT 1; DROP TABLE datasheets") is False


def test_clean_sql_removes_the_markdown():
    assert clean_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert clean_sql("SQL: SELECT 1") == "SELECT 1"


def test_text_to_sql_runs_the_query(unit_db, monkeypatch):
    llm = use_fake_llm(monkeypatch, "```sql\nSELECT Id, name FROM datasheets WHERE Id = '3'\n```")

    result = run_text_to_sql("What is squad 3?")

    assert result.error is None
    assert result.columns == ["Id", "name"]
    assert result.rows == [("3", "Squad 3")]
    prompt = llm.generate.call_args.args[0]
    assert "datasheets_keywords" in prompt  # the data dictionary is in the prompt


def test_at_most_max_rows_are_read(unit_db, monkeypatch):
    use_fake_llm(monkeypatch, "SELECT name FROM datasheets")
    assert len(run_text_to_sql("all the squads").rows) == MAX_ROWS


def test_failed_query_gets_one_retry_with_the_error(unit_db, monkeypatch):
    llm = use_fake_llm(monkeypatch, "SELECT name FROM units", "SELECT name FROM datasheets LIMIT 1")

    result = run_text_to_sql("one squad")

    assert result.error is None
    assert result.sql == "SELECT name FROM datasheets LIMIT 1"
    assert "no such table: units" in llm.generate.call_args_list[1].args[0]


def test_unsafe_query_is_never_run(unit_db, monkeypatch):
    use_fake_llm(monkeypatch, "DROP TABLE datasheets", "DELETE FROM datasheets")

    result = run_text_to_sql("delete everything")

    assert result.error == "SQL safety validation failed."
    with closing(sqlite3.connect(unit_db)) as conn:
        assert conn.execute("SELECT count(*) FROM datasheets").fetchone()[0] == N_SQUADS


def test_missing_database_does_not_call_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sqlite_path", tmp_path / "missing.sqlite")
    llm = use_fake_llm(monkeypatch)

    result = run_text_to_sql("anything")

    assert "Database not found" in result.error
    llm.generate.assert_not_called()
