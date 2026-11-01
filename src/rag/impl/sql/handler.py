import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from rag.base.sql import SQLResult
from rag.impl.llm.gemini import get_llm
from rag.impl.prompt_loader import render_prompt
from rag.settings import settings

SYSTEM = "You write read-only SQLite queries for the Warhammer 40k unit database."
MAX_ROWS = 50

# checked as whole words, so a table like last_update is still allowed
FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "ATTACH", "PRAGMA"]
_FORBIDDEN = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
# gemini sometimes wraps the query in ```sql ... ``` or starts with "SQL:"
_WRAPPING = re.compile(r"^(```(sql)?|sql:)\s*|\s*```$", re.IGNORECASE)


@lru_cache
def load_schema(path: Path) -> str:
    """The data dictionary given with the project: tables, columns, join keys, example queries."""
    return path.read_text(encoding="utf-8")


def validate_sql(sql: str) -> bool:
    """One SELECT (or WITH ... SELECT) statement, nothing that can write."""
    query = sql.strip().rstrip(";").strip()
    if not re.match(r"(SELECT|WITH)\b", query, re.IGNORECASE):
        return False
    if ";" in query:  # a second statement
        return False
    return not _FORBIDDEN.search(query)


def clean_sql(text: str) -> str:
    return _WRAPPING.sub("", text.strip()).strip()


def run_text_to_sql(question: str) -> SQLResult:
    """Gemini writes the query from the data dictionary and we run it read-only. If sqlite
    returns an error, gemini gets one more try with the error message."""
    if not settings.sqlite_path.exists():
        return SQLResult(
            question=question,
            sql="",
            error=f"Database not found: {settings.sqlite_path} (see the README)",
        )
    llm = get_llm()
    prompt = render_prompt(
        "sql.jinja", schema=load_schema(settings.data_dictionary), question=question
    )
    sql = clean_sql(llm.generate(prompt, system=SYSTEM))
    result = execute(question, sql)
    if result.error is None:
        return result

    retry_prompt = (
        f"{prompt}\n\nYour previous query:\n{sql}\nfailed with this error: {result.error}\n"
        "Write a corrected query."
    )
    sql = clean_sql(llm.generate(retry_prompt, system=SYSTEM))
    return execute(question, sql)


def execute(question: str, sql: str) -> SQLResult:
    if not validate_sql(sql):
        return SQLResult(question=question, sql=sql, error="SQL safety validation failed.")
    # mode=ro: sqlite itself refuses to write, the validation above is only a first filter
    uri = settings.sqlite_path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as conn:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            # the prompt asks for a LIMIT but we never read more than MAX_ROWS anyway
            rows = cursor.fetchmany(MAX_ROWS)
    except sqlite3.Error as error:
        return SQLResult(question=question, sql=sql, error=str(error))
    return SQLResult(question=question, sql=sql, columns=columns, rows=rows)
