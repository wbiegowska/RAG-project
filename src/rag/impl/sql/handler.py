import re
import sqlite3
from pathlib import Path

from rag.base.sql import SQLResult
from rag.impl.llm.gemini import get_llm

DATA_DICTIONARY = """
Database: wahadb.sqlite
Tables:
- units (id, name, faction, toughness, save, wounds, leadership, oc)
- weapons (id, unit_id, name, range, attacks, ballistic_skill, strength, armor_penetration, damage)
- stratagems (id, name, faction, turn, type, description)
- keywords (id, unit_id, keyword)
Note: "Space Marine" is a faction/army, not a specific unit name. Use LIKE for partial matches.
"""

FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
]


def validate_sql(sql: str) -> bool:
    sql_upper = sql.upper()
    # Check for forbidden mutation keywords
    if any(keyword in sql_upper for keyword in FORBIDDEN_KEYWORDS):
        return False
    # Block multi-statement queries separated by semicolons
    if ";" in sql.strip().rstrip(";"):
        return False
    return True


def run_text_to_sql(question: str) -> SQLResult:
    llm = get_llm()

    prompt = f"""
    {DATA_DICTIONARY}
    User Question: {question}

    Task: Write a single, read-only SQL SELECT query to answer the question. 
    Always include a LIMIT clause (max 50 rows). Return ONLY the raw SQL query.
    """

    sql_query = llm.generate(
        prompt, system="You write secure SQLite queries for Warhammer 40k data."
    ).strip()
    sql_query = re.sub(r"^```sql\s*|\s*```$", "", sql_query, flags=re.IGNORECASE).strip()

    if not validate_sql(sql_query):
        return SQLResult(
            question=question,
            sql=sql_query,
            columns=[],
            rows=[],
            error="SQL safety validation failed.",
        )

    db_path = Path("data/wahadb.sqlite")
    uri = f"file:{db_path.absolute()}?mode=ro"

    try:
        with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return SQLResult(
                question=question,
                sql=sql_query,
                columns=columns,
                rows=rows,
                error=None,
            )
    except Exception as e:
        repair_prompt = (
            f"The previous SQL query failed with error: {str(e)}\n"
            f"Query: {sql_query}\n"
            f"Fix the SQL query for question: {question}. Return only raw SQL."
        )
        fixed_sql = llm.generate(repair_prompt, system="You fix SQLite queries.").strip()
        fixed_sql = re.sub(r"^```sql\s*|\s*```$", "", fixed_sql, flags=re.IGNORECASE).strip()

        if not validate_sql(fixed_sql):
            return SQLResult(
                question=question,
                sql=fixed_sql,
                columns=[],
                rows=[],
                error=str(e),
            )

        try:
            with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(fixed_sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                return SQLResult(
                    question=question,
                    sql=fixed_sql,
                    columns=columns,
                    rows=rows,
                    error=None,
                )
        except Exception as retry_err:
            return SQLResult(
                question=question,
                sql=fixed_sql,
                columns=[],
                rows=[],
                error=str(retry_err),
            )
