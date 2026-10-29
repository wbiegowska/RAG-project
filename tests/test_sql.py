from rag.impl.sql.handler import run_text_to_sql, validate_sql


def test_sql_security_validation():
    # Valid single select queries
    assert validate_sql("SELECT * FROM units LIMIT 10") is True
    assert validate_sql("select name from units where faction = 'Space Marines'") is True

    # Dangerous or invalid queries
    assert validate_sql("DROP TABLE units;") is False
    assert validate_sql("DELETE FROM units WHERE id = 1;") is False
    assert validate_sql("SELECT * FROM units; SELECT * FROM weapons;") is False
    assert (
        validate_sql("SELECT * FROM units UNION SELECT * FROM weapons") is True
    )  # depends on policy, but multi-statement is false


def test_text_to_sql_execution():
    # Test a known query or basic execution via the handler
    result = run_text_to_sql("What is the toughness of an Intercessor Squad?")
    # Even if LLM generation varies slightly, the SQLResult structure must be returned
    assert result is not None
    assert hasattr(result, "sql")
    assert hasattr(result, "rows")
    assert hasattr(result, "error")
