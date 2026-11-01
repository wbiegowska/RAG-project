from unittest.mock import MagicMock, patch

from rag.base.sql import SQLResult
from rag.pipeline import PipelineResult, rewrite_query, run_pipeline


@patch("rag.pipeline.get_llm")
def test_rewrite_query_without_history(mock_llm):
    result = rewrite_query("What is unit stats?", history=[])
    assert result == "What is unit stats?"


@patch("rag.pipeline.get_llm")
def test_rewrite_query_with_history(mock_llm):
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "What is the save of the Intercessor Squad?"
    mock_llm.return_value = mock_gemini

    history = [
        {"role": "user", "content": "Tell me about Intercessor Squad"},
        {"role": "assistant", "content": "Intercessors are Space Marine units."},
    ]
    result = rewrite_query("And its save?", history=history)

    assert result == "What is the save of the Intercessor Squad?"


@patch("rag.pipeline.get_llm")
@patch("rag.pipeline.run_text_to_sql")
@patch("rag.pipeline.route_query")
def test_pipeline_units_route(mock_route, mock_sql, mock_llm):
    mock_route.return_value = "units"
    mock_sql.return_value = SQLResult(
        question="What is unit stats?",
        sql="SELECT * FROM units LIMIT 10;",
        columns=["name", "toughness"],
        rows=[("Intercessor", 4)],
        error=None,
    )
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "Intercessors have toughness 4."
    mock_llm.return_value = mock_gemini

    result = run_pipeline("What is unit stats?")

    assert isinstance(result, PipelineResult)
    assert result.route == "units"
    assert result.sql_result is not None
    assert "Intercessors" in result.answer


@patch("rag.pipeline.get_llm")
@patch("rag.pipeline.route_query")
def test_pipeline_rules_route(mock_route, mock_llm):
    mock_route.return_value = "rules"
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["Rule 1: Cover gives +1 Save."]

    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "Cover gives +1 to armor save."
    mock_llm.return_value = mock_gemini

    result = run_pipeline("How does cover work?", retriever=mock_retriever)

    assert result.route == "rules"
    assert len(result.retrieved_chunks) == 1
    assert "Cover gives +1" in result.answer
