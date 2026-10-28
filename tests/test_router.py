from unittest.mock import patch

from rag.impl.agent.router import rewrite_query, route_query


@patch("rag.impl.agent.router.get_llm")
def test_router_classification(mock_get_llm):
    # Mock the LLM return value for different routes
    mock_get_llm.return_value.generate.return_value = "units"
    assert route_query("What is the toughness of a Space Marine?") == "units"

    mock_get_llm.return_value.generate.return_value = "rules"
    assert route_query("What is the rule for grenades?") == "rules"

    mock_get_llm.return_value.generate.return_value = "both"
    assert route_query("Can an Intercessor Squad use the Grenade stratagem?") == "both"


def test_query_rewriter_no_history():
    question = "How do I score points?"
    assert rewrite_query(question, []) == question
