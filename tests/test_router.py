from unittest.mock import patch

from rag.impl.agent.router import route_query
from rag.pipeline import rewrite_query


@patch("rag.impl.agent.router.get_llm")
def test_router_classification(mock_get_llm):
    # Mock the LLM return value for different routes
    mock_get_llm.return_value.generate.return_value = "units"
    assert route_query("What is the toughness of a Space Marine?") == "units"

    mock_get_llm.return_value.generate.return_value = "rules"
    assert route_query("What is the rule for grenades?") == "rules"

    mock_get_llm.return_value.generate.return_value = "both"
    assert route_query("Can an Intercessor Squad use the Grenade stratagem?") == "both"


@patch("rag.impl.agent.router.get_llm")
def test_router_uses_the_template_and_defaults_to_rules(mock_get_llm):
    mock_get_llm.return_value.generate.return_value = "Not sure."
    assert route_query("How does cover work?") == "rules"
    prompt = mock_get_llm.return_value.generate.call_args.args[0]
    assert "Question: How does cover work?" in prompt


def test_query_rewriter_no_history():
    question = "How do I score points?"
    assert rewrite_query(question, []) == question
