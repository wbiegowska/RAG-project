from rag.impl.prompt_loader import render_prompt


def test_render_router_template():
    rendered = render_prompt(
        "router.jinja", question="What is the toughness of a Space Marine?", history=[]
    )
    assert "What is the toughness of a Space Marine?" in rendered
    assert "rules" in rendered
    assert "units" in rendered


def test_render_sql_template():
    rendered = render_prompt(
        "sql.jinja", schema="Table: units (id, name, toughness)", question="List all units"
    )
    assert "Table: units" in rendered
    assert "List all units" in rendered
    assert "LIMIT clause" in rendered


def test_render_rewrite_template():
    history = [{"role": "user", "content": "Tell me about Space Marines"}]
    rendered = render_prompt("rewrite.jinja", question="And their save?", history=history)
    assert "Tell me about Space Marines" in rendered
    assert "And their save?" in rendered


def test_render_answer_template_with_metadata():
    class DummyChunk:
        section_path = "Movement Phase > Normal Move"
        page = 12
        text = "Units can move up to their Movement characteristic."

    class DummySQLResult:
        sql = "SELECT name, toughness FROM units WHERE name = 'Intercessor';"
        columns = ["name", "toughness"]
        rows = [("Intercessor", 4)]

    rendered = render_prompt(
        "answer.jinja",
        question="How fast can units move?",
        rules_context=[DummyChunk()],
        sql_context=DummySQLResult(),
    )

    assert "Movement Phase > Normal Move" in rendered
    assert "Page 12" in rendered
    assert "SELECT name, toughness FROM units" in rendered
    assert "I don't know based on the provided context." in rendered
