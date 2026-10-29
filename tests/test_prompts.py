from rag.impl.prompt_loader import render_prompt


def test_render_router_template():
    rendered = render_prompt(
        "router.jinja", question="What is the toughness of a Space Marine?", history=[]
    )
    # Check that our question and routing instructions made it into the template
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
