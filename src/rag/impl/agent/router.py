from rag.impl.llm.gemini import get_llm
from rag.impl.prompt_loader import render_prompt


def route_query(question: str) -> str:
    """Asks gemini if the question needs the rulebook, the unit database or both."""
    prompt = render_prompt("router.jinja", question=question)
    response = (
        get_llm()
        .generate(prompt, system="You are an intent router for a Warhammer 40k RAG system.")
        .strip()
        .lower()
    )
    if "both" in response:
        return "both"
    elif "units" in response:
        return "units"
    return "rules"
