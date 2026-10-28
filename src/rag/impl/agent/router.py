from rag.impl.llm.gemini import get_llm


def rewrite_query(question: str, history: list[dict[str, str]]) -> str:
    if not history:
        return question
    llm = get_llm()
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"Conversation History:\n{history_str}\n\nFollow-up question: {question}\n\nRewrite the follow-up question into a standalone question containing all necessary context. Return only the rewritten question."
    return llm.generate(
        prompt,
        system="You rewrite follow-up conversational questions into standalone search queries.",
    ).strip()


def route_query(question: str) -> str:
    llm = get_llm()
    prompt = f"""
    Classify the following question into one of three routes:
    - 'rules': for rules, phases, stratagems, victory points, movement, core mechanics found in the rulebook PDF.
    - 'units': for unit statistics, toughness, weapons, point costs, or datasheet values found in the SQLite database.
    - 'both': for queries requiring both unit stats and rulebook text (e.g. "Can an Intercessor Squad use the Grenade stratagem?").

    Question: {question}
    Return ONLY the exact word: rules, units, or both.
    """
    response = (
        llm.generate(prompt, system="You are an intent router for a Warhammer 40k RAG system.")
        .strip()
        .lower()
    )
    if "both" in response:
        return "both"
    elif "units" in response:
        return "units"
    return "rules"
