from dataclasses import dataclass, field
from typing import Any

from rag.base.sql import SQLResult
from rag.impl.agent.router import route_query
from rag.impl.embedder.sbert import SbertEmbedder
from rag.impl.llm.gemini import get_llm
from rag.impl.prompt_loader import render_prompt
from rag.impl.retriever.vector import VectorRetriever
from rag.impl.sql.handler import run_text_to_sql
from rag.monitoring.logger import TraceLogger  # Import the tracer


@dataclass
class PipelineResult:
    question: str
    standalone_question: str
    answer: str
    route: str
    sql_result: SQLResult | None = None
    retrieved_chunks: list[Any] = field(default_factory=list)


def rewrite_query(question: str, history: list[dict[str, str]]) -> str:
    """Turns follow-up questions into clear, standalone questions using history."""
    if not history:
        return question

    llm = get_llm()
    prompt = render_prompt("rewrite.jinja", question=question, history=history)
    standalone = llm.generate(prompt).strip()
    return standalone if standalone else question


def run_pipeline(
    question: str,
    history: list[dict[str, str]] | None = None,
    retriever: VectorRetriever | None = None,
) -> PipelineResult:
    """Orchestrates query rewriting, routing, retrieval (SQL and/or Vector),
    prompt rendering, and generation with integrated tracing."""
    if history is None:
        history = []

    # Initialize the trace logger for this turn
    with TraceLogger(question) as tracer:
        # Rewrite query into standalone question if history exists
        standalone_q = rewrite_query(question, history)
        tracer.standalone_question = standalone_q

        # Determine intent/route using standalone question
        route = route_query(standalone_q)
        tracer.route = route

        sql_res: SQLResult | None = None
        chunks: list[Any] = []

        # Retrieve data based on route
        if route in ("units", "both"):
            sql_res = run_text_to_sql(standalone_q)
            if sql_res and getattr(sql_res, "sql", None):
                tracer.generated_sql = sql_res.sql

        if route in ("rules", "both"):
            if retriever is None:
                retriever = VectorRetriever(SbertEmbedder())
            chunks = retriever.retrieve(standalone_q, k=3)
            # Record retrieved chunk IDs or page references
            tracer.retrieved_chunk_ids = [getattr(c, "id", str(i)) for i, c in enumerate(chunks)]

        # Format context with Jinja2 prompt template
        prompt = render_prompt(
            "answer.jinja",
            question=standalone_q,
            rules_context=chunks,
            sql_context=sql_res if (sql_res and not sql_res.error) else None,
        )

        # Generate final answer with Gemini
        llm = get_llm()
        answer = llm.generate(prompt)
        tracer.answer = answer

        # Construct and return the standard result
        result = PipelineResult(
            question=question,
            standalone_question=standalone_q,
            answer=answer,
            route=route,
            sql_result=sql_res,
            retrieved_chunks=chunks,
        )

        return result
