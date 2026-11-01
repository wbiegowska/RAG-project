from rag.base.chunk import Chunk
from rag.base.pipeline import BasePipeline, Message, PipelineResult
from rag.base.retriever import BaseRetriever
from rag.base.sql import SQLResult
from rag.impl.agent.router import route_query
from rag.impl.llm.gemini import get_llm
from rag.impl.prompt_loader import render_prompt
from rag.impl.retriever.factory import get_rulebook_retriever
from rag.impl.sql.handler import run_text_to_sql
from rag.monitoring.logger import TraceLogger  # Import the tracer
from rag.settings import settings


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
    retriever: BaseRetriever | None = None,
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
        chunks: list[Chunk] = []

        # Retrieve data based on route
        if route in ("units", "both"):
            sql_res = run_text_to_sql(standalone_q)
            if sql_res and getattr(sql_res, "sql", None):
                tracer.generated_sql = sql_res.sql

        if route in ("rules", "both"):
            if retriever is None:
                # shared hybrid search (BM25 + vectors), the models are only loaded once
                retriever = get_rulebook_retriever()
            chunks = retriever.retrieve(standalone_q, k=settings.top_k)
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
            answer=answer,
            route=route,
            standalone_question=standalone_q,
            rule_chunks=chunks,
            sql_result=sql_res,
        )

        return result


class RagPipeline(BasePipeline):
    """run_pipeline behind the shared BasePipeline interface, used by the evaluation."""

    def __init__(self, retriever: BaseRetriever | None = None) -> None:
        self.retriever = retriever

    def run(self, question: str, history: list[Message]) -> PipelineResult:
        return run_pipeline(question, history, retriever=self.retriever)
