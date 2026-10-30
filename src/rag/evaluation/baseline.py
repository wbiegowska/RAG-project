"""Simple rulebook-only pipeline: search + one gemini call. Used to evaluate the answers
until the full pipeline (router, SQL, memory) is ready, and then as a baseline for it."""

from rag.base.llm import LLMClient
from rag.base.pipeline import BasePipeline, Message, PipelineResult
from rag.base.retriever import BaseRetriever
from rag.settings import settings

SYSTEM = (
    "You answer questions about the Warhammer 40,000 Core Rules. Use only the rules given "
    "with the question. If they don't contain the answer, say that you don't know. Mention "
    "the section and the page you used."
)


class StandInPipeline(BasePipeline):
    def __init__(self, retriever: BaseRetriever, llm: LLMClient, k: int | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k or settings.top_k

    def run(self, question: str, history: list[Message]) -> PipelineResult:
        # history is ignored, the test questions are all single turn
        chunks = self.retriever.retrieve(question, k=self.k)
        context = "\n\n".join(f"[{c.section_path}, page {c.page}]\n{c.text}" for c in chunks)
        prompt = f"Rules:\n\n{context}\n\nQuestion: {question}"
        answer = self.llm.generate(prompt, system=SYSTEM)
        return PipelineResult(
            answer=answer, route="rules", standalone_question=question, rule_chunks=chunks
        )
