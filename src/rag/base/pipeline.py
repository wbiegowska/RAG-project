from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag.base.chunk import Chunk
from rag.base.sql import SQLResult

# One message of the conversation: {"role": "user" | "assistant", "content": "..."}
Message = dict[str, str]


@dataclass
class PipelineResult:
    """Everything one chat turn produces. The evaluation reads these fields."""

    answer: str
    route: str  # "rules", "units" or "both"
    standalone_question: str = ""
    rule_chunks: list[Chunk] = field(default_factory=list)
    sql_result: SQLResult | None = None


class BasePipeline(ABC):
    """The full RAG system: question + history in, answer + sources out."""

    @abstractmethod
    def run(self, question: str, history: list[Message]) -> PipelineResult:
        """Answer one question, given the previous messages of the conversation."""
