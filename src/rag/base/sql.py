from dataclasses import dataclass, field
from typing import Any


@dataclass
class SQLResult:
    """Output of the text-to-SQL retriever for one question."""

    question: str
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str | None = None
