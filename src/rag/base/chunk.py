from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A piece of the rulebook, as stored in the index and returned by retrievers."""

    id: str
    text: str
    section_path: str  # e.g. "SHOOTING PHASE > MAKING ATTACKS > HIT ROLL"
    page: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
