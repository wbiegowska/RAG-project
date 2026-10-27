"""Checks that the project installs and the shared interfaces import."""

from rag.base.chunk import Chunk
from rag.base.pipeline import PipelineResult
from rag.settings import settings


def test_settings_load():
    assert settings.top_k > 0
    assert settings.sqlite_path.name == "wahadb.sqlite"


def test_chunk_defaults():
    chunk = Chunk(id="1", text="Roll six D6", section_path="STRATAGEMS > GRENADE")
    assert chunk.metadata == {}
    assert chunk.score is None


def test_pipeline_result_defaults():
    result = PipelineResult(answer="42", route="rules")
    assert result.rule_chunks == []
    assert result.sql_result is None
