"""Tests for the evaluation code (no gemini and no real embedding model)."""

import asyncio

import pytest

pytest.importorskip("ragas")  # the eval group is not installed for everyone

from langchain_core.prompt_values import StringPromptValue  # noqa: E402

from rag.base.embedder import Embedder  # noqa: E402
from rag.base.llm import LLMClient  # noqa: E402
from rag.evaluation.ragas_adapters import EmbedderRagasEmbeddings, GeminiRagasLLM  # noqa: E402
from rag.evaluation.testset import select_sections  # noqa: E402

SENTENCE = "This rule explains what happens when a unit does something on the battlefield. "
SAMPLE = f"""<!-- page: 10 -->

# Core Rules

## THE BATTLE ROUND

### SHOOTING PHASE

{SENTENCE * 6}

#### MAKING ATTACKS

{SENTENCE * 8}

<!-- page: 41 -->

## STRATAGEMS

Too short.

##### GRENADE

{SENTENCE * 5}
"""


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return f"answer {len(self.prompts)}"

    def stream(self, prompt: str, system: str | None = None):
        yield self.generate(prompt, system)


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append("documents")
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append("query")
        return [float(len(text)), 1.0]


def test_select_sections_keeps_the_path_and_the_page():
    documents = select_sections(SAMPLE, max_chars=1200)
    paths = [d.metadata["section_path"] for d in documents]
    # STRATAGEMS is too short, the others are kept
    assert paths == [
        "THE BATTLE ROUND > SHOOTING PHASE",
        "THE BATTLE ROUND > SHOOTING PHASE > MAKING ATTACKS",
        "STRATAGEMS > GRENADE",
    ]
    assert documents[2].metadata["page"] == 41
    assert documents[2].page_content.startswith("STRATAGEMS > GRENADE\n\nThis rule")


def test_select_sections_skips_sections_longer_than_a_chunk():
    documents = select_sections(SAMPLE, max_chars=450)
    assert [d.metadata["section_path"] for d in documents] == ["STRATAGEMS > GRENADE"]


def test_ragas_llm_uses_our_client():
    fake = FakeLLM()
    llm = GeminiRagasLLM(fake)
    result = llm.generate_text(StringPromptValue(text="hello"), n=2)
    assert [g.text for g in result.generations[0]] == ["answer 1", "answer 2"]
    assert fake.prompts == ["hello", "hello"]
    result = asyncio.run(llm.agenerate_text(StringPromptValue(text="again")))
    assert result.generations[0][0].text == "answer 3"


def test_ragas_embeddings_dont_use_the_query_instruction():
    fake = FakeEmbedder()
    embeddings = EmbedderRagasEmbeddings(fake)
    assert embeddings.embed_query("abc") == [3.0, 1.0]
    assert asyncio.run(embeddings.aembed_documents(["ab"])) == [[2.0, 1.0]]
    assert fake.calls == ["documents", "documents"]
