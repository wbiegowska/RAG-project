"""Test set generation with ragas, from a selection of sections of the cleaned rulebook."""

from collections import defaultdict

from langchain_core.documents import Document
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked

from rag.impl.chunker.markdown_chunker import split_into_sections
from rag.settings import settings

# sections shorter than this have nothing to ask about, and longer than chunk_size they
# would be split in several chunks, then the reference text wouldnt match one of our chunks
MIN_CHARS = 300

LLM_CONTEXT = (
    "The questions are about the Warhammer 40,000 Core Rules (10th edition). "
    "Write them in English, the way a player would ask them."
)
PERSONAS = [
    Persona(
        name="New player",
        role_description="Just started Warhammer 40,000 and is learning how the core rules "
        "work, asks simple and general questions.",
    ),
    Persona(
        name="Experienced player",
        role_description="Plays regularly and checks precise rules during games, like how "
        "stratagems, shooting, charges or terrain interact.",
    ),
]

# few workers because of the throttle (10 calls per minute), long timeout for the same reason
RUN_CONFIG = RunConfig(max_workers=4, timeout=300, max_retries=3, max_wait=60)


def select_sections(markdown: str, max_chars: int | None = None) -> list[Document]:
    """A few sections of every chapter, the big chapters get a few more. Generating from the
    whole book would take too many gemini calls for the free tier."""
    max_chars = max_chars or settings.chunk_size
    by_chapter = defaultdict(list)
    for section in split_into_sections(markdown):
        if MIN_CHARS <= len(section.text) <= max_chars:
            by_chapter[_chapter(section.path)].append(section)

    documents = []
    for sections in by_chapter.values():
        count = 2 + len(sections) // 10
        step = max(1, len(sections) // count)  # spread over the chapter, not just the start
        for section in sections[::step][:count]:
            documents.append(
                Document(
                    # same format as what we embed, so the texts can be compared later
                    page_content=f"{section.path}\n\n{section.text}",
                    metadata={"section_path": section.path, "page": section.start_page},
                )
            )
    return documents


def build_knowledge_graph(documents: list[Document], llm, embeddings) -> KnowledgeGraph:
    """Summaries, themes and entities for each section (the part with most gemini calls),
    then links between similar sections, which are used for the multi-hop questions."""
    nodes = [
        Node(
            type=NodeType.CHUNK,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata},
        )
        for doc in documents
    ]
    graph = KnowledgeGraph(nodes=nodes)
    transforms = default_transforms_for_prechunked(llm=llm, embedding_model=embeddings)
    apply_transforms(graph, transforms, run_config=RUN_CONFIG)
    return graph


def generate_testset(graph: KnowledgeGraph, llm, embeddings, size: int) -> list[dict]:
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=graph,
        persona_list=PERSONAS,
        llm_context=LLM_CONTEXT,
    )
    distribution = query_distribution(llm, graph)
    if not distribution:
        raise RuntimeError("The knowledge graph has no usable sections, rebuild it (rebuild=1)")
    testset = generator.generate(
        testset_size=size,
        query_distribution=distribution,
        num_personas=len(PERSONAS),
        run_config=RUN_CONFIG,
    )
    return testset.to_list()


def query_distribution(llm, graph: KnowledgeGraph) -> list[tuple]:
    """Half single-hop questions, a quarter of each multi-hop type (the 3 types of the
    course). A type is skipped if the graph has no sections to build it from."""
    wanted = [
        (SingleHopSpecificQuerySynthesizer(llm=llm, llm_context=LLM_CONTEXT), 0.5),
        (MultiHopSpecificQuerySynthesizer(llm=llm, llm_context=LLM_CONTEXT), 0.25),
        (MultiHopAbstractQuerySynthesizer(llm=llm, llm_context=LLM_CONTEXT), 0.25),
    ]
    available = [(synth, weight) for synth, weight in wanted if _can_build(synth, graph)]
    total = sum(weight for _, weight in available)
    return [(synth, weight / total) for synth, weight in available]


def _can_build(synthesizer, graph: KnowledgeGraph) -> bool:
    try:
        return bool(synthesizer.get_node_clusters(graph))
    except Exception:  # ragas raises when there is nothing to cluster
        return False


def _chapter(path: str) -> str:
    # the battle round is most of the book, so each phase counts as a chapter
    parts = path.split(" > ")
    if parts[0] == "THE BATTLE ROUND" and len(parts) > 1:
        return " > ".join(parts[:2])
    return parts[0]
