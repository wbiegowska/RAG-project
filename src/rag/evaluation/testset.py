"""Test set generation with ragas, from a selection of sections of the cleaned rulebook."""

from collections import defaultdict
from pathlib import Path

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
from ragas.testset.transforms import Parallel, apply_transforms, default_transforms_for_prechunked
from ragas.testset.transforms.base import Extractor, NodeFilter

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
    """One or two sections of every chapter (a bit more for the big ones). Using the whole
    book would take too many gemini calls for the free tier."""
    max_chars = max_chars or settings.chunk_size
    by_chapter = defaultdict(list)
    for section in split_into_sections(markdown):
        if MIN_CHARS <= len(section.text) <= max_chars:
            by_chapter[_chapter(section.path)].append(section)

    documents = []
    for sections in by_chapter.values():
        count = 1 + len(sections) // 8
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


def build_knowledge_graph(
    documents: list[Document], llm, embeddings, partial_path: Path
) -> KnowledgeGraph:
    """Summary, themes and entities for each section (the part with most gemini calls),
    then the links between similar sections used for the multi-hop questions.

    The graph is saved in partial_path after every step and a step only processes the
    sections that dont have its result yet. So when it stops in the middle (quota, crash)
    the next run continues from there instead of starting again."""
    if partial_path.exists():
        graph = KnowledgeGraph.load(partial_path)
        print(f"Continuing from {partial_path}")
    else:
        graph = KnowledgeGraph(
            nodes=[
                Node(
                    type=NodeType.CHUNK,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
                for doc in documents
            ]
        )

    builders = []
    for transform in _flatten(default_transforms_for_prechunked(llm, embeddings)):
        if isinstance(transform, NodeFilter):
            # it compares a chunk with the summary of its document, our sections dont have
            # a parent document so it would skip every node anyway
            continue
        if isinstance(transform, Extractor):
            _only_missing(transform)
            try:
                apply_transforms(graph, transform, run_config=RUN_CONFIG)
            finally:
                # saved even when some calls failed, ragas raises at the end of the step
                graph.save(partial_path)
        else:
            builders.append(transform)  # links between sections, run once everything is there

    for builder in builders:
        apply_transforms(graph, builder, run_config=RUN_CONFIG)
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


def _flatten(transforms) -> list:
    flat = []
    for transform in transforms:
        if isinstance(transform, Parallel):
            flat.extend(_flatten(transform.transformations))
        else:
            flat.append(transform)
    return flat


def _only_missing(extractor: Extractor) -> None:
    """Makes the extractor skip the sections that already have its result."""
    keep = extractor.filter_nodes
    name = extractor.property_name
    extractor.filter_nodes = lambda node: keep(node) and node.get_property(name) is None


def _chapter(path: str) -> str:
    # the battle round is most of the book, so each phase counts as a chapter
    parts = path.split(" > ")
    if parts[0] == "THE BATTLE ROUND" and len(parts) > 1:
        return " > ".join(parts[:2])
    return parts[0]
