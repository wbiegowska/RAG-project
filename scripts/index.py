"""Builds the vector index: cleaned rulebook -> chunks -> embeddings -> ChromaDB.

Usage: make index
"""

from rag.impl.chunker.markdown_chunker import chunk_markdown
from rag.impl.embedder.sbert import SbertEmbedder
from rag.impl.retriever.vector import VectorRetriever
from rag.settings import settings


def main() -> None:
    markdown = settings.rulebook_clean_md.read_text(encoding="utf-8")
    chunks = chunk_markdown(markdown)
    print(f"Split the rulebook into {len(chunks)} chunks, now embedding them")
    print("(the first time the embedding model is downloaded, it can take a minute)")

    retriever = VectorRetriever(SbertEmbedder())
    retriever.index(chunks)
    print(f"Indexed {retriever.collection.count()} chunks in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
