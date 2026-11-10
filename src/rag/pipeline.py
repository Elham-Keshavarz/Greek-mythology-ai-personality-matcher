"""Public entry point: load knowledge, chunk, embed, index and expose retrieval."""
from __future__ import annotations

from .embeddings import DEFAULT_MODEL, SentenceEmbedder
from .knowledge_base import load_documents
from .text_chunker import chunk_documents
from .vector_store import FaissStore


def build_pipeline(model_name: str = DEFAULT_MODEL, rebuild: bool = False) -> FaissStore:
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedder = SentenceEmbedder(model_name)
    return FaissStore.load_or_build(chunks, embedder, rebuild=rebuild)
