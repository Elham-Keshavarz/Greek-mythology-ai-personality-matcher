"""FAISS exact vector search with a content fingerprint for safe, reproducible rebuilds."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .text_chunker import Chunk

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "rag_index"


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def fingerprint(chunks: list[Chunk], model_name: str) -> str:
    payload = {"model": model_name, "chunks": [c.to_dict() for c in chunks]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class FaissStore:
    def __init__(self, index, chunks: list[Chunk], embedder: Embedder):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "FaissStore":
        import faiss
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        vectors = np.asarray(embedder.encode([c.text for c in chunks]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedding matrix must be shape (document_count, dimension)")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Embeddings must be finite")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, embedder)

    @classmethod
    def load_or_build(cls, chunks: list[Chunk], embedder: Embedder, index_dir: Path = INDEX_DIR,
                      rebuild: bool = False) -> "FaissStore":
        import faiss
        directory = Path(index_dir)
        wanted = fingerprint(chunks, embedder.model_name)
        meta_file = directory / "manifest.json"
        index_file = directory / "faiss.index"
        # Only open an index generated in this trusted local directory, not user-uploaded indexes.
        if not rebuild and meta_file.exists() and index_file.exists():
            manifest = json.loads(meta_file.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == wanted and manifest.get("count") == len(chunks):
                index = faiss.read_index(str(index_file))
                if index.ntotal == len(chunks):
                    return cls(index, chunks, embedder)
        store = cls.build(chunks, embedder)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(store.index, str(index_file))
        meta_file.write_text(json.dumps({
            "fingerprint": wanted, "count": len(chunks), "model": embedder.model_name
        }, indent=2), encoding="utf-8")
        return store

    def search(self, question: str, top_k: int = 4, min_score: float = 0.25) -> list[Hit]:
        import faiss
        if not question.strip() or not 1 <= top_k <= 20:
            raise ValueError("Question must not be blank and top_k must be between 1 and 20")
        vec = np.asarray(self.embedder.encode([question]), dtype=np.float32)
        if vec.shape != (1, self.index.d):
            raise ValueError("Question embedding shape differs from indexed vectors")
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, min(top_k, len(self.chunks)))
        return [Hit(self.chunks[int(i)], float(score)) for i, score in zip(indices[0], distances[0])
                if i >= 0 and score >= min_score]
