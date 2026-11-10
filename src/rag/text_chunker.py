"""Split larger documents into overlapping word-based passages for indexing."""
from __future__ import annotations

from dataclasses import dataclass

from .knowledge_base import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    text: str
    source_url: str
    source_type: str
    source_note: str

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def chunk_documents(documents: list[Document], max_words: int = 120, overlap_words: int = 25) -> list[Chunk]:
    if max_words < 1 or not (0 <= overlap_words < max_words):
        raise ValueError("Require max_words > overlap_words >= 0")
    chunks: list[Chunk] = []
    stride = max_words - overlap_words
    for document in documents:
        words = document.text.split()
        for part_number, start in enumerate(range(0, len(words), stride)):
            part = " ".join(words[start:start + max_words])
            if not part:
                break
            chunks.append(Chunk(
                id=f"{document.id}-chunk-{part_number}", document_id=document.id,
                title=document.title, text=part, source_url=document.source_url,
                source_type=document.source_type, source_note=document.source_note,
            ))
            if start + max_words >= len(words):
                break
    return chunks
