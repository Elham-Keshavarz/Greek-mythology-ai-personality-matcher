"""Local embedding model. First use downloads weights; later calls use the local cache."""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty list")
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                    show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
