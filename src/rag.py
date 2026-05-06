"""RAG index and retrieval built on sentence-transformers."""

from dataclasses import dataclass
from typing import List
import importlib

try:
    faiss = importlib.import_module("faiss")
except ModuleNotFoundError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

import numpy as np

from src.config import EMBEDDING_MODEL


@dataclass
class CodeChunk:
    text: str
    source: str


class CodeIndexer:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.vectors = None
        self.chunks: List[CodeChunk] = []

    def build(self, chunks: List[CodeChunk]) -> None:
        """Build or rebuild vector index from chunks."""
        filtered = [c for c in chunks if c.text.strip()]
        self.chunks = filtered
        if not filtered:
            self.index = None
            self.vectors = None
            return

        vectors = self.embedder.encode([c.text for c in filtered], convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype="float32")
        vectors = _normalize(vectors)
        self.vectors = vectors

        if faiss is not None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        else:
            self.index = None

    def search(self, query: str, top_k: int = 3) -> List[CodeChunk]:
        """Search similar chunks for a query."""
        if self.vectors is None or not self.chunks or not query.strip():
            return []

        k = max(1, min(top_k, len(self.chunks)))
        q = self.embedder.encode([query], convert_to_numpy=True)
        q = _normalize(np.asarray(q, dtype="float32"))

        if self.index is not None:
            _, indices = self.index.search(q, k)
            matched_indices = indices[0]
        else:
            scores = self.vectors @ q[0]
            matched_indices = np.argsort(scores)[::-1][:k]

        return [self.chunks[i] for i in matched_indices if 0 <= i < len(self.chunks)]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
