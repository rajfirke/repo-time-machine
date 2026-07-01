"""Shared embedding model — loaded once, used by all retrievers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_FALLBACK_DIM = 384  # pre-load fallback; overridden once the model is loaded


class Embedder:
    """
    Thin wrapper around sentence-transformers so the rest of the codebase
    doesn't need to import or configure it directly.

    The model is loaded lazily on first call to embed().
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._detected_dim: int | None = None

    def _load(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._detected_dim = self._model.get_sentence_embedding_dimension()
        logger.info("Detected embedding dimension: %d", self._detected_dim)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, D) float32 numpy array of embeddings."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self._model is None:
            self._load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Return the embedding dimension, auto-detected from the model when available."""
        if self._detected_dim is not None:
            return self._detected_dim
        return _FALLBACK_DIM


def get_embedder(model_name: str = _DEFAULT_MODEL) -> Embedder:
    return Embedder(model_name)
