"""Shared embedding model — loaded once, used by all retrievers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384  # bge-small output dimension


class Embedder:
    """
    Thin wrapper around sentence-transformers so the rest of the codebase
    doesn't need to import or configure it directly.

    The model is loaded lazily on first call to embed().
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, D) float32 numpy array of embeddings."""
        if not texts:
            return np.empty((0, _EMBED_DIM), dtype=np.float32)
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
        return _EMBED_DIM


def get_embedder(model_name: str = _DEFAULT_MODEL) -> Embedder:
    return Embedder(model_name)
