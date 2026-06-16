"""Semantic search over source file chunks using FAISS + bge-small embeddings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss

from repo_time_machine.ingestion.code_loader import FileChunk
from repo_time_machine.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class CodeResult:
    file: str
    start_line: int
    end_line: int
    content: str
    language: str
    score: float = 0.0

    def header(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line} ({self.language})"


class CodeRetriever:
    """
    Embeds source file chunks and answers semantic queries via FAISS.

    index_dir: directory where the FAISS index and metadata are persisted.
    """

    INDEX_FILE = "code.faiss"
    META_FILE = "code_meta.json"

    def __init__(self, index_dir: Path, embedder: Embedder):
        self.index_dir = index_dir
        self.embedder = embedder
        self._index: faiss.IndexFlatIP | None = None
        self._meta: list[dict] = []

    def build(self, chunks: list[FileChunk]) -> int:
        """Embed all chunks, build the FAISS index, and persist to disk."""
        if not chunks:
            logger.warning("No chunks to index")
            return 0

        texts = [self._chunk_to_text(c) for c in chunks]
        vectors = self.embedder.embed(texts)

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

        self._meta = [
            {
                "file": c.file,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
                "language": c.language,
            }
            for c in chunks
        ]

        self._save()
        logger.info("Indexed %d code chunks", len(chunks))
        return len(chunks)

    def load(self) -> bool:
        """Load a previously persisted index from disk. Returns True on success."""
        idx_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)
        logger.info("Loaded code index: %d vectors", self._index.ntotal)
        return True

    def query(self, question: str, top_k: int = 5) -> list[CodeResult]:
        """Return the top-k most relevant code chunks for the question."""
        if self._index is None or self._index.ntotal == 0:
            return []
        q_vec = self.embedder.embed([question])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)
        results: list[CodeResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = self._meta[idx]
            results.append(
                CodeResult(
                    file=m["file"],
                    start_line=m["start_line"],
                    end_line=m["end_line"],
                    content=m["content"],
                    language=m["language"],
                    score=float(score),
                )
            )
        return results

    def _save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))
        with open(self.index_dir / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False)

    @staticmethod
    def _chunk_to_text(chunk: FileChunk) -> str:
        return f"{chunk.file} ({chunk.language})\n{chunk.content}"
