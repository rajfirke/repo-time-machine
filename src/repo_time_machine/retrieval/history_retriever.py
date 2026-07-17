"""Search commit history and diffs for context relevant to a question."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import faiss

from repo_time_machine.ingestion.git_history import CommitRecord
from repo_time_machine.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class HistoryResult:
    commit: CommitRecord
    relevance: str  # short explanation of why this matched
    score: float = 0.0


class HistoryRetriever:
    """
    Finds commits most related to a question using two signals:

    1. Keyword overlap between the question and commit messages/file paths.
    2. Cosine similarity of embeddings (commit text vs question).

    The final score is a weighted blend: 0.7 * semantic + 0.3 * keyword.
    """

    INDEX_FILE = "commits.faiss"
    META_FILE = "commits_meta.json"

    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3

    def __init__(self, index_dir: Path, embedder: Embedder):
        self.index_dir = index_dir
        self.embedder = embedder
        self._index: faiss.IndexFlatIP | None = None
        self._records: list[CommitRecord] = []

    def build(self, commits: list[CommitRecord]) -> int:
        """Embed commit texts, build FAISS index, and persist."""
        if not commits:
            logger.warning("No commits to index")
            return 0

        self._records = commits
        texts = [self._commit_to_text(c) for c in commits]
        vectors = self.embedder.embed(texts)

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

        self._save()
        logger.info("Indexed %d commits", len(commits))
        return len(commits)

    def load(self) -> bool:
        """Load a previously persisted index from disk."""
        idx_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return False
        try:
            index = faiss.read_index(str(idx_path))
            with open(meta_path, encoding="utf-8") as f:
                raw = json.load(f)
            records = [CommitRecord(**r) for r in raw]
            if index.ntotal != len(records):
                logger.error(
                    "Commit index mismatch: %d vectors vs %d records — re-run `rtm index`",
                    index.ntotal,
                    len(records),
                )
                return False
            self._index = index
            self._records = records
            logger.info("Loaded commit index: %d vectors", self._index.ntotal)
            return True
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError) as exc:
            logger.error("Failed to load commit index: %s", exc)
            return False

    def query(self, question: str, top_k: int = 5, min_score: float = 0.0) -> list[HistoryResult]:
        """Return the top-k most relevant commits, blending semantic + keyword.

        Results with a blended score below *min_score* are dropped.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        if top_k < 1:
            return []

        q_vec = self.embedder.embed([question])
        n = self._index.ntotal
        scores_sem, indices = self._index.search(q_vec, n)
        scores_sem = scores_sem[0]
        indices = indices[0]

        q_tokens = _tokenize(question)
        scored: list[tuple[float, int, float, float]] = []
        for sem_score, idx in zip(scores_sem, indices):
            if idx < 0:
                continue
            rec = self._records[idx]
            kw_score = _keyword_score(q_tokens, rec)
            blended = self.SEMANTIC_WEIGHT * float(sem_score) + self.KEYWORD_WEIGHT * kw_score
            scored.append((blended, idx, float(sem_score), kw_score))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: list[HistoryResult] = []
        filtered = 0
        for blended, idx, sem, kw in scored[:top_k]:
            if min_score > 0 and blended < min_score:
                filtered += 1
                continue
            rec = self._records[idx]
            relevance = _explain(sem, kw)
            results.append(HistoryResult(commit=rec, relevance=relevance, score=blended))
        if filtered:
            logger.info("Filtered %d history result(s) below min_score=%.2f", filtered, min_score)
        return results

    def timeline_for_file(self, relative_path: str) -> list[CommitRecord]:
        """Return commits that touched a specific file.

        Accepts exact relative paths, bare filenames, or ``./``-prefixed paths.
        """
        normalized = relative_path.lstrip("./")
        return [
            c
            for c in self._records
            if any(f == normalized or f.endswith("/" + normalized) for f in c.files_changed)
        ]

    def _save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))
        raw = [
            {
                "sha": c.sha,
                "author": c.author,
                "date": c.date,
                "message": c.message,
                "files_changed": c.files_changed,
                "diff_summary": c.diff_summary,
                "is_merge": c.is_merge,
            }
            for c in self._records
        ]
        with open(self.index_dir / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)

    @staticmethod
    def _commit_to_text(commit: CommitRecord) -> str:
        files = ", ".join(commit.files_changed[:10]) if commit.files_changed else ""
        return f"{commit.message}\nFiles: {files}"


_SPLIT_RE = re.compile(r"\W+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t for t in _SPLIT_RE.split(text.lower()) if len(t) >= 2}


def _keyword_score(q_tokens: set[str], commit: CommitRecord) -> float:
    """Jaccard-like keyword overlap between question tokens and commit text."""
    if not q_tokens:
        return 0.0
    commit_text = f"{commit.message} {' '.join(commit.files_changed)}"
    c_tokens = _tokenize(commit_text)
    if not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return overlap / len(q_tokens)


def _explain(semantic: float, keyword: float) -> str:
    parts: list[str] = []
    if semantic > 0.5:
        parts.append("strong semantic match")
    elif semantic > 0.3:
        parts.append("moderate semantic match")
    if keyword > 0.4:
        parts.append("keyword overlap")
    return ", ".join(parts) if parts else "weak match"
