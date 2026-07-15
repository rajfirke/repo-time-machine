"""Fetch and search GitHub issues and PRs for context."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss

from repo_time_machine.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class IssueRecord:
    number: int
    title: str
    body: str
    url: str
    labels: list[str] = field(default_factory=list)
    is_pr: bool = False
    state: str = "open"


@dataclass
class IssueResult:
    issue: IssueRecord
    relevance: str
    score: float = 0.0


class IssueRetriever:
    """
    Fetches issues and PRs from a GitHub repo, embeds them, and ranks
    by relevance using semantic + keyword scoring.

    Requires GITHUB_TOKEN env var (free, read-only public scope is enough).
    """

    INDEX_FILE = "issues.faiss"
    META_FILE = "issues_meta.json"

    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3

    def __init__(self, index_dir: Path, embedder: Embedder, repo_slug: str = ""):
        self.index_dir = index_dir
        self.embedder = embedder
        self.repo_slug = repo_slug
        self._index: faiss.IndexFlatIP | None = None
        self._records: list[IssueRecord] = []

    def fetch_and_build(self, max_items: int = 200) -> int:
        """
        Pull issues and PRs from the GitHub API, embed them, and persist.

        Returns the number of items indexed.
        """
        if not self.repo_slug:
            logger.warning("No GitHub repo slug provided — skipping issue fetch")
            return 0

        token = os.environ.get("GITHUB_TOKEN", "")
        records = _fetch_from_github(self.repo_slug, token, max_items)
        if not records:
            logger.warning("No issues/PRs fetched from %s", self.repo_slug)
            return 0

        self._records = records
        texts = [_issue_to_text(r) for r in records]
        vectors = self.embedder.embed(texts)

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

        self._save()
        logger.info("Indexed %d issues/PRs from %s", len(records), self.repo_slug)
        return len(records)

    def build_from_records(self, records: list[IssueRecord]) -> int:
        """Build an index from pre-fetched records (useful for testing)."""
        if not records:
            return 0
        self._records = records
        texts = [_issue_to_text(r) for r in records]
        vectors = self.embedder.embed(texts)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._save()
        return len(records)

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
            records = [IssueRecord(**r) for r in raw]
            if index.ntotal != len(records):
                logger.error(
                    "Issue index mismatch: %d vectors vs %d records — re-run `rtm index`",
                    index.ntotal,
                    len(records),
                )
                return False
            self._index = index
            self._records = records
            logger.info("Loaded issue index: %d items", self._index.ntotal)
            return True
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError) as exc:
            logger.error("Failed to load issue index: %s", exc)
            return False

    def query(self, question: str, top_k: int = 5, min_score: float = 0.0) -> list[IssueResult]:
        """Return the top-k most relevant issues/PRs.

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
        scored: list[tuple[float, int]] = []
        for sem_score, idx in zip(scores_sem, indices):
            if idx < 0:
                continue
            rec = self._records[idx]
            kw = _keyword_score(q_tokens, rec)
            blended = self.SEMANTIC_WEIGHT * float(sem_score) + self.KEYWORD_WEIGHT * kw
            scored.append((blended, idx))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: list[IssueResult] = []
        filtered = 0
        for blended, idx in scored[:top_k]:
            if min_score > 0 and blended < min_score:
                filtered += 1
                continue
            rec = self._records[idx]
            tag = "PR" if rec.is_pr else "issue"
            results.append(
                IssueResult(
                    issue=rec,
                    relevance=f"matched {tag} #{rec.number}",
                    score=blended,
                )
            )
        if filtered:
            logger.info("Filtered %d issue result(s) below min_score=%.2f", filtered, min_score)
        return results

    @property
    def available(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def _save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))
        raw = [
            {
                "number": r.number,
                "title": r.title,
                "body": r.body,
                "url": r.url,
                "labels": r.labels,
                "is_pr": r.is_pr,
                "state": r.state,
            }
            for r in self._records
        ]
        with open(self.index_dir / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)


def _fetch_from_github(repo_slug: str, token: str, max_items: int) -> list[IssueRecord]:
    """Fetch issues and PRs from GitHub using PyGitHub."""
    try:
        from github import Github
    except ImportError:
        logger.error("PyGitHub is not installed — run: pip install PyGithub")
        return []

    gh = Github(token) if token else Github()

    try:
        repo = gh.get_repo(repo_slug)
    except Exception:
        logger.error("Could not access repo: %s", repo_slug)
        return []

    records: list[IssueRecord] = []

    issues = repo.get_issues(state="all", sort="updated", direction="desc")
    count = 0
    for item in issues:
        if count >= max_items:
            break
        body = (item.body or "")[:2000]
        labels = [lbl.name for lbl in item.labels]
        records.append(
            IssueRecord(
                number=item.number,
                title=item.title,
                body=body,
                url=item.html_url,
                labels=labels,
                is_pr=item.pull_request is not None,
                state=item.state,
            )
        )
        count += 1

    return records


def _issue_to_text(rec: IssueRecord) -> str:
    tag = "PR" if rec.is_pr else "Issue"
    labels_str = ", ".join(rec.labels) if rec.labels else "none"
    body_preview = rec.body[:500] if rec.body else ""
    return f"[{tag} #{rec.number}] {rec.title}\nLabels: {labels_str}\n{body_preview}"


_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t for t in _SPLIT_RE.split(text.lower()) if len(t) >= 2}


def _keyword_score(q_tokens: set[str], rec: IssueRecord) -> float:
    if not q_tokens:
        return 0.0
    issue_text = f"{rec.title} {rec.body} {' '.join(rec.labels)}"
    i_tokens = _tokenize(issue_text)
    if not i_tokens:
        return 0.0
    overlap = len(q_tokens & i_tokens)
    return overlap / len(q_tokens)
