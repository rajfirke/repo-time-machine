"""Classify a question and decide which retrievers to invoke."""

from __future__ import annotations

from enum import StrEnum


class QuestionType(StrEnum):
    CODE = "code"  # about current source: what does X do?
    HISTORY = "history"  # about past changes: why/when did X change?
    ISSUE = "issue"  # about intent/discussion: which PR explains X?
    MIXED = "mixed"  # needs multiple sources


_HISTORY_KEYWORDS = {
    "why",
    "when",
    "changed",
    "introduced",
    "added",
    "removed",
    "commit",
    "history",
    "before",
    "regression",
    "broke",
    "breaking",
    "deprecated",
    "replaced",
    "refactor",
    "refactored",
    "modified",
    "timeline",
    "evolution",
    "previous",
    "originally",
}
_ISSUE_KEYWORDS = {
    "issue",
    "pr",
    "pull request",
    "discussion",
    "decision",
    "design",
    "explains",
    "reason",
    "motivation",
    "requested",
    "feature request",
    "bug report",
    "reported",
}
_CODE_KEYWORDS = {
    "function",
    "class",
    "method",
    "variable",
    "import",
    "module",
    "implementation",
    "signature",
    "parameter",
    "returns",
    "type",
    "how does",
    "what does",
    "where is",
    "defined",
    "called",
}


def classify(question: str) -> QuestionType:
    """
    Keyword-based classifier that determines which retrieval sources to use.

    Returns MIXED if the question touches multiple categories, which causes
    the pipeline to query all available retrievers.
    """
    tokens = set(question.lower().split())
    q_lower = question.lower()

    has_history = bool(tokens & _HISTORY_KEYWORDS) or any(
        phrase in q_lower for phrase in ("what changed", "who changed", "was added")
    )
    has_issue = bool(tokens & _ISSUE_KEYWORDS) or any(
        phrase in q_lower for phrase in ("pull request", "bug report", "feature request")
    )
    has_code = bool(tokens & _CODE_KEYWORDS) or any(
        phrase in q_lower for phrase in ("how does", "what does", "where is")
    )

    signals = sum([has_history, has_issue, has_code])
    if signals >= 2:
        return QuestionType.MIXED
    if has_history:
        return QuestionType.HISTORY
    if has_issue:
        return QuestionType.ISSUE
    return QuestionType.CODE


def should_search_code(qt: QuestionType) -> bool:
    return qt in (QuestionType.CODE, QuestionType.MIXED)


def should_search_history(qt: QuestionType) -> bool:
    return qt in (QuestionType.HISTORY, QuestionType.MIXED)


def should_search_issues(qt: QuestionType) -> bool:
    return qt in (QuestionType.ISSUE, QuestionType.MIXED)
