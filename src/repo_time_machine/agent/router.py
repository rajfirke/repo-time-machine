"""Classify a question and decide which retrievers to invoke."""

from enum import Enum


class QuestionType(str, Enum):
    CODE = "code"         # about current source: what does X do?
    HISTORY = "history"   # about past changes: why/when did X change?
    ISSUE = "issue"       # about intent/discussion: which PR explains X?
    MIXED = "mixed"       # needs all three


_HISTORY_KEYWORDS = {
    "why", "when", "changed", "introduced", "added", "removed",
    "commit", "history", "before", "regression", "broke",
}
_ISSUE_KEYWORDS = {
    "issue", "pr", "pull request", "discussion", "decision",
    "design", "explains", "reason", "motivation",
}


def classify(question: str) -> QuestionType:
    """
    Simple keyword-based classifier.

    Will be replaced with a lightweight LLM call once the stack is wired up.
    """
    tokens = set(question.lower().split())
    has_history = bool(tokens & _HISTORY_KEYWORDS)
    has_issue = bool(tokens & _ISSUE_KEYWORDS)

    if has_history and has_issue:
        return QuestionType.MIXED
    if has_history:
        return QuestionType.HISTORY
    if has_issue:
        return QuestionType.ISSUE
    return QuestionType.CODE
