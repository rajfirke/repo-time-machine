"""Build a cited answer from retrieved evidence, optionally using a local LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from repo_time_machine.agent.llm import generate
from repo_time_machine.retrieval.code_retriever import CodeResult
from repo_time_machine.retrieval.history_retriever import HistoryResult
from repo_time_machine.retrieval.issue_retriever import IssueResult


@dataclass
class Evidence:
    source: str  # "code", "commit", "issue"
    reference: str  # file path, commit SHA, or issue URL
    excerpt: str  # the relevant text snippet


@dataclass
class Answer:
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    suggested_action: str = ""
    used_llm: bool = False

    def render(self) -> str:
        """Format the answer as a readable markdown string."""
        lines = [f"## Answer\n\n{self.summary}\n"]

        if self.evidence:
            lines.append("## Evidence\n")
            for ev in self.evidence:
                lines.append(f"**[{ev.source}]** `{ev.reference}`\n```\n{ev.excerpt}\n```\n")

        if self.timeline:
            lines.append("## Timeline\n")
            lines.extend(f"- {entry}" for entry in self.timeline)
            lines.append("")

        if self.suggested_action:
            lines.append(f"## Suggested Action\n\n{self.suggested_action}\n")

        return "\n".join(lines)


def _code_to_evidence(results: list[CodeResult]) -> list[Evidence]:
    items: list[Evidence] = []
    for cr in results:
        snippet = cr.content.strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "\n..."
        items.append(
            Evidence(
                source="code",
                reference=cr.header(),
                excerpt=snippet,
            )
        )
    return items


def _history_to_evidence(results: list[HistoryResult]) -> list[Evidence]:
    items: list[Evidence] = []
    for hr in results:
        c = hr.commit
        excerpt_parts = [c.message]
        if c.files_changed:
            excerpt_parts.append(f"Files: {', '.join(c.files_changed[:8])}")
        if c.diff_summary:
            summary_lines = c.diff_summary.strip().split("\n")[:6]
            excerpt_parts.append("\n".join(summary_lines))
        items.append(
            Evidence(
                source="commit",
                reference=f"{c.short_sha} ({c.date})",
                excerpt="\n".join(excerpt_parts),
            )
        )
    return items


def _issue_to_evidence(results: list[IssueResult]) -> list[Evidence]:
    items: list[Evidence] = []
    for ir in results:
        rec = ir.issue
        tag = "PR" if rec.is_pr else "issue"
        body_preview = rec.body.strip()[:400] if rec.body else "(no description)"
        excerpt = f"[{tag}] {rec.title}\n{body_preview}"
        if rec.labels:
            excerpt += f"\nLabels: {', '.join(rec.labels)}"
        items.append(
            Evidence(
                source="issue",
                reference=f"#{rec.number} ({rec.url})",
                excerpt=excerpt,
            )
        )
    return items


def _build_timeline(hist_results: list[HistoryResult]) -> list[str]:
    seen = set()
    entries: list[str] = []
    for hr in sorted(hist_results, key=lambda r: r.commit.date):
        c = hr.commit
        if c.sha in seen:
            continue
        seen.add(c.sha)
        first_line = c.message.strip().split("\n", 1)[0]
        entries.append(f"`{c.short_sha}` {c.date} — {first_line}")
    return entries


SYSTEM_PROMPT = """\
You are Repo Time Machine, a code historian. You answer questions about a \
codebase using evidence from source files, git commits, and GitHub issues/PRs.

Rules:
- Cite every claim with [code], [commit], or [issue] references from the evidence below.
- If the evidence is insufficient, say so honestly.
- End with a concrete suggested next action.
- Be concise. No filler."""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Evidence:
{evidence_block}

Provide:
1. A direct answer (2-4 sentences) citing the evidence.
2. A suggested next action (1-2 sentences)."""


def _format_evidence_block(evidence: list[Evidence]) -> str:
    parts: list[str] = []
    for i, ev in enumerate(evidence, 1):
        parts.append(f"[{ev.source} #{i}] {ev.reference}\n{ev.excerpt}")
    return "\n\n".join(parts)


class AnswerBuilder:
    """
    Collect evidence from retrievers and produce a structured Answer.

    If Ollama is running, the LLM synthesizes a cited answer.
    If not, a structured evidence-only answer is returned.
    """

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model

    def build(
        self,
        question: str,
        code_results: list[CodeResult],
        hist_results: list[HistoryResult],
        issue_results: list[IssueResult] | None = None,
        skip_llm: bool = False,
    ) -> Answer:
        code_evidence = _code_to_evidence(code_results)
        hist_evidence = _history_to_evidence(hist_results)
        issue_evidence = _issue_to_evidence(issue_results) if issue_results else []
        all_evidence = code_evidence + hist_evidence + issue_evidence
        timeline = _build_timeline(hist_results)

        if not all_evidence:
            return Answer(
                summary="No relevant evidence found in the indexed repository.",
                used_llm=False,
            )

        if skip_llm:
            return _fallback_answer(question, all_evidence, timeline, raw=True)

        evidence_block = _format_evidence_block(all_evidence)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{USER_PROMPT_TEMPLATE.format(question=question, evidence_block=evidence_block)}"
        )

        llm_resp = generate(prompt, model=self.model)

        if llm_resp.used_llm and llm_resp.text:
            summary, action = _split_llm_response(llm_resp.text)
            return Answer(
                summary=summary,
                evidence=all_evidence,
                timeline=timeline,
                suggested_action=action,
                used_llm=True,
            )

        return _fallback_answer(question, all_evidence, timeline)


def _split_llm_response(text: str) -> tuple[str, str]:
    """Split LLM output into (summary, suggested_action)."""
    lower = text.lower()
    for marker in ["suggested next action", "suggested action", "next action", "next step"]:
        idx = lower.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline == -1:
                newline = idx + len(marker)
            summary = text[:idx].strip().rstrip(":#-")
            action = text[newline:].strip().lstrip(":#- ")
            return summary, action
    return text.strip(), ""


def _fallback_answer(
    question: str,
    evidence: list[Evidence],
    timeline: list[str],
    raw: bool = False,
) -> Answer:
    """Build an answer without an LLM — just structured evidence."""
    code_refs = [e for e in evidence if e.source == "code"]
    commit_refs = [e for e in evidence if e.source == "commit"]
    issue_refs = [e for e in evidence if e.source == "issue"]

    parts: list[str] = []
    if code_refs:
        parts.append(
            f"Found {len(code_refs)} relevant code section(s): "
            + ", ".join(f"`{e.reference}`" for e in code_refs[:3])
            + "."
        )
    if commit_refs:
        parts.append(
            f"Found {len(commit_refs)} relevant commit(s): "
            + ", ".join(f"`{e.reference}`" for e in commit_refs[:3])
            + "."
        )
    if issue_refs:
        parts.append(
            f"Found {len(issue_refs)} relevant issue(s)/PR(s): "
            + ", ".join(f"`{e.reference}`" for e in issue_refs[:3])
            + "."
        )

    if raw:
        action = "Run without `--raw` to get an LLM-synthesized answer."
    else:
        parts.append(
            "(Ollama was not reachable — showing raw evidence."
            " Start Ollama for synthesized answers.)"
        )
        action = "Start Ollama (`ollama serve`) and re-run this query for a synthesized answer."

    return Answer(
        summary="\n\n".join(parts),
        evidence=evidence,
        timeline=timeline,
        suggested_action=action,
        used_llm=False,
    )
