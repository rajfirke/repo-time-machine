"""Build a cited answer from retrieved evidence and a local LLM."""

from dataclasses import dataclass, field


@dataclass
class Evidence:
    source: str      # "code", "commit", "issue"
    reference: str   # file path, commit SHA, or issue URL
    excerpt: str     # the relevant text snippet


@dataclass
class Answer:
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    suggested_action: str = ""

    def render(self) -> str:
        """Format the answer as a readable markdown string."""
        lines = [
            f"## Answer\n\n{self.summary}\n",
            "## Evidence\n",
        ]
        for ev in self.evidence:
            lines.append(f"**[{ev.source}]** `{ev.reference}`\n```\n{ev.excerpt}\n```\n")
        if self.timeline:
            lines.append("## Timeline\n")
            lines.extend(f"- {entry}" for entry in self.timeline)
            lines.append("")
        if self.suggested_action:
            lines.append(f"## Suggested Action\n\n{self.suggested_action}\n")
        return "\n".join(lines)


class AnswerBuilder:
    """
    Send evidence to the local LLM and return a structured Answer.

    model: Ollama model tag, e.g. "qwen2.5-coder:7b"
    """

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model

    def build(self, question: str, evidence: list[Evidence]) -> Answer:
        """Generate a cited Answer from the question and collected evidence."""
        # TODO: build prompt, call Ollama API, parse response into Answer
        raise NotImplementedError
