"""Tests for the agent layer: router, answer builder, LLM fallback, and pipeline."""

from unittest.mock import patch

from repo_time_machine.agent.answer import (
    Answer,
    AnswerBuilder,
    Evidence,
    _build_timeline,
    _code_to_evidence,
    _fallback_answer,
    _history_to_evidence,
    _split_llm_response,
)
from repo_time_machine.agent.llm import LLMResponse
from repo_time_machine.agent.router import (
    QuestionType,
    classify,
    should_search_code,
    should_search_history,
    should_search_issues,
)
from repo_time_machine.ingestion.git_history import CommitRecord
from repo_time_machine.retrieval.code_retriever import CodeResult
from repo_time_machine.retrieval.history_retriever import HistoryResult

# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestClassify:
    def test_code_question(self):
        assert classify("What does the validate function do?") == QuestionType.CODE

    def test_history_question(self):
        assert classify("Why was the retry logic added?") == QuestionType.HISTORY

    def test_issue_question(self):
        assert classify("Which issue explains the design decision?") == QuestionType.ISSUE

    def test_mixed_question(self):
        q = "Why was this function changed and what issue explains it?"
        assert classify(q) == QuestionType.MIXED

    def test_default_is_code(self):
        assert classify("hello world") == QuestionType.CODE

    def test_phrase_detection(self):
        assert classify("How does the parser work?") == QuestionType.CODE
        assert classify("What changed in utils.py?") == QuestionType.HISTORY


class TestShouldSearch:
    def test_code_searches_code(self):
        assert should_search_code(QuestionType.CODE)
        assert not should_search_history(QuestionType.CODE)

    def test_history_searches_history(self):
        assert should_search_history(QuestionType.HISTORY)
        assert not should_search_code(QuestionType.HISTORY)

    def test_mixed_searches_all(self):
        assert should_search_code(QuestionType.MIXED)
        assert should_search_history(QuestionType.MIXED)
        assert should_search_issues(QuestionType.MIXED)


# ---------------------------------------------------------------------------
# Evidence conversion tests
# ---------------------------------------------------------------------------


def _sample_code_results():
    return [
        CodeResult(
            file="src/main.py",
            start_line=1,
            end_line=20,
            content="def hello():\n    print('hello')\n",
            language="python",
            score=0.9,
        ),
    ]


def _sample_hist_results():
    commit = CommitRecord(
        sha="aaa11122",
        author="Alice",
        date="2025-01-15 10:00:00",
        message="Add input validation",
        files_changed=["src/utils.py"],
        diff_summary="M src/utils.py",
    )
    return [HistoryResult(commit=commit, relevance="strong semantic match", score=0.8)]


class TestEvidenceConversion:
    def test_code_to_evidence(self):
        results = _code_to_evidence(_sample_code_results())
        assert len(results) == 1
        assert results[0].source == "code"
        assert "main.py" in results[0].reference

    def test_history_to_evidence(self):
        results = _history_to_evidence(_sample_hist_results())
        assert len(results) == 1
        assert results[0].source == "commit"
        assert "aaa11122" in results[0].reference

    def test_build_timeline(self):
        timeline = _build_timeline(_sample_hist_results())
        assert len(timeline) == 1
        assert "aaa11122" in timeline[0]
        assert "2025-01-15" in timeline[0]


# ---------------------------------------------------------------------------
# Answer rendering tests
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_render_with_evidence(self):
        answer = Answer(
            summary="The function exists.",
            evidence=[Evidence(source="code", reference="main.py:1-10", excerpt="def hello():")],
            timeline=["2025-01-15 — added"],
            suggested_action="Read the tests.",
        )
        rendered = answer.render()
        assert "## Answer" in rendered
        assert "## Evidence" in rendered
        assert "## Timeline" in rendered
        assert "## Suggested Action" in rendered

    def test_render_empty(self):
        answer = Answer(summary="Nothing found.")
        rendered = answer.render()
        assert "Nothing found." in rendered
        assert "Evidence" not in rendered


# ---------------------------------------------------------------------------
# LLM response splitting
# ---------------------------------------------------------------------------


class TestSplitLLMResponse:
    def test_splits_on_suggested_action(self):
        text = "The code does X.\n\nSuggested next action:\nAdd tests for Y."
        summary, action = _split_llm_response(text)
        assert "does X" in summary
        assert "Add tests" in action

    def test_no_action_marker(self):
        text = "The code simply returns true."
        summary, action = _split_llm_response(text)
        assert summary == text
        assert action == ""

    def test_no_false_positive_on_narrative_text(self):
        text = "The suggested action was to refactor the parser module for better performance."
        summary, action = _split_llm_response(text)
        assert summary == text
        assert action == ""

    def test_no_false_positive_on_mid_sentence_next_step(self):
        text = "A good next step would be adding validation to the input handler."
        summary, action = _split_llm_response(text)
        assert summary == text
        assert action == ""

    def test_splits_on_markdown_header(self):
        text = "The function validates input.\n\n## Suggested Action\nAdd edge case tests."
        summary, action = _split_llm_response(text)
        assert "validates input" in summary
        assert "edge case" in action

    def test_splits_on_numbered_action(self):
        text = "Answer here.\n\n2. Suggested next action:\nRefactor the module."
        summary, action = _split_llm_response(text)
        assert "Answer here" in summary
        assert "Refactor" in action

    def test_splits_on_next_step_header(self):
        text = "The bug was introduced in commit abc.\n\nNext step:\nCheck related tests."
        summary, action = _split_llm_response(text)
        assert "bug was introduced" in summary
        assert "Check related" in action


# ---------------------------------------------------------------------------
# Fallback answer (no LLM) tests
# ---------------------------------------------------------------------------


class TestFallbackAnswer:
    def test_includes_evidence_summary(self):
        evidence = [
            Evidence(source="code", reference="a.py:1-10", excerpt="x"),
            Evidence(source="commit", reference="abc123", excerpt="y"),
        ]
        answer = _fallback_answer("why?", evidence, ["2025 — change"])
        assert "code section" in answer.summary
        assert "commit" in answer.summary
        assert not answer.used_llm
        assert len(answer.timeline) == 1

    def test_suggests_starting_ollama(self):
        answer = _fallback_answer("why?", [], [])
        assert "Ollama" in answer.suggested_action


# ---------------------------------------------------------------------------
# AnswerBuilder with mocked LLM
# ---------------------------------------------------------------------------


class TestAnswerBuilder:
    def test_with_llm(self):
        builder = AnswerBuilder(model="test")
        fake_resp = LLMResponse(
            text=(
                "The validation was added for safety.\n\n"
                "Suggested next action:\nAdd edge case tests."
            ),
            model="test",
            used_llm=True,
        )
        with patch("repo_time_machine.agent.answer.generate", return_value=fake_resp):
            answer = builder.build(
                "Why was validation added?",
                _sample_code_results(),
                _sample_hist_results(),
            )
        assert answer.used_llm
        assert "validation" in answer.summary.lower() or "safety" in answer.summary.lower()
        assert len(answer.evidence) > 0
        assert len(answer.timeline) > 0

    def test_without_llm(self):
        builder = AnswerBuilder(model="test")
        fake_resp = LLMResponse(text="", model="test", used_llm=False)
        with patch("repo_time_machine.agent.answer.generate", return_value=fake_resp):
            answer = builder.build("Why?", _sample_code_results(), _sample_hist_results())
        assert not answer.used_llm
        assert "Ollama" in answer.suggested_action

    def test_skip_llm_returns_raw_evidence(self):
        builder = AnswerBuilder(model="test")
        answer = builder.build(
            "Why was validation added?",
            _sample_code_results(),
            _sample_hist_results(),
            skip_llm=True,
        )
        assert not answer.used_llm
        assert "code section" in answer.summary
        assert "commit" in answer.summary
        assert "--raw" in answer.suggested_action
        assert len(answer.evidence) > 0
        assert len(answer.timeline) > 0

    def test_skip_llm_never_calls_generate(self):
        builder = AnswerBuilder(model="test")
        with patch("repo_time_machine.agent.answer.generate") as mock_gen:
            builder.build(
                "Why?",
                _sample_code_results(),
                _sample_hist_results(),
                skip_llm=True,
            )
            mock_gen.assert_not_called()

    def test_empty_evidence(self):
        builder = AnswerBuilder(model="test")
        answer = builder.build("Why?", [], [])
        assert "No relevant evidence" in answer.summary
