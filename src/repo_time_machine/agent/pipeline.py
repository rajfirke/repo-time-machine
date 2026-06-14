"""End-to-end pipeline: question → classify → retrieve → answer."""

from __future__ import annotations

import logging
from pathlib import Path

from repo_time_machine.agent.answer import Answer, AnswerBuilder
from repo_time_machine.agent.router import (
    classify,
    should_search_code,
    should_search_history,
)
from repo_time_machine.retrieval.code_retriever import CodeResult, CodeRetriever
from repo_time_machine.retrieval.embeddings import get_embedder
from repo_time_machine.retrieval.history_retriever import HistoryResult, HistoryRetriever
from repo_time_machine.retrieval.store import ensure_rtm_dir, load_config

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full ask flow:

    1. Load config and indexes.
    2. Classify the question.
    3. Retrieve evidence from the right sources.
    4. Build a cited answer (with or without LLM).
    """

    def __init__(
        self,
        repo_path: str | Path,
        llm_model: str = "qwen2.5-coder:7b",
        top_k: int = 5,
    ):
        self.repo = Path(repo_path).resolve()
        self.top_k = top_k

        rtm_path = ensure_rtm_dir(self.repo)
        cfg = load_config(self.repo) or {}
        embed_model = cfg.get("model", "BAAI/bge-small-en-v1.5")
        embedder = get_embedder(embed_model)

        self.code_retriever = CodeRetriever(rtm_path, embedder)
        self.hist_retriever = HistoryRetriever(rtm_path, embedder)
        self.answer_builder = AnswerBuilder(model=llm_model)

        self._code_loaded = self.code_retriever.load()
        self._hist_loaded = self.hist_retriever.load()

    @property
    def ready(self) -> bool:
        return self._code_loaded and self._hist_loaded

    def ask(self, question: str) -> Answer:
        """Run the full pipeline and return a structured Answer."""
        qtype = classify(question)
        logger.info("Question classified as: %s", qtype.value)

        code_results: list[CodeResult] = []
        hist_results: list[HistoryResult] = []

        if should_search_code(qtype) and self._code_loaded:
            code_results = self.code_retriever.query(question, top_k=self.top_k)
            logger.info("Code retriever returned %d results", len(code_results))

        if should_search_history(qtype) and self._hist_loaded:
            hist_results = self.hist_retriever.query(question, top_k=self.top_k)
            logger.info("History retriever returned %d results", len(hist_results))

        if not code_results and not hist_results:
            if self._code_loaded:
                code_results = self.code_retriever.query(question, top_k=self.top_k)
            if self._hist_loaded:
                hist_results = self.hist_retriever.query(question, top_k=self.top_k)

        return self.answer_builder.build(question, code_results, hist_results)
