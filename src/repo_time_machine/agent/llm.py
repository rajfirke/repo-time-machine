"""Talk to a local Ollama instance or fall back to a no-LLM summary."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:7b"


@dataclass
class LLMResponse:
    text: str
    model: str
    used_llm: bool


def generate(prompt: str, model: str = DEFAULT_MODEL) -> LLMResponse:
    """
    Send a prompt to Ollama and return the response.

    Falls back to a placeholder if Ollama is unreachable, so the rest
    of the pipeline keeps working even without a running model.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        return LLMResponse(
            text=body.get("response", "").strip(),
            model=model,
            used_llm=True,
        )
    except (requests.ConnectionError, requests.Timeout):
        logger.warning(
            "Ollama not reachable at %s — returning evidence-only answer", OLLAMA_URL
        )
        return LLMResponse(
            text="",
            model=model,
            used_llm=False,
        )
    except Exception:
        logger.exception("Unexpected error calling Ollama")
        return LLMResponse(text="", model=model, used_llm=False)
