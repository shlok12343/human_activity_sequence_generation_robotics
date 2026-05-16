"""Shared Gemini client factory."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Project root is parent of ``soda_can_problem/`` (where ``.env`` lives).
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


def build_llm(
    model_name: str,
    temperature: float,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> ChatGoogleGenerativeAI:
    kwargs: dict = {"model": model_name, "temperature": temperature}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return ChatGoogleGenerativeAI(**kwargs)
