"""
Model factory.

Every LLM the app uses is created here, which gives three properties for free:

* one place to configure models/timeouts/retries,
* ``None`` when no key is configured -> callers fall back to deterministic paths,
* a test seam: :func:`override_chat_model` swaps in a scripted model so the whole
  LangGraph agent (tools, streaming, persistence) is tested offline in CI.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from backend.core.config import get_settings

Purpose = Literal["agent", "memory"]
_override: BaseChatModel | None = None


@lru_cache(maxsize=8)
def _gemini(model: str, temperature: float, max_tokens: int, api_key: str, timeout: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=api_key,
        max_retries=2,
        timeout=timeout,
    )


def get_chat_model(purpose: Purpose = "agent") -> BaseChatModel | None:
    if _override is not None:
        return _override
    s = get_settings()
    if not s.llm_enabled:
        return None
    if purpose == "memory":
        return _gemini(s.memory_model, 0.1, 512, s.google_api_key, s.llm_timeout_s)
    return _gemini(s.chat_model, 0.3, 2048, s.google_api_key, s.llm_timeout_s)


def llm_mode() -> str:
    if _override is not None:
        return "scripted"
    return "gemini" if get_settings().llm_enabled else "offline"


@contextmanager
def override_chat_model(model: BaseChatModel | None):
    """Temporarily replace every chat model (tests, demos)."""
    global _override
    previous, _override = _override, model
    try:
        yield model
    finally:
        _override = previous
