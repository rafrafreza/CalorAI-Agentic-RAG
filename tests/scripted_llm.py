"""
A scripted chat model for testing the real LangGraph agent offline.

Each step is either an ``AIMessage`` or a callable ``(messages) -> AIMessage`` so a
step can react to earlier tool results (e.g. read the meal id returned by get_meals).
Text replies are streamed word by word, exercising the SSE token path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


def call(name: str, args: dict, call_id: str | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{name}"}])


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def last_tool_result(messages: list[BaseMessage], name: str) -> dict:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == name:
            return json.loads(m.content)
    raise AssertionError(f"no {name} result in context")


class ScriptedChatModel(BaseChatModel):
    steps: list[Any]
    cursor: int = 0
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    fail_with: type[Exception] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        if self.fail_with:
            raise self.fail_with("scripted failure")
        self.seen.append(list(messages))
        step = self.steps[self.cursor]
        self.cursor += 1
        return step(messages) if callable(step) else step

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        if msg.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                        for i, tc in enumerate(msg.tool_calls)
                    ],
                )
            )
            return
        words = msg.content.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
