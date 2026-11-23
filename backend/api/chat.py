"""
Chat endpoints.

    POST /api/chat          multipart (message, session_id?, user_id?, image?) -> JSON
    POST /api/chat/stream   same input -> Server-Sent Events

Both are driven by the same async generator (:func:`run_turn`), so streaming and
non-streaming responses can never diverge. Event types:

    start      {session_id, mode}
    step       {node, label}                    a graph node started
    vision     {items, confidence, ...}         photo analysis result
    retrieval  {groundings, memories, profile}  speculative RAG results (with scores)
    tool_call  {name, args}
    tool_result{tool, result}
    meal       {action, meal}                   a meal was logged / updated / deleted
    token      {text}                           streamed answer tokens
    done       {response, latency_ms, day, trace, ...}
    error      {message}
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.errors import GraphRecursionError
from starlette.background import BackgroundTask

from backend.agent.graph import agent_graph
from backend.agent.memory_writer import extract_and_store
from backend.agent.nodes import message_text
from backend.api.uploads import save_image
from backend.core.config import get_settings
from backend.core.llm import llm_mode
from backend.core.telemetry import recorder
from backend.db.database import get_db
from backend.db.memory import get_conversation_history, save_message
from backend.models.schemas import ChatResponse
from backend.services.nutrition_log import day_summary, parse_day

log = logging.getLogger("calorai.api.chat")
router = APIRouter(prefix="/api", tags=["chat"])
MEAL_TOOLS = {"log_meal": "logged", "update_meal": "updated", "delete_meal": "deleted"}


def _history(user_id: str, session_id: str) -> list:
    turns = get_settings().history_turns
    with get_db() as conn:
        rows = get_conversation_history(conn, user_id, session_id, limit=turns * 2)
    out = []
    for r in rows:
        if r["role"] == "user":
            out.append(HumanMessage(content=r["content"]))
        elif r["role"] == "assistant":
            out.append(AIMessage(content=r["content"]))
    return out


async def run_turn(message: str, user_id: str, session_id: str, image_path: str | None = None) -> AsyncIterator[dict]:
    started = time.perf_counter()
    mode = llm_mode()
    state = {
        "messages": [*_history(user_id, session_id), HumanMessage(content=message or "(photo)")],
        "user_id": user_id,
        "session_id": session_id,
        "user_text": message,
        "image_path": image_path,
        "actions": [],
        "trace": [],
    }
    source = "image+text" if image_path and message else ("image" if image_path else "text")
    config = {
        "configurable": {
            "user_id": user_id,
            "session_id": session_id,
            "raw_input": message,
            "source": source,
            "image_path": image_path,
        },
        "recursion_limit": 25,
    }
    yield {"event": "start", "data": {"session_id": session_id, "mode": mode, "has_image": bool(image_path)}}

    final, streamed = "", ""
    meals_changed: list[dict] = []
    groundings: list[dict] = []
    memories: list[dict] = []
    trace: list[dict] = []
    vision = None
    try:
        async for stream_mode, chunk in agent_graph.astream(
            state, config, stream_mode=["custom", "updates", "messages"]
        ):
            if stream_mode == "custom":
                yield {"event": "step", "data": chunk}
            elif stream_mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "agent" and isinstance(msg, AIMessageChunk):
                    text = message_text(msg.content)
                    if text:
                        streamed += text
                        yield {"event": "token", "data": {"text": text}}
            else:
                for node, update in chunk.items():
                    if not update:
                        continue
                    trace.extend(update.get("trace", []))
                    if node == "ingest" and update.get("vision"):
                        vision = update["vision"]
                        yield {"event": "vision", "data": vision}
                    if node == "retrieve":
                        groundings, memories = update.get("groundings", []), update.get("memories_used", [])
                        yield {
                            "event": "retrieval",
                            "data": {"groundings": groundings, "memories": memories, "profile": update.get("profile")},
                        }
                    for m in update.get("messages", []):
                        if node == "agent" and isinstance(m, AIMessage):
                            for tc in m.tool_calls:
                                yield {"event": "tool_call", "data": {"name": tc["name"], "args": tc["args"]}}
                            if not m.tool_calls:
                                final = message_text(m.content)
                    for action in update.get("actions", []):
                        yield {"event": "tool_result", "data": action}
                        status = action["result"].get("status")
                        if action["tool"] in MEAL_TOOLS and status == MEAL_TOOLS[action["tool"]]:
                            change = {
                                "action": status,
                                "meal": action["result"]["meal"],
                                "previous": action["result"].get("previous"),
                            }
                            meals_changed.append(change)
                            yield {"event": "meal", "data": change}
    except GraphRecursionError:
        final = "I went around in circles on that one — could you rephrase it more simply?"
    except Exception as exc:  # never leave the client hanging
        log.exception("turn failed")
        yield {"event": "error", "data": {"message": f"{exc.__class__.__name__}: {exc}"}}
        final = final or "Something went wrong on my side. Please try again."

    final = (final or streamed).strip() or "Sorry, I couldn't process that. Could you rephrase?"
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    recorder.record(f"chat.{'image' if image_path else 'text'}", latency_ms)

    with get_db() as conn:
        save_message(conn, user_id, session_id, "user", message or "(photo)", {"image": bool(image_path)})
        save_message(
            conn,
            user_id,
            session_id,
            "assistant",
            final,
            {
                "meals_changed": meals_changed,
                "groundings": groundings,
                "memories_used": memories,
                "latency_ms": latency_ms,
                "mode": mode,
                "trace": trace,
                "vision": vision,
            },
        )
        day = day_summary(conn, user_id, parse_day("today"))

    yield {
        "event": "done",
        "data": {
            "response": final,
            "session_id": session_id,
            "mode": mode,
            "latency_ms": latency_ms,
            "meals_changed": meals_changed,
            "groundings": groundings,
            "memories_used": memories,
            "day": day,
            "trace": trace,
        },
    }


async def _parse_form(message: str, session_id: str | None, user_id: str, image: UploadFile | None):
    settings = get_settings()
    message = (message or "").strip()
    if len(message) > settings.max_message_chars:
        raise HTTPException(413, f"Message longer than {settings.max_message_chars} characters")
    image_path = await save_image(image) if image is not None and image.filename else None
    if not message and not image_path:
        raise HTTPException(400, "Send a message or a photo")
    user_id = (user_id or "default").strip()[:64] or "default"
    session_id = (session_id or uuid.uuid4().hex[:12]).strip()[:64]
    return message, session_id, user_id, image_path


@router.post("/chat", response_model=ChatResponse)
async def chat(
    background: BackgroundTasks,
    message: str = Form(""),
    session_id: str | None = Form(None),
    user_id: str = Form("default"),
    image: UploadFile | None = File(None),
):
    message, session_id, user_id, image_path = await _parse_form(message, session_id, user_id, image)
    done: dict = {}
    async for event in run_turn(message, user_id, session_id, image_path):
        if event["event"] == "done":
            done = event["data"]
    background.add_task(extract_and_store, user_id, message, done.get("response", ""))
    return ChatResponse(**done)


def _sse(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    message: str = Form(""),
    session_id: str | None = Form(None),
    user_id: str = Form("default"),
    image: UploadFile | None = File(None),
):
    message, session_id, user_id, image_path = await _parse_form(message, session_id, user_id, image)
    final: dict = {}

    async def events():
        async for event in run_turn(message, user_id, session_id, image_path):
            if event["event"] == "done":
                final.update(event["data"])
            yield _sse(event)

    async def after():
        await extract_and_store(user_id, message, final.get("response", ""))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        background=BackgroundTask(after),
    )
