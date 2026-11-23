"""
Graph nodes. Each node has one job and records a timed span into ``state.trace``.

    ingest   - photo? run the vision model, merge its items with the caption (one meal)
    retrieve - memory RAG (pinned + relevant facts) and *speculative* nutrition retrieval:
               food mentions are grounded BEFORE the LLM runs, so the model usually
               logs in one tool round instead of lookup -> log -> reply
    agent    - Gemini with tools, or the deterministic offline engine (no key / outage)
    tools    - executes tool calls and records structured results as ``actions``
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolNode

from backend.agent import offline
from backend.agent.prompts import SYSTEM_PROMPT
from backend.agent.state import AgentState
from backend.agent.tools import ALL_TOOLS
from backend.core.llm import get_chat_model
from backend.core.telemetry import span
from backend.db.database import get_db
from backend.db.memory import get_all_memory, get_relevant_memories
from backend.rag.mentions import detect_meal_type
from backend.rag.profile import profile_from_memories
from backend.rag.service import Grounding, get_rag

log = logging.getLogger("calorai.agent")
MAX_TOOL_ROUNDS = 6
_tool_node = ToolNode(ALL_TOOLS)


def message_text(content) -> str:
    """Text of a message whose content may be a string or a list of content blocks (Gemini 3)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _emit(event: dict) -> None:
    with contextlib.suppress(Exception):  # not running under a streaming context
        get_stream_writer()(event)


def _uid(state: AgentState, config: RunnableConfig | None) -> str:
    return state.get("user_id") or (config or {}).get("configurable", {}).get("user_id") or "default"


# ── ingest ───────────────────────────────────────────────────────────────────


async def ingest(state: AgentState, config: RunnableConfig) -> dict:
    image_path = state.get("image_path")
    if not image_path:
        return {"trace": []}
    trace: list[dict] = []
    _emit({"type": "step", "node": "ingest", "label": "Reading your photo"})
    from backend.vision.processor import extract_food_from_image, summarize_for_agent

    with span("vision", trace) as info:
        if get_chat_model("agent") is None:
            info["skipped"] = "offline"
            return {"vision": {"error": "Photo analysis needs a Gemini API key."}, "trace": trace}
        try:
            result = await asyncio.to_thread(extract_food_from_image, image_path)
        except Exception as exc:
            log.warning("vision failed: %s", exc)
            info["error"] = str(exc)
            return {"vision": {"error": f"Couldn't analyse the photo ({exc.__class__.__name__})."}, "trace": trace}
        info["items"] = len(result.items)

    last = state["messages"][-1]
    merged = HumanMessage(content=summarize_for_agent(result, state.get("user_text", "")), id=last.id)
    return {"messages": [merged], "vision": result.model_dump(), "trace": trace}


# ── retrieve ─────────────────────────────────────────────────────────────────


def _render_memory(facts: list[dict]) -> str:
    if not facts:
        return "No stored preferences or goals yet."
    return "\n".join(f"- [{f['category']}] {f['key']}: {f['value']}" for f in facts)


def _render_groundings(groundings: list[Grounding]) -> str:
    if not groundings:
        return "(No food mentions detected in the latest message. Use lookup_nutrition if needed.)"
    return "\n".join(json.dumps(g.to_llm(), ensure_ascii=False) for g in groundings)


async def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    trace: list[dict] = []
    user_id = _uid(state, config)
    text = state.get("user_text", "")
    _emit({"type": "step", "node": "retrieve", "label": "Searching nutrition knowledge"})
    with span("retrieve", trace) as info:
        with get_db() as conn:
            memories = get_relevant_memories(conn, user_id, text)
            profile = profile_from_memories(get_all_memory(conn, user_id))
        rag = get_rag()

        def _ground() -> list[Grounding]:
            found = rag.ground_message(text, profile) if text else []
            vision = state.get("vision") or {}
            for item in vision.get("items", []):
                found.append(rag.ground(item["name"], item.get("quantity"), item.get("unit"), profile))
            return found

        groundings = await asyncio.to_thread(_ground)
        info.update(mentions=len(groundings), grounded=sum(g.usable for g in groundings), memories=len(memories))
    return {
        "memory_context": _render_memory(memories),
        "memories_used": [{k: m.get(k) for k in ("key", "value", "category", "pinned", "score")} for m in memories],
        "profile": profile.as_dict(),
        "groundings": [g.to_ui() for g in groundings],
        "grounding_context": _render_groundings(groundings),
        "meal_type_hint": detect_meal_type(text),
        "trace": trace,
    }


# ── agent ────────────────────────────────────────────────────────────────────


def _system_prompt(state: AgentState) -> SystemMessage:
    now = datetime.now()
    return SystemMessage(
        content=SYSTEM_PROMPT.format(
            weekday=now.strftime("%A"),
            today=now.date().isoformat(),
            now=now.strftime("%H:%M"),
            memory_context=state.get("memory_context", ""),
            grounding_context=state.get("grounding_context", ""),
        )
    )


def _current_turn(messages: list) -> list:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _offline_turn(state: AgentState, user_id: str, degraded: bool) -> tuple[str, list[dict]]:
    rag = get_rag()
    with get_db() as conn:
        profile = profile_from_memories(get_all_memory(conn, user_id))
    text = state.get("user_text", "")
    groundings = rag.ground_message(text, profile) if text else []
    return offline.respond(user_id, text, groundings, state.get("meal_type_hint"), raw_input=text, degraded=degraded)


async def agent(state: AgentState, config: RunnableConfig) -> dict:
    trace: list[dict] = []
    user_id = _uid(state, config)
    llm = get_chat_model("agent")
    _emit({"type": "step", "node": "agent", "label": "Thinking"})
    with span("agent", trace) as info:
        if llm is None:
            info["mode"] = "offline"
            reply, actions = await asyncio.to_thread(_offline_turn, state, user_id, False)
            return {"messages": [AIMessage(content=reply)], "actions": actions, "trace": trace}

        turn = _current_turn(state["messages"])
        rounds = sum(1 for m in turn if isinstance(m, AIMessage) and m.tool_calls)
        try:
            model = llm.bind_tools(ALL_TOOLS) if rounds < MAX_TOOL_ROUNDS else llm
            response = await model.ainvoke([_system_prompt(state), *state["messages"]], config)
        except Exception as exc:
            log.warning("LLM call failed (%s): %s", exc.__class__.__name__, exc)
            info["error"] = exc.__class__.__name__
            if not any(isinstance(m, ToolMessage) for m in turn):
                info["mode"] = "degraded"
                reply, actions = await asyncio.to_thread(_offline_turn, state, user_id, True)
                return {"messages": [AIMessage(content=reply)], "actions": actions, "trace": trace}
            done = [a["tool"] for a in state.get("actions", [])]
            reply = (
                (
                    "I completed " + ", ".join(done) + " but lost connection to the AI model before replying. "
                    "Check the dashboard — your log is saved."
                )
                if done
                else "The AI model is unavailable. Please try again."
            )
            return {"messages": [AIMessage(content=reply)], "trace": trace}

        usage = getattr(response, "usage_metadata", None) or {}
        info.update(
            mode="gemini",
            tool_calls=[tc["name"] for tc in response.tool_calls],
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
    return {"messages": [response], "trace": trace}


# ── tools ────────────────────────────────────────────────────────────────────


def _parse(content) -> dict:
    try:
        data = json.loads(message_text(content))
        return data if isinstance(data, dict) else {"result": data}
    except (TypeError, json.JSONDecodeError):
        return {"status": "error", "error": message_text(content)[:500]}


async def tools(state: AgentState, config: RunnableConfig) -> dict:
    trace: list[dict] = []
    last = state["messages"][-1]
    names = [tc["name"] for tc in getattr(last, "tool_calls", [])]
    _emit({"type": "step", "node": "tools", "label": "Running " + ", ".join(names)})
    with span("tools", trace, calls=names):
        out = await _tool_node.ainvoke(state, config)
    actions = [
        {"tool": m.name, "call_id": m.tool_call_id, "result": _parse(m.content)}
        for m in out.get("messages", [])
        if isinstance(m, ToolMessage)
    ]
    return {"messages": out.get("messages", []), "actions": actions, "trace": trace}


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"
