"""
The real LangGraph agent, driven by a scripted chat model (no network).

Covers the LLM path end to end: speculative retrieval in the prompt, tool calling,
server-side grounding, corrections, streaming over SSE, and graceful degradation
when the model provider fails.
"""

import asyncio
import json

import pytest

from backend.api.chat import run_turn
from backend.core.llm import override_chat_model
from backend.db.database import get_db
from backend.db.meals import get_daily_totals, get_meals
from tests.scripted_llm import ScriptedChatModel, call, last_tool_result, say

U, S = "default", "graph-session"


def turn(message: str) -> dict:
    async def _run():
        events = [e async for e in run_turn(message, U, S)]
        return events

    events = asyncio.run(_run())
    done = events[-1]
    assert done["event"] == "done"
    return {"events": events, **done["data"]}


def test_retrieved_context_is_in_the_prompt_and_logging_is_grounded():
    model = ScriptedChatModel(
        steps=[
            call(
                "log_meal",
                {
                    "items": [
                        {"name": "paratha", "kb_id": "paratha", "quantity": 2},
                        {"name": "chai", "kb_id": "masala_chai"},
                    ],
                    "meal_type": "breakfast",
                },
            ),
            say("Logged 2 parathas + chai — 640 kcal."),
        ]
    )
    with override_chat_model(model):
        out = turn("had 2 parathas and chai for breakfast")

    system_prompt = model.seen[0][0].content
    assert '"kb_id": "paratha"' in system_prompt and '"kb_id": "masala_chai"' in system_prompt
    assert out["response"] == "Logged 2 parathas + chai — 640 kcal."
    assert out["meals_changed"][0]["meal"]["calories"] == 640
    assert [s["name"] for s in out["trace"]] == ["retrieve", "agent", "tools", "agent"]
    kinds = [e["event"] for e in out["events"]]
    assert kinds.index("retrieval") < kinds.index("tool_call") < kinds.index("meal") < kinds.index("done")


def test_correction_updates_in_place():
    with override_chat_model(
        ScriptedChatModel(
            steps=[
                call("log_meal", {"items": [{"name": "roti", "kb_id": "roti", "quantity": 2}]}),
                say("Logged 2 rotis."),
            ]
        )
    ):
        turn("2 rotis")

    def correct(messages):
        meal_id = last_tool_result(messages, "get_meals")["meals"][-1]["meal_id"]
        return call("update_meal", {"meal_id": meal_id, "items": [{"name": "roti", "kb_id": "roti", "quantity": 3}]})

    with override_chat_model(
        ScriptedChatModel(steps=[call("get_meals", {"date": "today"}), correct, say("Fixed: 3 rotis.")])
    ):
        out = turn("actually that was 3 rotis not 2")

    assert out["meals_changed"][0]["action"] == "updated"
    with get_db() as conn:
        assert get_daily_totals(conn, U)["total_calories"] == 300  # not 200 + 300
        assert len(get_meals(conn, U)) == 1


def test_needs_confirmation_is_relayed_not_logged():
    with override_chat_model(
        ScriptedChatModel(
            steps=[
                call("log_meal", {"items": [{"name": "paratha", "kb_id": "paratha", "quantity": 50}]}),
                lambda msgs: say(
                    "50 parathas?" if last_tool_result(msgs, "log_meal")["status"] == "needs_confirmation" else "?"
                ),
            ]
        )
    ):
        out = turn("ate 50 parathas")
    assert out["response"] == "50 parathas?" and out["meals_changed"] == []


def test_provider_outage_degrades_to_offline_engine():
    with override_chat_model(ScriptedChatModel(steps=[], fail_with=ConnectionError)):
        out = turn("had 2 rotis and dal")
    assert out["meals_changed"] and "offline engine" in out["response"]
    assert out["trace"][-1].get("mode") == "degraded"


def test_tool_round_cap_forces_an_answer():
    looping = ScriptedChatModel(steps=[call("get_daily_totals", {}, f"c{i}") for i in range(6)] + [say("Done.")])
    with override_chat_model(looping):
        out = turn("how am I doing?")
    assert out["response"] == "Done."


def test_sse_stream_emits_tokens(client):
    model = ScriptedChatModel(steps=[say("Hello there friend")])
    form = {"message": "hi", "session_id": "sse"}
    with override_chat_model(model), client.stream("POST", "/api/chat/stream", data=form) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events, tokens = [], []
        current = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                current = line[7:]
                events.append(current)
            elif line.startswith("data:") and current == "token":
                tokens.append(json.loads(line[6:])["text"])
    assert "".join(tokens) == "Hello there friend"
    assert events[0] == "start" and events[-1] == "done"


@pytest.mark.parametrize("tool_name", ["log_meal", "update_meal", "lookup_nutrition", "set_user_memory"])
def test_tool_schemas_convert_for_gemini(tool_name):
    """Every tool's JSON schema must convert to a Gemini FunctionDeclaration (no network)."""
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    from backend.agent.tools import ALL_TOOLS

    tool = next(t for t in ALL_TOOLS if t.name == tool_name)
    decl = convert_to_genai_function_declarations([tool])[0].function_declarations[0]
    assert decl.name == tool_name and "config" not in (decl.parameters.properties or {})
