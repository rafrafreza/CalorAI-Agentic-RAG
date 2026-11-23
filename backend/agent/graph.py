"""
The CalorAI agent as an explicit LangGraph state machine.

    START ─► ingest ─► retrieve ─► agent ─┬─► END
                                    ▲     │
                                    └─ tools ◄┘   (tool calls)

Why a graph rather than an opaque agent executor: the decisions this product is
judged on (ask vs. log, correct vs. re-log, grounded vs. estimated) are visible,
testable control flow; every node is timed and streamed to the UI.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agent import nodes
from backend.agent.state import AgentState


def create_graph():
    graph = StateGraph(AgentState)
    graph.add_node("ingest", nodes.ingest)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", nodes.tools)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", nodes.route_after_agent, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


agent_graph = create_graph()
