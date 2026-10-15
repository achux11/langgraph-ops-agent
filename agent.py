"""LangGraph agent: classify -> route -> tool -> risk gate -> (human review) -> respond, with an audit trail."""
import time
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import data, tools


class State(TypedDict, total=False):
    request: str
    intent: str
    result: dict
    high_risk: bool
    approved: Optional[bool]
    response: str
    audit: list
    usage: dict


def build_graph(llm, invoices=None, logs=None, exposure_threshold=10000):
    invoices = invoices or data.make_invoices()
    logs = logs or data.make_access_logs()

    def log(state, step, detail):
        return state.get("audit", []) + [{"step": step, "detail": detail, "ts": time.time()}]

    def classify(state):
        label, meta = llm.classify(state["request"])
        return {"intent": label, "usage": meta, "audit": log(state, "classify", {"intent": label, **meta})}

    def run_invoices(state):
        r = tools.find_duplicate_invoices(invoices)
        return {"result": r, "audit": log(state, "tool:duplicate_invoices", r)}

    def run_access(state):
        r = tools.find_access_anomalies(logs)
        return {"result": r, "audit": log(state, "tool:access_anomalies", r)}

    def run_policy(state):
        r = tools.answer_policy(state["request"])
        return {"result": r, "audit": log(state, "tool:policy", r)}

    def risk_gate(state):
        r = state.get("result", {})
        high = (r.get("exposure_usd", 0) > exposure_threshold or r.get("anomalies", 0) > 0
                or state.get("intent") == "other" or r.get("policy", "x") is None)
        return {"high_risk": high, "audit": log(state, "risk_gate", {"high_risk": high})}

    def human_review(state):                     # graph pauses BEFORE this node until a person sets `approved`
        return {"audit": log(state, "human_review", {"approved": state.get("approved")})}

    def respond(state):
        if state.get("high_risk") and not state.get("approved"):
            msg = "Escalated: waiting for or denied human approval."
        else:
            msg = f"{state['intent']}: {state.get('result')}"
        return {"response": msg, "audit": log(state, "respond", {"response": msg[:80]})}

    g = StateGraph(State)
    for name, fn in [("classify", classify), ("invoices", run_invoices), ("access", run_access),
                     ("policy", run_policy), ("risk_gate", risk_gate), ("human_review", human_review), ("respond", respond)]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", lambda s: s["intent"],
                            {"duplicate_invoices": "invoices", "access_anomalies": "access",
                             "policy_question": "policy", "other": "risk_gate"})
    for n in ["invoices", "access", "policy"]:
        g.add_edge(n, "risk_gate")
    g.add_conditional_edges("risk_gate", lambda s: "human_review" if s["high_risk"] else "respond",
                            {"human_review": "human_review", "respond": "respond"})
    g.add_edge("human_review", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])


def run(graph, request, thread_id, approve=True):
    cfg = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"request": request, "audit": []}, cfg)
    paused = graph.get_state(cfg).next == ("human_review",)
    if paused:                                    # human-in-the-loop: record the decision, then resume
        graph.update_state(cfg, {"approved": approve})
        graph.invoke(None, cfg)
    return graph.get_state(cfg).values, paused
