"""Integration tests: run the whole LangGraph agent end to end with the fake LLM."""
import os
os.environ["FAKE_LLM"] = "1"
from agent import build_graph, run
from llm import get_llm

graph = build_graph(get_llm())


def test_routes_to_duplicate_tool_and_escalates_high_exposure():
    state, paused = run(graph, "Did we pay any vendor twice?", "it1", approve=True)
    assert state["intent"] == "duplicate_invoices" and paused
    assert [s["step"] for s in state["audit"]][:2] == ["classify", "tool:duplicate_invoices"]
    assert state["response"].startswith("duplicate_invoices")


def test_denied_approval_blocks_action():
    state, paused = run(graph, "Any suspicious logins?", "it2", approve=False)
    assert paused and state["response"].startswith("Escalated")


def test_low_risk_policy_answer_skips_human():
    state, paused = run(graph, "What is the expense policy?", "it3")
    assert not paused and "director approval" in state["response"]


def test_out_of_scope_request_goes_to_human():
    state, paused = run(graph, "Book me a flight", "it4", approve=False)
    assert state["intent"] == "other" and paused


def test_every_run_has_audit_trail():
    state, _ = run(graph, "What is the vendor policy?", "it5")
    steps = [s["step"] for s in state["audit"]]
    assert steps[0] == "classify" and steps[-1] == "respond"
