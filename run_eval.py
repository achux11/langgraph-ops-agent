"""
Golden-set evaluation of the LangGraph agent.
Run in Google Colab (T4 GPU recommended, CPU works, ~3-6 minutes):
    !pip install -q langgraph transformers accelerate pytest
    upload this folder (or git clone your repo), then:  !python run_eval.py
Paste the RESULTS block back to Claude.
"""
import os, statistics, time
from agent import build_graph, run
from llm import get_llm

GOLDEN = [  # (request, expected intent) - 60 hand-labeled cases
    *[(t, "duplicate_invoices") for t in [
        "Did we pay any vendor twice this quarter?", "Find duplicate invoices from our subcontractors",
        "Check for double payments to Acme Concrete", "Are there repeated invoices in accounts payable?",
        "Scan AP for invoices billed two times", "Any vendor bills we might have paid more than once?",
        "Look for duplicate billing in procurement", "I think Summit Steel invoiced us twice, can you check?",
        "Run the duplicate payment check", "Identify invoices with same vendor amount and date",
        "Were any invoices processed twice last month?", "Detect double-billed invoices",
        "Show me possible duplicate vendor payments", "Check AP for repeat invoices", "Find twice-paid bills"]],
    *[(t, "access_anomalies") for t in [
        "Who logged into the ERP at 3am?", "Any suspicious logins this week?", "Show unusual access to HRIS",
        "Find users with many failed logins", "Was there off-hours admin access?", "Check access logs for anomalies",
        "Did anyone access procurement at night?", "List suspicious login attempts", "Any brute force login patterns?",
        "Review ERP access outside business hours", "Who had repeated failed logins?", "Flag strange access activity",
        "Check for unauthorized access attempts", "Any anomalies in system logins?", "Show risky access events"]],
    *[(t, "policy_question") for t in [
        "What is the policy for expenses over 5000?", "What's the rule for onboarding a new vendor?",
        "Is off-hours access allowed by policy?", "Do I need approval for a large expense?",
        "What does the vendor policy require?", "Explain the expense approval rule", "What is our access review policy?",
        "Which approval is needed for big purchases?", "Policy on paying a new vendor?", "Rules for ERP access reviews?",
        "Is a receipt required for expenses?", "What does policy say about W-9s?", "Who approves expenses above the limit?",
        "Access policy for HRIS?", "Vendor insurance requirement policy?"]],
    *[(t, "other") for t in [
        "Book me a flight to Austin", "What's the weather tomorrow?", "Write a poem about concrete",
        "Order lunch for the team", "Translate this into Spanish", "Who won the game last night?",
        "Schedule a meeting with Sam", "Tell me a joke", "Reset my laptop password", "Recommend a good movie",
        "Summarize the news", "Plan a team offsite", "What time is it in Tokyo?", "Draft a birthday message",
        "How tall is the Eiffel Tower?"]],
]

llm = get_llm()
graph = build_graph(llm)
rows, t_all = [], time.perf_counter()
for i, (req, gold) in enumerate(GOLDEN):
    t0 = time.perf_counter()
    state, paused = run(graph, req, thread_id=f"t{i}")
    rows.append({"ok": state["intent"] == gold, "paused": paused, "e2e_ms": (time.perf_counter() - t0) * 1000,
                 "tokens": state["usage"]["prompt_tokens"] + state["usage"]["completion_tokens"],
                 "fallback": state["usage"]["fallback"], "audit_steps": len(state["audit"]),
                 "has_response": bool(state.get("response"))})
n = len(rows)
acc = sum(r["ok"] for r in rows) / n
tok = sum(r["tokens"] for r in rows)
print("\n================ RESULTS ================")
print(f"Model: {llm.name} | Golden set: {n} labeled requests across 4 intents | Graph nodes: 7 (incl. human review)")
print(f"Intent routing accuracy: {acc:.1%} | Invalid LLM outputs caught by fallback guardrail: {sum(r['fallback'] for r in rows)}")
print(f"Escalated to human review: {sum(r['paused'] for r in rows)} of {n} | Every run completed with a response: {all(r['has_response'] for r in rows)}")
print(f"Audit trail: {statistics.mean(r['audit_steps'] for r in rows):.1f} logged steps per request")
print(f"Latency end-to-end: median {statistics.median(r['e2e_ms'] for r in rows):.0f} ms | total tokens {tok:,} "
      f"({tok / n:.0f} per request; est. cost ${tok / 1e6 * 0.15:.4f} for all {n} at $0.15 per 1M tokens)")
