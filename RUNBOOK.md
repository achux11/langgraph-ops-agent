# Runbook: LangGraph Ops Agent

How to operate the agent, recognize a healthy run, and handle the exceptions it is designed to raise.
For what the agent does and how it was evaluated, see the [README](README.md).

## 1. What a healthy run looks like

Every run appends to `state["audit"]`. Read the `step` values in order:

| Path | Audit steps | Meaning |
|---|---|---|
| Low risk | `classify` → `tool:<name>` → `risk_gate` → `respond` | Answered without a person |
| High risk | `classify` → `tool:<name>` → `risk_gate` → *(paused)* → `human_review` → `respond` | A person approved or denied |
| Out of scope | `classify` → `risk_gate` → *(paused)* → `human_review` → `respond` | Request was not one of the 3 supported tasks |

`risk_gate.detail.high_risk` tells you which path was taken. `classify.detail` records the intent plus `prompt_tokens`, `completion_tokens` and `latency_ms`.

**Red flag:** a `respond` step with no `risk_gate` before it should never happen. If you see one, the graph was changed; run the tests (section 6) before using it.

## 2. Why a request was escalated

`risk_gate` sends a run to human review when **any** of these is true:

| Trigger | Where it comes from | Tune it |
|---|---|---|
| Duplicate-invoice exposure > $10,000 | `tools.find_duplicate_invoices` → `exposure_usd` | `build_graph(..., exposure_threshold=...)` |
| Any access anomaly | `tools.find_access_anomalies` → `anomalies > 0` | `off_hours` and `max_failed` in `find_access_anomalies` |
| Request is out of scope | `intent == "other"` | Add the task as a new tool and label (section 5) |
| No matching policy | `answer_policy` → `policy is None` | Add the policy to `data.POLICIES` |

Escalation is intentional: in the golden-set eval, 48 of 60 runs escalated because the synthetic data contains planted risks. **A rising escalation rate on real data is a signal to investigate, not a bug to suppress.** Never raise `exposure_threshold` to reduce review volume without sign-off from the business owner.

## 3. A run is paused waiting for a person

High-risk runs stop **before** the `human_review` node (`interrupt_before=["human_review"]`).

**Check:**
```python
cfg = {"configurable": {"thread_id": "<thread id>"}}
graph.get_state(cfg).next      # ('human_review',) means it is waiting
graph.get_state(cfg).values    # the tool result the reviewer should look at
```

**Resume with a decision** (same `thread_id`):
```python
graph.update_state(cfg, {"approved": True})   # or False to deny
graph.invoke(None, cfg)
```
A denied or missing approval returns `"Escalated: waiting for or denied human approval."` and is logged in the audit trail.

**Known limitation:** the checkpointer is `MemorySaver`, which lives in process memory. **If the process restarts, paused runs are lost** and must be resubmitted. Before real use, swap in a persistent checkpointer (for example SQLite or Postgres) so paused runs survive restarts.

## 4. Failures and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| `OSError` / download error on startup | Model can't be fetched (offline, no Hugging Face access) | Run with `FAKE_LLM=1` to keep tests and demos working; fix network access or point `MODEL_ID` at a cached model |
| Very slow responses (seconds per request) | Running the 1.5B model on CPU | Use a GPU, or set `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct` and re-run the eval first (accuracy will change) |
| `TypeError` mentioning `dtype` | transformers version differences | Already handled in `llm.py` (tries `dtype`, falls back to `torch_dtype`); if it still fails, pin the version in `requirements.txt` |
| `KeyError: 'vendor'`, `'amount'`, `'date'` or `'invoice_id'` | Invoice export is missing a column or uses different names | Map the export to these 4 fields before passing `invoices=` to `build_graph` |
| `KeyError: 'hour'`, `'failed_logins'` or `'user'` | Access-log export has a different shape | Map the log to these 3 fields before passing `logs=` |
| Request routed to the wrong tool | Wording the model hasn't seen | Add the request to `GOLDEN` in `run_eval.py` with the correct label, then follow section 5 |
| CI is red | A code change broke routing, tools or the graph | Run `FAKE_LLM=1 pytest -v` locally; the failing test names the broken behavior |

The agent cannot return an invalid route: the model only scores the 4 allowed labels, so there is no free-text output to parse.

## 5. Making a change safely

1. Add or update cases in `GOLDEN` (`run_eval.py`) for the behavior you are changing.
2. Run `FAKE_LLM=1 pytest -v`. All tests must pass.
3. Run `python run_eval.py` with the real model (GPU recommended) and compare against the README baseline: **accuracy 95.0%, 48/60 escalations, ~1.6 s median latency**.
4. Do not merge if accuracy drops or the escalation count changes without a reason you can explain.
5. Update the README results table and note the change.

## 6. Checks to run after any change

```bash
FAKE_LLM=1 pytest -v          # 9 tests: tools, routing, approve/deny, low-risk path, out-of-scope, audit trail
FAKE_LLM=1 python demo.py     # 3 requests end to end; inspect the printed decision trail
```
