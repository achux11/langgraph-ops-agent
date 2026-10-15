"""Deterministic tools the agent can call. Each returns a dict and never mutates source data."""
from collections import defaultdict
from data import POLICIES


def find_duplicate_invoices(invoices):
    groups = defaultdict(list)
    for inv in invoices:
        groups[(inv["vendor"], inv["amount"], inv["date"])].append(inv["invoice_id"])
    dups = [ids for ids in groups.values() if len(ids) > 1]
    exposure = sum(inv["amount"] for inv in invoices
                   if any(inv["invoice_id"] in ids[1:] for ids in dups))
    return {"duplicate_groups": len(dups), "examples": dups[:3], "exposure_usd": round(exposure, 2)}


def find_access_anomalies(logs, off_hours=(0, 5), max_failed=3):
    hits = [r for r in logs if off_hours[0] <= r["hour"] <= off_hours[1] and r["failed_logins"] > max_failed]
    return {"anomalies": len(hits), "users": sorted({r["user"] for r in hits})}


def answer_policy(question):
    q = question.lower()
    for key, text in POLICIES.items():
        if key in q:
            return {"policy": key, "answer": text}
    return {"policy": None, "answer": "No matching policy found; routing to a human."}
