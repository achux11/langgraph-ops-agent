"""Synthetic business data the agent's tools work on (no real or sensitive data)."""
import random

random.seed(7)
VENDORS = ["Acme Concrete", "Lone Star Electric", "Metro Plumbing", "Summit Steel", "Prairie HVAC"]


def make_invoices(n=120):
    rows = []
    for i in range(n):
        rows.append({"invoice_id": f"INV-{1000 + i}", "vendor": random.choice(VENDORS),
                     "amount": round(random.uniform(500, 25000), 2), "date": f"2026-0{random.randint(1, 9)}-1{random.randint(0, 9)}"})
    for j in range(8):                                   # planted duplicates (same vendor + amount + date)
        src = rows[j * 7]
        rows.append({**src, "invoice_id": f"INV-D{j}"})
    return rows


def make_access_logs(n=200):
    users = [f"user{u}" for u in range(25)]
    rows = [{"user": random.choice(users), "system": random.choice(["ERP", "HRIS", "Procurement"]),
             "hour": random.randint(7, 19), "failed_logins": random.choice([0, 0, 0, 1])} for _ in range(n)]
    for k in range(6):                                   # planted anomalies: off-hours + repeated failures
        rows.append({"user": f"user{k}", "system": "ERP", "hour": random.choice([1, 2, 3]), "failed_logins": random.randint(4, 9)})
    return rows


POLICIES = {
    "expense": "Expenses over $5,000 need director approval and an itemized receipt.",
    "vendor": "New vendors must pass a W-9 and insurance check before the first payment.",
    "access": "Access to ERP and HRIS is reviewed quarterly; off-hours admin access must be justified.",
}
