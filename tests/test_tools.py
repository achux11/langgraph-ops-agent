import data, tools


def test_duplicate_invoices_finds_planted_duplicates():
    r = tools.find_duplicate_invoices(data.make_invoices())
    assert r["duplicate_groups"] >= 8 and r["exposure_usd"] > 0


def test_no_duplicates_in_clean_data():
    inv = [{"invoice_id": "A", "vendor": "X", "amount": 1.0, "date": "2026-01-01"},
           {"invoice_id": "B", "vendor": "X", "amount": 2.0, "date": "2026-01-01"}]
    assert tools.find_duplicate_invoices(inv)["duplicate_groups"] == 0


def test_access_anomalies_detected():
    r = tools.find_access_anomalies(data.make_access_logs())
    assert r["anomalies"] >= 6


def test_policy_lookup_and_fallback():
    assert tools.answer_policy("what is the expense policy")["policy"] == "expense"
    assert tools.answer_policy("book a flight")["policy"] is None
