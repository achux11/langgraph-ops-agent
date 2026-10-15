"""Intent classifier for the agent.

HFLLM uses *constrained label scoring*: instead of letting the model generate free text, it scores each allowed label
(log-probability given the prompt) and picks the best one. Small models handle this far better than free generation,
and the output can never be an invalid label. FakeLLM is a keyword stand-in for unit tests and CI.
"""
import os, time

LABELS = ["duplicate_invoices", "access_anomalies", "policy_question", "other"]
SYSTEM = ("You route back-office requests. Labels:\n"
          "duplicate_invoices = paying a vendor twice, duplicate or repeated invoices or bills.\n"
          "access_anomalies = suspicious logins, failed logins, off-hours or unauthorized system access.\n"
          "policy_question = asking what a rule, policy, approval or requirement is.\n"
          "other = anything unrelated to finance operations or access security.\n"
          "Answer with the label only.")
FEW_SHOT = [("Check if we paid any vendor twice", "duplicate_invoices"),
            ("Find invoices billed two times by the same supplier", "duplicate_invoices"),
            ("Who logged into the ERP at 3am?", "access_anomalies"),
            ("List users with repeated failed logins", "access_anomalies"),
            ("What is the rule for expenses above 5000?", "policy_question"),
            ("Do new vendors need a W-9 before payment?", "policy_question"),
            ("Book me a flight to Austin", "other"),
            ("Tell me a joke", "other")]


class FakeLLM:
    """Keyword stand-in so unit/integration tests and CI run without downloading a model."""
    name = "fake-keyword"

    def classify(self, text):
        t = text.lower()
        label = ("duplicate_invoices" if any(w in t for w in ["twice", "duplicate", "double"]) else
                 "access_anomalies" if any(w in t for w in ["login", "logged", "access", "suspicious"]) else
                 "policy_question" if any(w in t for w in ["policy", "rule", "allowed", "approval"]) else "other")
        return label, {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0, "fallback": False}


class HFLLM:
    def __init__(self, model_id=None, model=None, tok=None):
        import torch
        self.torch = torch
        self.name = model_id or os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.name)
            kw = {"device_map": "auto"} if torch.cuda.is_available() else {}
            try:
                model = AutoModelForCausalLM.from_pretrained(self.name, dtype="auto", **kw)        # transformers 5.x
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype="auto", **kw)  # transformers 4.x
        self.model, self.tok = model.eval(), tok
        self.label_ids = [tok(l, add_special_tokens=False)["input_ids"] for l in LABELS]

    def _prompt(self, text):
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEW_SHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": text})
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def classify(self, text):
        torch = self.torch
        p_ids = self.tok(self._prompt(text), add_special_tokens=False)["input_ids"]
        t0, scores = time.perf_counter(), []
        with torch.no_grad():
            for l_ids in self.label_ids:                  # score each allowed label: mean log-prob of its tokens
                ids = torch.tensor([p_ids + l_ids], device=self.model.device)
                logp = torch.log_softmax(self.model(ids).logits[0, :-1].float(), dim=-1)
                tgt = ids[0, 1:]
                n = len(l_ids)
                scores.append(logp[-n:].gather(1, tgt[-n:].unsqueeze(1)).mean().item())
        label = LABELS[max(range(len(LABELS)), key=lambda i: scores[i])]
        meta = {"prompt_tokens": len(p_ids), "completion_tokens": sum(len(x) for x in self.label_ids),
                "latency_ms": (time.perf_counter() - t0) * 1000, "fallback": False}
        return label, meta


def get_llm():
    return FakeLLM() if os.getenv("FAKE_LLM") == "1" else HFLLM()
