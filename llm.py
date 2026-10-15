"""Intent classifier. Uses a small open LLM (Qwen2.5-0.5B-Instruct) in Colab, or a fake model in tests/CI."""
import os, time

LABELS = ["duplicate_invoices", "access_anomalies", "policy_question", "other"]
PROMPT = """Classify the request into exactly one label: duplicate_invoices, access_anomalies, policy_question, other.
Request: "Check if we paid any vendor twice" -> duplicate_invoices
Request: "Who logged into the ERP at 3am?" -> access_anomalies
Request: "What is the rule for expenses above 5000?" -> policy_question
Request: "Book me a flight to Austin" -> other
Request: "{text}" ->"""


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
    name = "Qwen2.5-0.5B-Instruct"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tok = AutoTokenizer.from_pretrained(mid)
        kw = {"device_map": "auto"} if torch.cuda.is_available() else {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(mid, dtype="auto", **kw)        # transformers 5.x
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype="auto", **kw)  # transformers 4.x
        self.fallback = FakeLLM()

    def classify(self, text):
        msgs = [{"role": "user", "content": PROMPT.format(text=text)}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        t0 = time.perf_counter()
        out = self.model.generate(ids, max_new_tokens=8, do_sample=False)
        latency = (time.perf_counter() - t0) * 1000
        gen = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().lower()
        label = next((l for l in LABELS if l in gen), None)
        meta = {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(out.shape[1] - ids.shape[1]),
                "latency_ms": latency, "fallback": label is None}
        if label is None:                                    # guardrail: invalid output -> safe fallback
            label, _ = self.fallback.classify(text)
        return label, meta


def get_llm():
    return FakeLLM() if os.getenv("FAKE_LLM") == "1" else HFLLM()
