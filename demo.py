"""Quick demo: run 3 requests through the agent and print the decision trail.
    FAKE_LLM=1 python demo.py      # instant, no model download
    python demo.py                 # real Qwen2.5-1.5B-Instruct (Colab)
"""
from agent import build_graph, run
from llm import get_llm

REQUESTS = [("Did we pay any vendor twice this quarter?", True),
            ("What is the rule for expenses above 5000?", True),
            ("Who logged into the ERP at 3am?", False)]

graph = build_graph(get_llm())
for i, (req, approve) in enumerate(REQUESTS):
    state, paused = run(graph, req, thread_id=f"demo{i}", approve=approve)
    print(f"\nRequest : {req}")
    print(f"Intent  : {state['intent']}")
    print(f"Human   : {'paused for review -> ' + ('APPROVED' if approve else 'DENIED') if paused else 'not needed (low risk)'}")
    print(f"Response: {state['response'][:110]}")
    print("Trail   : " + " -> ".join(s["step"] for s in state["audit"]))
