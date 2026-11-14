"""Greedy decode of fixed prompts, one request at a time, with each token's logprob: saved per build to compare the decode
path of two builds (identical arithmetic gives identical tokens and logprobs).
   python3 -I greedy_compare.py URL OUT.json [MAX_TOKENS=96]       compare: python3 -I greedy_compare.py --cmp A.json B.json"""
import json
import sys
import urllib.request

PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a haiku about a lighthouse in winter.",
    "What is 17 multiplied by 23? Show the steps briefly.",
    "List five uses of a paperclip, one per line.",
    "Translate into French: The train leaves at nine in the morning.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Give three tips for keeping houseplants alive.",
    "用两句话介绍长城。",
    "What are the main differences between TCP and UDP?",
    "Describe the taste of a lemon to someone who has never had one.",
    "Count from 1 to 25, separated by spaces.",
]

if sys.argv[1] == "--cmp":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    same, firsts, dmax = 0, [], 0.0
    for x, y in zip(a, b):
        tx, ty = x["tokens"], y["tokens"]
        n = next((i for i, (u, v) in enumerate(zip(tx, ty)) if u != v), min(len(tx), len(ty)))
        same += tx == ty
        if tx != ty:
            firsts.append(n)
        d = [abs(u - v) for u, v in zip(x["lp"][:n], y["lp"][:n])]
        dmax = max([dmax] + d)
    print(f"{same}/{len(a)} identical; first divergence at token {sorted(firsts)}; max |dlogprob| before divergence "
          f"{dmax:.2e}")
    sys.exit()

url, out = sys.argv[1], sys.argv[2]
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 96
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
res = []
for p in PROMPTS:
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0,
            "logprobs": True, "top_logprobs": 1, "reasoning_effort": "none"}
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=600))
    c = r["choices"][0]
    lps = (c.get("logprobs") or {}).get("content") or []
    res.append({"prompt": p, "text": c["message"].get("content") or "", "tokens": [t["token"] for t in lps],
                "lp": [t["logprob"] for t in lps]})
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print(f"wrote {out}: {len(res)} prompts, {sum(len(x['tokens']) for x in res)} tokens")
