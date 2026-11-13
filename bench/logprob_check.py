"""Prompt logprobs of a fixed ~N-token text (teacher-forced prefill), saved for comparing two builds; also the prefill time.
   python3 -I logprob_check.py URL OUT.json [TOKENS=8000]      compare: python3 -I logprob_check.py --cmp A.json B.json"""
import json, sys, time, urllib.request
import os
if sys.argv[1] == "--cmp":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    d = [abs(x - y) for x, y in zip(a["lp"], b["lp"]) if x is not None and y is not None]
    agree = sum(x == y for x, y in zip(a["top"], b["top"]))
    print(f"{len(d)} tokens: mean |dlogprob| {sum(d) / len(d):.4f}, max {max(d):.3f}, top-1 agree {agree}/{len(a['top'])}; "
          f"prefill {a['s']:.2f} s vs {b['s']:.2f} s")
    sys.exit()
url, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
raw = json.load(open(os.environ.get("INK2_CORPUS", "/data/scratch/kld/corpus/raw.json")))
start = {8000: 0, 32000: 40000}.get(n, 200000)          # separate regions: no prefix-cache hits between sizes
text = "".join(raw["wikitext_test"])[start:start + int(n * 4.4)]
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0}
if n <= 16000:
    body["prompt_logprobs"] = 1
t = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=1800))
s = time.perf_counter() - t
pl = (r["choices"][0].get("prompt_logprobs") or [None])[1:]
lp, top = [], []
for d in pl:
    # entries {token_id: {logprob, rank, decoded_token}}: the top-1 (rank 1) and the prompt token (the highest rank;
    # the same entry when the prompt token is the top-1)
    vals = list(d.values())
    top.append(min(vals, key=lambda v: v["rank"])["decoded_token"])
    lp.append(max(vals, key=lambda v: v["rank"])["logprob"])
json.dump({"lp": lp, "top": top, "s": s, "tokens": r["usage"]["prompt_tokens"]}, open(out, "w"))
print(f"{r['usage']['prompt_tokens']} prompt tokens, {s:.2f} s")
