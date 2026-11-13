"""Bit-exactness probe for prefill-path changes: greedy continuations of fixed long prompts
(wikitext from the KLD raw corpus, a fixed first line per length so nothing is shared through the prefix cache),
64 tokens each, thinking off. Run it once per server boot and diff the outputs of two runs.
   python3 -I longcheck.py LABEL OUT.json [--url URL] [--sizes 20000,40000]"""
import argparse, json, urllib.request
import os

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8111/v1")
ap.add_argument("--sizes", default="20000,40000")
ap.add_argument("--raw", default=os.environ.get("INK2_CORPUS", "/data/scratch/kld/corpus/raw.json"))
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
words = "\n\n".join(json.load(open(a.raw))["wikitext_test"]).split(" ")


def post(path, body, root=False):
    url = (a.url.rsplit("/v1", 1)[0] if root else a.url) + path
    return json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=1800))


res = {}
for N in [int(x) for x in a.sizes.split(",") if x]:
    lo, hi, body = 1, len(words), None
    while lo < hi:
        mid = (lo + hi) // 2
        msg = f"[longcheck {N}]\n" + " ".join(words[:mid])
        if post("/tokenize", {"model": model, "prompt": msg}, root=True)["count"] < N:
            lo = mid + 1
        else:
            hi, body = mid, msg
    r = post("/chat/completions", {"model": model, "messages": [{"role": "user", "content": body + "\n\nContinue the text."}],
                                   "max_tokens": 64, "temperature": 0, "reasoning_effort": "none"})
    text = r["choices"][0]["message"]["content"]
    res[str(N)] = {"prompt_tokens": r["usage"]["prompt_tokens"], "text": text}
    print(f"{a.label}: longcheck {N}: prompt {r['usage']['prompt_tokens']} tokens, output {text[:80]!r}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
