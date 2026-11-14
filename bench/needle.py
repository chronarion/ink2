"""Long-context needle test through the OpenAI-compatible API: a passphrase sentence inserted at a given depth into
filler text built from the corpus's raw wiki/code texts, sized to N tokens with /tokenize, then the model is asked
for the passphrase (thinking off, greedy). Reports TTFT (= prefill time) and whether the answer contains it.
   python3 -I needle.py [URL] --tokens 131072,524288,1000000 --depths 0.1,0.5,0.9 [--max-tokens 32] [--seed S]"""
import argparse, json, random, time, urllib.request
import os

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8111/v1")
ap.add_argument("--tokens", default="131072,524288")
ap.add_argument("--depths", default="0.5")
ap.add_argument("--raw", default=os.environ.get("INK2_CORPUS", "/data/scratch/kld/corpus/raw.json"))
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--seed", type=int, default=None, help="fixed passphrases (compare servers on the same needles)")
a = ap.parse_args()
random.seed(a.seed)
base = a.url.rsplit("/v1", 1)[0]
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]


def post(path, body, timeout=3600):
    req = urllib.request.Request(path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ntok(text):
    return post(f"{base}/tokenize", {"model": model, "prompt": text})["count"]


raw = json.load(open(a.raw))
texts = [t for v in (raw.values() if isinstance(raw, dict) else [raw]) for t in (v if isinstance(v, list) else [v])
         if isinstance(t, str)]
filler = "\n\n".join(texts)
per_char = ntok(filler[:200000]) / 200000
for N in [int(x) for x in a.tokens.split(",")]:
    for depth in [float(x) for x in a.depths.split(",")]:
        code = f"{random.randint(100000, 999999)}-{random.choice(['amber', 'cobalt', 'violet', 'saffron'])}"
        needle = f"\n\nThe secret passphrase for the vault is {code}. Remember it.\n\n"
        need_chars = int(N / per_char)
        body = (filler * (need_chars // len(filler) + 1))[:need_chars]
        cut = int(len(body) * depth)
        doc = body[:cut] + needle + body[cut:]
        q = "\n\nWhat is the secret passphrase for the vault? Reply with the passphrase only."
        n = ntok(doc + q)
        t0 = time.time()
        r = post(f"{a.url}/chat/completions", dict(model=model, messages=[{"role": "user", "content": doc + q}],
                                                   max_tokens=a.max_tokens, temperature=0, reasoning_effort="none"))
        dt = time.time() - t0
        ans = r["choices"][0]["message"].get("content") or ""
        print(f"needle N={n} depth={depth}: {'PASS' if code in ans else 'FAIL'} in {dt:.1f} s "
              f"({n / dt:.0f} tok/s end to end) answer {ans[:160]!r}", flush=True)
