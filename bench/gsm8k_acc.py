"""End-to-end answer quality under load: GSM8K test questions at a given concurrency and temperature, thinking on, scored on the final
number. Catches output damage that QA at c=1 and the speed benches cannot see (masked rows at batch > 1, sampling paths).
   python3 -I gsm8k_acc.py LABEL [--url URL] [--c 4] [--n 200] [--temperature 1.0] [--raw $INK2_CORPUS]"""
import argparse, json, re, threading, urllib.request
import os

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default="http://127.0.0.1:8111/v1")
ap.add_argument("--c", type=int, default=4)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--raw", default=os.environ.get("INK2_CORPUS", "/data/scratch/kld/corpus/raw.json"))
ap.add_argument("--dump", default="", help="write one JSON row per question here")
a = ap.parse_args()
model = json.load(urllib.request.urlopen(f"{a.url}/models"))["data"][0]["id"]
rows = json.load(open(a.raw))["gsm8k_test"][:a.n]
num = lambda s: re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
gold = lambda r: float(num(r["a"].split("####")[-1])[-1].replace(",", ""))
res, lock, nxt = [], threading.Lock(), [0]


def one(q):
    req = {"model": model, "messages": [{"role": "user", "content": q + "\nEnd with the final number alone on the last line."}],
           "max_tokens": 4096, "temperature": a.temperature, "top_p": 0.95, "reasoning_effort": "medium"}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.url}/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=3600))
    c = r["choices"][0]
    return c["message"].get("content") or "", r["usage"]["completion_tokens"], c["finish_reason"], c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""


def worker():
    while True:
        with lock:
            i = nxt[0]; nxt[0] += 1
        if i >= len(rows):
            return
        text, n, fin, think = one(rows[i]["q"])
        v = num(text)
        ok = bool(v) and abs(float(v[-1].rstrip(".").replace(",", "")) - gold(rows[i])) < 1e-6
        with lock:
            res.append((ok, n, fin, i, text[-200:], think[-400:]))


th = [threading.Thread(target=worker) for _ in range(a.c)]
[t.start() for t in th]; [t.join() for t in th]
cap = sum(r[2] == "length" for r in res)
print(f"{a.label}: GSM8K c={a.c} T={a.temperature}: {sum(r[0] for r in res)}/{len(res)} correct, mean {sum(r[1] for r in res) / len(res):.0f} tokens, {cap} at the cap")
if a.dump:
    json.dump(sorted(res, key=lambda r: r[3]), open(a.dump, "w"))
