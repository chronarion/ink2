"""Decode throughput on unrelated conversations: STREAMS concurrent requests, each a different prompt (wiki summaries,
chat, GSM8K, Chinese, code), 1,024 generated tokens each (ignore_eos), temperature 1, default reasoning. Unlike a lil
cell, whose streams share one prompt, nothing here is shared, so it shows what promotion that follows the text gives
real mixed traffic. Prints aggregate generated tokens per second over the measured window.
   python3 -I mixed_bench.py URL STREAMS [SECONDS=120]"""
import glob, json, random, sys, threading, time, urllib.request
import os

URL, STREAMS = sys.argv[1], int(sys.argv[2])
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 120
raw = json.load(open(os.environ.get("INK2_CORPUS", "/data/scratch/kld/corpus/raw.json")))
rnd = random.Random(7)
prompts = []
wiki = "".join(raw["wikitext_test"]).split(" = ")
prompts += [[{"role": "user", "content": "Summarize this article:\n\n" + c[:6000]}]
            for c in rnd.sample([c for c in wiki if len(c) > 2000], 12)]
for conv in rnd.sample(raw["no_robots_test"], 12):
    msgs = [m for m in conv if m["role"] in ("system", "user", "assistant")]
    while msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]
    if msgs:
        prompts.append(msgs)
prompts += [[{"role": "user", "content": q["q"]}] for q in rnd.sample(raw["gsm8k_test"], 12)]
prompts += [[{"role": "user", "content": "请总结下面的文章：\n\n" + d["text"][:4000]}] for d in rnd.sample(raw["zhwiki"], 8)]
for f in sorted(glob.glob(os.path.join(os.environ.get("INK2_CODE_DIR", "/data/scratch/kld/calib"), "*.py")))[:8]:
    src = open(f, errors="replace").read()
    prompts.append([{"role": "user", "content": "Explain what this code does:\n\n```\n" + src[:8000] + "\n```"}])
rnd.shuffle(prompts)
model = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
t0, stop = time.time() + 20, time.time() + 20 + SECONDS      # 20 s warm-up, then the window
lock, events = threading.Lock(), []


def worker(i):
    j = i
    while time.time() < stop:
        body = {"model": model, "messages": prompts[j % len(prompts)], "max_tokens": 1024, "ignore_eos": True,
                "temperature": 1.0, "stream": True,
                # tokens from the server's running usage, not chunks: with speculative decoding a chunk carries
                # several tokens
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        j += STREAMS
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        seen = 0
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if line.startswith(b"data: {"):
                    d = json.loads(line[6:])
                    n = (d.get("usage") or {}).get("completion_tokens")
                    if n is not None and n > seen:
                        with lock:
                            events.append((time.time(), n - seen))
                        seen = n
                if time.time() > stop:
                    break


ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(STREAMS)]
[t.start() for t in ts]
time.sleep(max(0.0, stop - time.time()) + 1)
n = sum(k for e, k in events if t0 <= e <= stop)
print(f"mixed C{STREAMS}: {n / SECONDS:.1f} tok/s aggregate over {SECONDS:.0f} s ({len(prompts)} different prompts)")
