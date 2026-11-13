"""Routing statistics traffic for the expert placement: prompts from mixed text families (wikitext, chat, GSM8K,
Chinese Wikipedia, source code), each answered with up to 256 tokens, 8 at a time, for MINUTES minutes. The server
must run with INK2_COUNT (decode-sized calls are counted, inside the CUDA graphs).
   python3 -I route_corpus.py URL MINUTES [MAX_TOKENS=256] [STREAMS=8]"""
import glob, json, random, sys, threading, time, urllib.request
import os

URL, MINUTES = sys.argv[1], float(sys.argv[2])
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
STREAMS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
raw = json.load(open(os.environ.get("INK2_CORPUS", "/data/scratch/kld/corpus/raw.json")))
rnd = random.Random(13)
prompts = []
wiki = "".join(raw["wikitext_test"]).split(" = ")
for chunk in rnd.sample([c for c in wiki if len(c) > 2000], 60):
    prompts.append([{"role": "user", "content": "Summarize this article:\n\n" + chunk[:8000]}])
for conv in rnd.sample(raw["no_robots_test"], 80):
    msgs = [m for m in conv if m["role"] in ("system", "user", "assistant")]
    while msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]
    if msgs:
        prompts.append(msgs)
for q in rnd.sample(raw["gsm8k_test"], 60):
    prompts.append([{"role": "user", "content": q["q"]}])
for d in rnd.sample(raw["zhwiki"], 40):
    prompts.append([{"role": "user", "content": "请总结下面的文章：\n\n" + d["text"][:6000]}])
for f in sorted(glob.glob(os.path.join(os.environ.get("INK2_CODE_DIR", "/data/scratch/kld/calib"), "*.py")))[:30]:
    src = open(f, errors="replace").read()
    s = rnd.randrange(max(1, len(src) - 12000))
    prompts.append([{"role": "user", "content": "Explain what this code does and point out one possible bug:\n\n```\n"
                     + src[s:s + 12000] + "\n```"}])
rnd.shuffle(prompts)
print(f"{len(prompts)} prompts", flush=True)
stop, done, lock = time.time() + MINUTES * 60, [0, 0], threading.Lock()


def worker(i):
    j = i
    while time.time() < stop:
        msgs = prompts[j % len(prompts)]
        j += STREAMS
        body = {"model": "inkling2", "messages": msgs, "max_tokens": MAXTOK, "temperature": 1.0}
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions",
                           json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
            with lock:
                done[0] += 1
                done[1] += r["usage"]["total_tokens"]
        except Exception as e:  # noqa: BLE001
            print("error", repr(e)[:200], flush=True)


ts = [threading.Thread(target=worker, args=(i,)) for i in range(STREAMS)]
[t.start() for t in ts]
[t.join() for t in ts]
print(f"done: {done[0]} requests, {done[1]} tokens", flush=True)
