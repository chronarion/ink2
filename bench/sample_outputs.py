"""Read-it-yourself check: a handful of everyday prompts (essay, explanation, code, Chinese, a multi-turn chat, a list,
a summary), sent together so the server batches them, with thinking on and off; every full answer (and its reasoning)
is written to OUT for a human to read.
   python3 -I sample_outputs.py URL OUT.md [--temperature 1.0] [--max-tokens 900]"""
import argparse, json, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--max-tokens", type=int, default=900)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
P = [
    ("essay", "none", [{"role": "user", "content": "Write a short essay (about 300 words) on why lighthouses were "
                                                    "built where they were, and what replaced them."}]),
    ("explain", "medium", [{"role": "user", "content": "Explain how a refrigerator moves heat from inside to outside. "
                                                        "Keep it to a few paragraphs a teenager would follow."}]),
    ("code", "low", [{"role": "user", "content": "Write a Python function that merges overlapping intervals, with a "
                                                  "docstring and three small tests."}]),
    ("chinese", "none", [{"role": "user", "content": "用中文简单介绍一下长城的历史，大约200字。"}]),
    ("multiturn", "none", [{"role": "user", "content": "I'm planning a 3-day trip to Lisbon in March."},
                           {"role": "assistant", "content": "Great choice! March is mild in Lisbon. What do you enjoy "
                                                            "most: food, history, walking, or day trips?"},
                           {"role": "user", "content": "Food and walking. Give me a day-by-day plan."}]),
    ("list", "none", [{"role": "user", "content": "List ten board games that work well for two players, one line each "
                                                   "saying why."}]),
    ("summary", "none", [{"role": "user", "content": "Summarize the plot of Romeo and Juliet in five sentences."}]),
    ("math", "high", [{"role": "user", "content": "A train leaves at 9:40 and arrives at 13:05, stopping twice for 7 "
                                                   "minutes each. What was its moving time? Show your steps."}]),
]
res = {}


def one(name, effort, msgs):
    body = {"model": model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature,
            "reasoning_effort": effort}
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]
    res[name] = (effort, m.get("reasoning_content") or m.get("reasoning") or "", m.get("content") or "",
                 r["usage"]["completion_tokens"], r["choices"][0]["finish_reason"], time.perf_counter() - t)


th = [threading.Thread(target=one, args=p) for p in P]
[t.start() for t in th]; [t.join() for t in th]
with open(a.out, "w") as f:
    for name, _, _ in P:
        effort, think, text, n, fin, dt = res[name]
        f.write(f"## {name} (reasoning_effort {effort}; {n} tokens, {fin}, {dt:.1f} s)\n\n")
        if think:
            f.write("<reasoning>\n" + think.strip() + "\n</reasoning>\n\n")
        f.write(text.strip() + "\n\n")
print(f"wrote {a.out}: " + ", ".join(f"{k} {v[3]} tok {v[4]}" for k, v in res.items()))
