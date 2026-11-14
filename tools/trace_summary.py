"""Summary of a vLLM torch-profiler trace (Chrome JSON, .json.gz) of decode steps: GPU kernel time per device grouped by
kernel name, and per stream the busy time and the idle gaps between consecutive kernels (launch / dependency latency).
   python3 trace_summary.py TRACE.json.gz [STEPS] [TOP]       STEPS: decode steps in the window (per-step figures)"""
import collections
import gzip
import json
import sys

path = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 0
top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
    tr = json.load(f)
ev = tr["traceEvents"] if isinstance(tr, dict) else tr
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
print(f"{len(kern)} GPU events")
by_dev = collections.defaultdict(list)
for e in kern:
    a = e.get("args", {})
    by_dev[a.get("device", e.get("pid"))].append(e)


def short(n):
    n = n.replace("void ", "")
    return (n[:110] + "...") if len(n) > 113 else n


for dev, es in sorted(by_dev.items(), key=lambda x: str(x[0])):
    es.sort(key=lambda e: e["ts"])
    t0, t1 = es[0]["ts"], max(e["ts"] + e["dur"] for e in es)
    busy = sum(e["dur"] for e in es)
    per = f", per step {busy / steps:.1f} us of kernels, span {(t1 - t0) / steps:.1f} us" if steps else ""
    print(f"\n=== device {dev}: {len(es)} kernels over {(t1 - t0) / 1e3:.1f} ms, kernel time {busy / 1e3:.1f} ms{per}")
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in es:
        k = agg[short(e["name"])]
        k[0] += 1
        k[1] += e["dur"]
    print(f"{'total us':>10} {'share':>6} {'count':>7} {'mean us':>8}  kernel")
    for name, (n, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
        print(f"{d:10.0f} {d / busy * 100:5.1f}% {n:7d} {d / n:8.2f}  {name}")
    streams = collections.defaultdict(list)
    for e in es:
        streams[e.get("args", {}).get("stream", e.get("tid"))].append(e)
    for s, xs in sorted(streams.items(), key=lambda x: -len(x[1]))[:4]:
        xs.sort(key=lambda e: e["ts"])
        gaps = [b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(xs, xs[1:])]
        small = [g for g in gaps if 0 < g < 200]
        big = [g for g in gaps if g >= 200]
        b = sum(e["dur"] for e in xs)
        print(f"  stream {s}: {len(xs)} kernels, busy {b / 1e3:.1f} ms; gaps < 200 us: {len(small)} totalling "
              f"{sum(small) / 1e3:.1f} ms (mean {sum(small) / max(len(small), 1):.2f} us); gaps >= 200 us: {len(big)} "
              f"totalling {sum(big) / 1e3:.1f} ms")
