"""Expert placement for ink2/split.py from routing counts, by the COUNT rule: each layer's experts are ranked by rows
(most used first), and the experts kept in GPU memory are filled in blocks of STEP from one HBM byte budget, best rows
per expert first. That keeps the bytes read from Grace memory per step lowest, which is what the step time follows.
Blocks keep the number of distinct hot counts small: vLLM's route kernel compiles once per count and graph size.

   python3 mkplace.py COUNTS.json HBM_GiB OUT.json [STEP=64]
   LAYOUT=native (default: NVFP4 layers 0-39, MXFP8 layers 40-47, as the official checkpoint) | nvfp4 | mxfp8
   (the bytes per expert of every layer, e.g. when an INK2_EXPERT_OVERRIDE serves layers 40-47 as NVFP4)

COUNTS.json: {"<layer>": [rows per checkpoint expert id]}, as INK2_COUNT writes (placement/counts-gen.json: 267K
generated tokens of a general corpus)."""
import json
import os
import sys

C = json.load(open(sys.argv[1]))
budget = float(sys.argv[2]) * 2 ** 30
step = int(sys.argv[4]) if len(sys.argv) > 4 else 64
L, E = 48, 384
NV, MX = 17915904, 32845824          # bytes per expert: NVFP4, MXFP8
LAYOUT = os.environ.get("LAYOUT", "native")
b = {"native": [NV if l < 40 else MX for l in range(L)], "nvfp4": [NV] * L, "mxfp8": [MX] * L}[LAYOUT]
order = {l: sorted(range(E), key=lambda e: (-C[str(l)][e], e)) for l in range(L)}
blocks = []
for l in range(L):
    for s in range(0, E, step):
        blocks.append((sum(C[str(l)][e] for e in order[l][s:s + step]) / step, l, s))
hot = [0] * L
used = 0.0
for v, l, s in sorted(blocks, key=lambda t: -t[0]):
    if hot[l] == s and used + b[l] * step <= budget:
        hot[l] += step
        used += b[l] * step
json.dump({"layers": {str(l): order[l] for l in range(L)}, "hot": {str(l): hot[l] for l in range(L)}},
          open(sys.argv[3], "w"))
tok = sum(C["1"]) / 8.0
tot = sum(sum(C[str(l)]) for l in range(L))
cap = sum(sum(C[str(l)][e] for e in order[l][:hot[l]]) for l in range(L))
cold1 = sum(C[str(l)][e] / tok * b[l] for l in range(L) for e in order[l][hot[l]:])
print(f"{LAYOUT}: HBM {used / 2**30:.1f} GiB of experts, hot sizes {sorted(set(hot))}, layers 40-47: {hot[40:]}, "
      f"hot row mass {cap / tot:.3f}, Grace bytes per single-row step {cold1 / 1e9:.2f} GB")
