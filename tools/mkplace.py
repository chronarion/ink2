"""Expert placement for ink2/split.py from routing counts. Experts are ranked by rows per layer, and the hot count per
layer is filled in blocks of STEP experts from one shared budget (best rows per byte first), so skewed layers get fewer
hot experts and flat layers more. STEP keeps the number of distinct hot sizes small: the routing kernel compiles once
per size and graph shape.
   python3 mkplace.py COUNTS.json HBM_GiB OUT.json [STEP=32]"""
import json, sys
C = json.load(open(sys.argv[1])); budget = float(sys.argv[2]) * 2 ** 30
step = int(sys.argv[4]) if len(sys.argv) > 4 else 32
L, E = 48, 384
NV, MX = 17915904, 32845824
b = [NV if l < 40 else MX for l in range(L)]
order = {l: sorted(range(E), key=lambda e: (-C[str(l)][e], e)) for l in range(L)}
blocks = []
for l in range(L):
    for s in range(0, E, step):
        rows = sum(C[str(l)][e] for e in order[l][s:s + step])
        blocks.append((rows / (b[l] * step), l, s))
hot = [0] * L; used = 0.0
# Blocks within a layer rank in descending order already (rows sorted), so a greedy pass keeps them contiguous.
for v, l, s in sorted(blocks, key=lambda t: -t[0]):
    if hot[l] == s and used + b[l] * step <= budget:
        hot[l] += step; used += b[l] * step
out = {"layers": {str(l): order[l] for l in range(L)}, "hot": {str(l): hot[l] for l in range(L)}}
tot = sum(sum(C[str(l)]) for l in range(L)); cap = sum(sum(C[str(l)][e] for e in order[l][:hot[l]]) for l in range(L))
json.dump(out, open(sys.argv[3], "w"))
print(f"HBM {used / 2**30:.1f} GiB of experts, hot sizes {sorted(set(hot))}, mean {sum(hot) / L:.1f}, "
      f"hot row mass {cap / tot:.3f}")
