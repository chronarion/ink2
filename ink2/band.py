"""The RTX band (bundle tier 2): each layer's warm experts as 3-bit EXL3 / TR3 trellis in the RTX PRO 6000's memory,
computed there in place (rtx/moe3i.cu); only activations cross PCIe.

Placement (INK2_PLACEMENT) gains "warm": {"<layer>": W}: positions [0, H) hot (GB300 memory), [H, H + W) warm (RTX),
[H + W, 384) tail (Grace memory, read in place). Decode-sized MoE calls (<= INK2_BAND_MAX tokens, default 64) split
three ways; larger calls (prefill chunks) keep today's path, where the GB300 computes every non-hot expert natively
from Grace memory (the warm experts keep their native copies there for that).

Per decode-sized call, inside vLLM's own CUDA graphs (a graph captured on the GB300 may contain RTX work joined by
events; rtx-probe/xdev_graph.py):
  GB300 side stream: route() with the band's bounds [H, H + W) (the same kernel the other parts run, so all three agree
      on every token's selection), then one kernel writes each row (bf16) and its band picks (local id, -1 elsewhere;
      routing weight) to the layer's pinned host slot; an event;
  RTX stream: waits for the event, copies the slot into its memory, runs the band (moe3i), copies the fp32 partial back
      into the slot; an event;
  GB300: the combine waits for that event and adds the partial before its single rounding (combine2 Extra).
  INK2_BAND=<dir with layer{L}.safetensors: TR3 tensors model.layers.L.mlp.experts.E.{gate,up,down}_proj.{trellis,
  suh,svh}, E = checkpoint expert id> | random (timing only: random trellis, wrong outputs), INK2_RTX_DEVICE (cuda:1),
  INK2_BAND_MAX (64), INK2_BAND_CHECK=1 (count every slot's tiers: each selected expert exactly once, else error)"""
import os
import sys

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

DIR = os.environ.get("INK2_BAND", "")
ENABLE = bool(DIR)
MAXR = int(os.environ.get("INK2_BAND_MAX", "64"))
CHECK = os.environ.get("INK2_BAND_CHECK", "0") == "1"
_LOG = lambda m: sys.stderr.write(f"INK2_BAND {m}\n")
HID = 4608
_STATE = {}


def _rtx():
    if "dev" not in _STATE:
        dev = torch.device(os.environ.get("INK2_RTX_DEVICE", "cuda:1"))
        assert "RTX" in torch.cuda.get_device_name(dev), f"INK2_RTX_DEVICE {dev} is {torch.cuda.get_device_name(dev)}"
        sys.path.insert(0, os.environ.get("INK2_RTX_HOME", "/ink2rtx"))
        import moe3i
        with torch.cuda.device(dev):
            m = moe3i.load(build=os.environ.get("INK2_RTX_BUILD", "/root/.cache/vllm/moe3i"))
            _STATE.update(dev=dev, moe3i=moe3i, m=m, scratch=m.scratch(dev), stream=torch.cuda.Stream(dev))
        _LOG(f"RTX band on {torch.cuda.get_device_name(dev)} ({dev})")
    return _STATE


@triton.jit
def _stage_kernel(XS, XD, IS, ID, GS, GD, T, NX: tl.constexpr, NI: tl.constexpr, BLOCK: tl.constexpr):
    """RTX: the slot (pinned host memory, read through the RTX's mapping) into RTX memory, by SMs: x rows (bf16 words as
    int16), ids, gammas. M6 measured the copy-engine path +17-24 us per layer slower while the GB300 streams host memory,
    the SM path +1-7 us, and its round trip 11-13 us shorter."""
    pid = tl.program_id(0)
    nx = T * NX
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nx
    tl.store(XD + offs, tl.load(XS + offs, m), m)
    if pid == 0:
        k = tl.arange(0, 512)
        mi = k < T * NI
        tl.store(ID + k, tl.load(IS + k, mi), mi)
        tl.store(GD + k, tl.load(GS + k, mi), mi)


@triton.jit
def _send_kernel(X, Src, Gam, Tiles, XH, IdsH, GamH, TILE: tl.constexpr, TOPK: tl.constexpr, H: tl.constexpr,
                 BLOCK: tl.constexpr):
    """one program per token: its row to the pinned slot (bf16), its band picks (local expert of the layout row's tile,
    -1 elsewhere) and routing weights"""
    t = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, TOPK)
    r = tl.load(Src + t * TOPK + k).to(tl.int64)
    inb = r >= 0
    e = tl.load(Tiles + tl.where(inb, r, 0) // TILE, inb, other=-1)
    g = tl.load(Gam + tl.where(inb, r, 0), inb, other=0.0)
    tl.store(IdsH + t * TOPK + k, tl.where(inb, e, -1))
    tl.store(GamH + t * TOPK + k, g)
    for c in tl.static_range(0, H, BLOCK):        # (unmasked with BLOCK 1024 until 11-14: 4608 = 4.5 blocks, so the last
        cols = c + tl.arange(0, BLOCK)               # row read 1 KB past x and faulted when x ended a mapped segment)
        m = cols < H
        tl.store(XH + t * H + cols, tl.load(X + t * H + cols, m), m)


class LayerBand:
    """one layer's band: TR3 weights on the RTX, the pinned slot, the events"""

    def __init__(self, layer, H, W, perm, topk, gb300):
        S = _rtx()
        self.layer, self.H, self.W, self.topk = layer, H, W, topk
        ids = perm[H:H + W].tolist()          # local band id -> checkpoint expert
        experts, shared = _load_experts(layer, ids, S)
        with torch.cuda.device(S["dev"]):
            shared = {k: v.to(S["dev"]) for k, v in shared.items()} or None
            self.band = S["moe3i"].Band(S["m"], experts, S["dev"], scratch=S["scratch"], shared=shared)
            self.x_r = torch.empty(MAXR, HID, dtype=torch.bfloat16, device=S["dev"])
            self.ids_r = torch.empty(MAXR, topk, dtype=torch.int32, device=S["dev"])
            self.gam_r = torch.empty(MAXR, topk, dtype=torch.float32, device=S["dev"])
            self.y_r = torch.empty(MAXR, HID, dtype=torch.float32, device=S["dev"])
        # the slot: pinned (portable: every context sees it pinned) host memory, GB300-side device views
        # (vLLM builds the model under a CUDA default device: every host tensor names its device)
        self.x_h = torch.zeros(MAXR, HID, dtype=torch.bfloat16, device="cpu").pin_memory()
        self.ids_h = torch.full((MAXR, topk), -1, dtype=torch.int32, device="cpu").pin_memory()
        self.gam_h = torch.zeros(MAXR, topk, dtype=torch.float32, device="cpu").pin_memory()
        self.y_h = torch.zeros(MAXR, HID, dtype=torch.float32, device="cpu").pin_memory()
        with torch.cuda.device(gb300):
            self.x_g = get_accelerator_view_from_cpu_tensor(self.x_h)
            self.ids_g = get_accelerator_view_from_cpu_tensor(self.ids_h)
            self.gam_g = get_accelerator_view_from_cpu_tensor(self.gam_h)
            self.y_g = get_accelerator_view_from_cpu_tensor(self.y_h)
        with torch.cuda.device(S["dev"]):
            self.x_hr = get_accelerator_view_from_cpu_tensor(self.x_h)
            self.ids_hr = get_accelerator_view_from_cpu_tensor(self.ids_h)
            self.gam_hr = get_accelerator_view_from_cpu_tensor(self.gam_h)
            self.y_hr = get_accelerator_view_from_cpu_tensor(self.y_h)
        self.ev_go, self.ev_done = torch.cuda.Event(), torch.cuda.Event()
        from vllm.models.inkling2.ops.moe import NUM_COUNTERS
        with torch.cuda.device(gb300):    # the band's own route() state (bounds [H, H + W))
            self.bounds = torch.tensor([H, H + W], dtype=torch.int32, device=gb300)
            self.n_pad = 256 if W <= 256 else -(-W // 64) * 64   # one compile key for every layer's band route
            self.counts = torch.zeros(self.n_pad, dtype=torch.int32, device=gb300)
            self.counters = torch.zeros(NUM_COUNTERS, dtype=torch.int32, device=gb300)

    def send(self, x, src, gam, tiles, tile):
        T = x.shape[0]
        _send_kernel[(T,)](x, src, gam, tiles, self.x_g, self.ids_g, self.gam_g, tile, self.topk, HID, 512,
                           num_warps=4)

    def run(self, T):
        """on the RTX stream after ev_go: stage the slot into RTX memory (one SM kernel), run the band, which writes its
        fp32 partial straight into the slot through the RTX's mapping; ev_done after that"""
        S = _STATE
        with torch.cuda.device(S["dev"]), torch.cuda.stream(S["stream"]):
            S["stream"].wait_event(self.ev_go)
            nx = T * HID
            _stage_kernel[(triton.cdiv(nx, 4096),)](self.x_hr.view(torch.int16), self.x_r.view(torch.int16),
                                                   self.ids_hr, self.ids_r, self.gam_hr, self.gam_r, T,
                                                   HID, self.topk, 4096, num_warps=8)
            self.band(self.x_r[:T], self.ids_r[:T], self.gam_r[:T], out=self.y_hr[:T])
            self.ev_done.record(S["stream"])


def _load_experts(layer, ids, S):
    if DIR == "random":
        g = torch.Generator(device="cpu").manual_seed(layer)
        I = 2304

        def proj(i, o):
            return {"trellis": torch.randint(-32768, 32767, (i // 16, o // 16, 48), dtype=torch.int16, generator=g,
                                             device="cpu"),
                    "suh": (torch.randn(i, generator=g, device="cpu") * 0.05).half(),
                    "svh": (torch.randn(o, generator=g, device="cpu") * 0.05).half()}
        one = {"gate": proj(HID, I), "up": proj(HID, I), "down": proj(I, HID)}
        return [one] * len(ids), {}      # one random expert shared by every local id (timing only)
    from safetensors import safe_open
    import json
    path = os.path.join(DIR, f"layer{layer}.safetensors")
    out, shared = [], {}
    with safe_open(path, "pt", device="cpu") as f:
        names = set(f.keys())
        meta = json.loads((f.metadata() or {}).get("band", "{}"))
        # the placement's warm experts must be among the file's (a placement may move the band's top experts into HBM,
        # e.g. when GPTQ NVFP4 layers 40-47 free GPU memory; the rest keep their TR3 weights)
        if meta.get("warm") is not None and not set(int(e) for e in ids) <= set(int(e) for e in meta["warm"]):
            raise RuntimeError(f"{path}: this placement's warm experts are not all in the band file (layer {layer}): "
                               "requantize or use a placement built from the file's")
        for e in ids:
            d = {}
            for p in ("gate", "up", "down"):
                d[p] = {k: f.get_tensor(f"model.layers.{layer}.mlp.experts.{e}.{p}_proj.{k}")
                        for k in ("trellis", "suh", "svh")}
            out.append(d)
        for k in ("gate_up_suh", "down_svh"):          # the layer's shared residual-side vectors (TR3v4), if present
            name = f"model.layers.{layer}.mlp.experts.{k}"
            if name in names:
                shared[k] = f.get_tensor(name)
    return out, shared
