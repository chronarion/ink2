"""Python side of moe3i (the RTX band of EXL3 / TR3 trellis experts for Inkling 2): build, weight tables, call.

    m = load()                                   # JIT build for sm_120a (cached in ./build)
    band = Band(m, experts, dev, shared=None)    # experts: list of {"gate"|"up"|"down": {"trellis", "suh", "svh"}}
                                                 # shared: {"gate_up_suh": fp16 [4608], "down_svh": fp16 [4608]} or None
    out = band(x, ids, gam)                      # x bf16 [R, 4608], ids int32 [R, 8] (local ids, -1 = not in the band),
                                                 # gam fp32 [R, 8]  ->  fp32 [R, 4608]
Bit widths come from the trellis shapes (16 K int16 per tile, K = 3, 4 or 5, per expert and projection). With a shared
vector, the experts' own copy of it may be omitted (TR3v4 stores gate_up_suh / down_svh once per layer).
"""
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H, I, TOPK = 4608, 2304, 8
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_120a,code=sm_120a"]
SHAPES = {"gate": (H, I), "up": (H, I), "down": (I, H)}          # (in, out)


def load(name="moe3i", extra=(), build=None):
    from torch.utils.cpp_extension import load as _load
    bd = build or os.path.join(HERE, "build", name)
    os.makedirs(bd, exist_ok=True)
    return _load(name=name, sources=[os.path.join(HERE, "moe3i.cu")], build_directory=bd, verbose=False,
                 extra_cuda_cflags=FLAGS + list(extra))


def tensor_bits(trellis, proj):
    k, n = SHAPES[proj]
    assert trellis.dtype == torch.int16 and trellis.dim() == 3 and trellis.shape[:2] == (k // 16, n // 16), \
        f"{proj}: trellis {tuple(trellis.shape)}"
    K = trellis.shape[2] // 16
    assert trellis.shape[2] == 16 * K and 3 <= K <= 5, f"{proj}: {trellis.shape[2]} int16 per tile (K 3..5 supported)"
    return K


class Band:
    """One layer's band: per local id the trellis / suh / svh of gate, up, down (device tensors; several local ids may
    share one expert's tensors), the bit-width table, optionally the layer-shared gate / up input vector and down output
    vector. Holds the pointer tables and the scratch (one scratch can serve every layer: the chain leaves it clean)."""

    def __init__(self, m, experts, dev, scratch=None, shared=None):
        self.m, self.dev, self.nb = m, dev, len(experts)
        assert 1 <= self.nb <= m.NBMAX, f"band size {self.nb} > {m.NBMAX}"
        shared = shared or {}
        self.gsu = shared.get("gate_up_suh")
        self.dsv = shared.get("down_svh")
        for name, v in (("gate_up_suh", self.gsu), ("down_svh", self.dsv)):
            if v is not None:
                assert v.dtype == torch.float16 and v.numel() == H, f"{name}: fp16 [{H}]"
        self.gsu = self.gsu.to(dev).contiguous().view(-1) if self.gsu is not None else None
        self.dsv = self.dsv.to(dev).contiguous().view(-1) if self.dsv is not None else None
        self.keep, self.tabs = [], []
        kb = [[tensor_bits(e[p]["trellis"], p) for e in experts] for p in ("gate", "up", "down")]
        for proj in ("gate", "up", "down"):
            for key in ("trellis", "suh", "svh"):
                ts = []
                for e in experts:
                    t = e[proj].get(key)
                    if t is None:          # a vector the layer shares: the shared copy's pointer (never read per expert)
                        assert (key == "suh" and proj != "down" and self.gsu is not None) or \
                               (key == "svh" and proj == "down" and self.dsv is not None), f"{proj}.{key} missing"
                        t = self.gsu if key == "suh" else self.dsv
                    t = t.to(dev).contiguous()
                    ts.append(t)
                self.keep += ts
                self.tabs.append(torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev))
        self.kbits = torch.tensor(kb, dtype=torch.int8, device=dev).contiguous()
        self.kmax = max(max(k) for k in kb)
        self.scr = scratch if scratch is not None else m.scratch(dev)
        self.empty = torch.empty(0, dtype=torch.int32, device=dev)
        self.empty_h = torch.empty(0, dtype=torch.float16, device=dev)

    def forward(self, x, ids, gam, out=None, rows=None, round_bf16=False):
        """rows: optional int32 [1] device word with the live row count (<= x.shape[0]): one CUDA graph for every count"""
        if out is None:
            out = torch.empty(x.shape[0], H, dtype=torch.float32, device=self.dev)
        g, u, d = self.tabs[0:3], self.tabs[3:6], self.tabs[6:9]
        self.m.forward(x, ids, gam, self.nb, self.kbits, self.kmax, *g, *u, *d,
                       self.gsu if self.gsu is not None else self.empty_h, self.dsv if self.dsv is not None else self.empty_h,
                       *self.scr, out, self.empty if rows is None else rows, round_bf16)
        return out

    __call__ = forward
