"""Inkling 2 Small on one GB300: routed experts split between GPU memory and Grace memory.

Each MoE layer's 384 experts are renumbered so that the H most used ones ("hot") come first: the gate's rows and bias
are permuted with the expert weights, so routing is unchanged. Experts [0, H) live in GPU memory; experts [H, 384) live
in pinned host memory and the same NVFP4 / MXFP8 kernel reads them in place over NVLink-C2C. Both parts run the
image's own route / bmm kernels with their own expert bounds (for calls up to INK2_PAR_MAX tokens the Grace part on a
second stream); one combine (combine2.py) adds each token's 8 expert rows in fp32 in selection order and rounds once,
so the output does not depend on which part holds an expert.

Placement (INK2_PLACEMENT, JSON): {"layers": {"<layer>": [expert ids, most used first]}, "hot": {"<layer>": H}}.
Without it the order is the identity and H = INK2_HOT (default 192) on every layer. One tensor-parallel rank only.
Distinct hot counts should be few (tools/mkplace.py uses blocks of 64): vLLM's route kernel compiles once per count and
graph size.
"""
import json
import os
import sys

import torch
from torch import nn

from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling2 import moe as _moe
from vllm.models.inkling2.ops.block_scales import shuffle_rows, swizzle_block_scales
from vllm.models.inkling2.ops.moe import NUM_COUNTERS, combine, route
from vllm.models.inkling2.ops.activation import swiglu_quantize
from vllm.models.inkling2.ops.trtllm_bmm import bmm, expert_plan, load_kernels

from ink2.combine2 import combine2

_LOG = lambda m: sys.stderr.write(f"INK2_SPLIT {m}\n")
_PLACEMENT = None
# INK2_COUNT=path: rows routed to each expert by decode-sized calls (<= INK2_COUNT_MAX tokens, default 64: prefill
# sweeps every expert whatever the placement, so only generated tokens matter), accumulated per layer on the GPU inside
# the CUDA graphs and written (checkpoint expert ids) every 30 s by a thread. Measurement only.
_COUNT = os.environ.get("INK2_COUNT", "")
_COUNT_MAX = int(os.environ.get("INK2_COUNT_MAX", "64"))
_COUNTERS = {}
_COUNT_THREAD = []


def _count_dump():
    out = {}
    for layer, (perm, acc) in sorted(_COUNTERS.items()):
        v = acc.cpu().tolist()
        row = [0] * len(v)
        for new, old in enumerate(perm.tolist()):
            row[old] = int(v[new])
        out[str(layer)] = row
    with open(_COUNT + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(_COUNT + ".tmp", _COUNT)


def _count_start(device):
    if _COUNT_THREAD:
        return
    import threading, time

    def loop():
        torch.cuda.set_device(device)
        with torch.cuda.stream(torch.cuda.Stream(device)):
            while True:
                time.sleep(30)
                try:
                    _count_dump()
                except Exception as e:  # noqa: BLE001
                    _LOG(f"count dump failed: {e!r}")
    t = threading.Thread(target=loop, daemon=True, name="ink2-count")
    t.start()
    _COUNT_THREAD.append(t)


# INK2_PARALLEL=1: the Grace part runs on a side stream while the GPU-memory part runs (C2C and HBM reads overlap)
_PARALLEL = os.environ.get("INK2_PARALLEL", "1") == "1"
# only up to this many tokens (decode, verify): prefill chunks are bound by the sweep of every expert anyway, and a
# side stream keeps its own allocator pool (the profile peak at 8,192 tokens grew 3.6 GiB)
_PAR_MAX = int(os.environ.get("INK2_PAR_MAX", "512"))
_BAND_MAX = int(os.environ.get("INK2_BAND_MAX", "64"))
# smaller calls skip the RTX band and read the warm experts' native copies in place from Grace memory, as the two-tier
# engine does (measured 11-14: 23% slower at one row than the band's round trip, so by default no call skips it)
_BAND_MIN_T = int(os.environ.get("INK2_BAND_MIN_T", "1"))
# vLLM's route kernel takes the number of local experts as a compile-time constant (and the plan's tile count follows
# it): every distinct count compiles again for every graph size. The route state and the plan use the count rounded
# up to a block of 64; the expert bounds stay exact, so the extra experts never get rows.
_PAD = int(os.environ.get("INK2_PAD", "64"))
# INK2_ROUTE3 (default 1; 0: the three routes): band calls of up to 16 tokens route once (ink2/route3.py: the hot and tail layouts and the band slot
# from one router GEMV and top-k) instead of three route launches and the band's send
_ROUTE3 = os.environ.get("INK2_ROUTE3", "1") == "1"


def _pad(n):
    return -(-n // _PAD) * _PAD if n else 0
_SIDE = []


def _side_stream():
    if not _SIDE:
        _SIDE.append(torch.cuda.Stream())
    return _SIDE[0]


def _placement():
    global _PLACEMENT
    if _PLACEMENT is None:
        path = os.environ.get("INK2_PLACEMENT", "")
        _PLACEMENT = json.load(open(path)) if path else {"layers": {}, "hot": {}}
        if path:
            _LOG(f"placement {path}: {len(_PLACEMENT['layers'])} layers")
    return _PLACEMENT


def _override_file(layer):
    """(path, format) of this layer's expert override, or None"""
    d = os.environ.get("INK2_EXPERT_OVERRIDE", "")
    path = os.path.join(d, f"layer{layer}.safetensors") if d else ""
    if not path or not os.path.exists(path):
        return None
    from safetensors import safe_open
    with safe_open(path, "pt", device="cpu") as f:
        fmt = (f.metadata() or {}).get("format", "nvfp4")
    assert fmt in ("nvfp4", "mxfp8"), f"{path}: format {fmt}"
    return path, fmt


def _host_tensor(shape, dtype, device):
    """a tensor in registered (pinned) host memory, returned as a device view the kernels can read in place"""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    h = torch.empty(shape, dtype=dtype, device="cpu")   # (vLLM builds the model under a CUDA default device)
    if h.numel():
        err = torch.cuda.cudart().cudaHostRegister(h.data_ptr(), h.numel() * h.element_size(), 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {err}")
    return h, (get_accelerator_view_from_cpu_tensor(h) if h.numel() else torch.empty(shape, dtype=dtype, device=device))


class _Part(nn.Module):
    """experts [start, start + n) of the renumbered layer, in GPU memory or in host memory; buffers, not parameters
    (the checkpoint's names are the parent's placeholders)"""

    def __init__(self, n, start, nvfp4, shapes, topk, gate_cap, up_cap, host):
        super().__init__()
        self.n, self.start, self.nvfp4, self.shapes, self.topk = n, start, nvfp4, shapes, topk
        self.n_pad = _pad(n)
        self.gate_cap, self.up_cap, self.host = gate_cap, up_cap, host
        dev = torch.device("cuda", torch.cuda.current_device())
        pack, group = (2, 16) if nvfp4 else (1, 32)
        wdt = torch.uint8 if nvfp4 else torch.float8_e4m3fn
        sdt = torch.float8_e4m3fn if nvfp4 else torch.uint8
        self._host = {}
        for name, (rows, k) in zip(("w13", "w2"), shapes):
            for suffix, shape, dt in (("weight", (n, rows, k // pack), wdt), ("weight_scale", (n, rows, k // group), sdt)):
                if host:
                    h, t = _host_tensor(shape, dt, dev)
                    self._host[f"{name}_{suffix}"] = h
                else:
                    t = torch.empty(shape, dtype=dt, device=dev)
                self.register_buffer(f"{name}_{suffix}", t, persistent=False)
            if nvfp4:   # small: always in GPU memory
                self.register_buffer(f"{name}_weight_amax", torch.empty(n, dtype=torch.float32, device=dev),
                                     persistent=False)
                self.register_buffer(f"{name}_weight_outer_scale_denominator",
                                     torch.empty(1, dtype=torch.float32, device=dev), persistent=False)
        self.register_buffer("expert_bounds", torch.tensor([start, start + n], dtype=torch.int32, device=dev),
                             persistent=False)
        self.register_buffer("counts", torch.zeros(self.n_pad, dtype=torch.int32, device=dev), persistent=False)
        # own completion counters: the two parts' route kernels may run at the same time (INK2_PARALLEL)
        self.register_buffer("counters", torch.zeros(NUM_COUNTERS, dtype=torch.int32, device=dev), persistent=False)
        self.register_buffer("ones", torch.ones(self.n_pad, dtype=torch.float32, device=dev), persistent=False)

    @classmethod
    def view(cls, parent, off):
        """experts [parent.start + off, parent.start + parent.n) of a parent part, sharing its buffers"""
        v = cls.__new__(cls)
        nn.Module.__init__(v)
        v.n, v.start = parent.n - off, parent.start + off
        v.n_pad = _pad(v.n)
        v.nvfp4, v.shapes, v.topk = parent.nvfp4, parent.shapes, parent.topk
        v.gate_cap, v.up_cap, v.host = parent.gate_cap, parent.up_cap, parent.host
        v._host = {k: t[off:] for k, t in parent._host.items()}
        dev = parent.expert_bounds.device
        for name in ("w13", "w2"):
            for suffix in ("weight", "weight_scale"):
                v.register_buffer(f"{name}_{suffix}", getattr(parent, f"{name}_{suffix}")[off:], persistent=False)
            if parent.nvfp4:
                v.register_buffer(f"{name}_weight_amax", getattr(parent, f"{name}_weight_amax")[off:], persistent=False)
                v.register_buffer(f"{name}_weight_outer_scale_denominator",
                                  getattr(parent, f"{name}_weight_outer_scale_denominator"), persistent=False)
        v.register_buffer("expert_bounds", torch.tensor([v.start, v.start + v.n], dtype=torch.int32, device=dev),
                          persistent=False)
        v.register_buffer("counts", torch.zeros(v.n_pad, dtype=torch.int32, device=dev), persistent=False)
        v.register_buffer("counters", torch.zeros(NUM_COUNTERS, dtype=torch.int32, device=dev), persistent=False)
        v.register_buffer("ones", torch.ones(v.n_pad, dtype=torch.float32, device=dev), persistent=False)
        return v

    def warm(self):
        if self.n:
            load_kernels((self.w13_weight, self.w2_weight), (self.w13_weight_scale, self.w2_weight_scale), self.topk)

    def plan(self, tokens):
        return expert_plan(tokens, self.topk, self.n_pad, self.nvfp4, self.shapes)

    def scales(self, name):
        if not self.nvfp4:
            return None, None
        return getattr(self, f"{name}_weight_amax"), getattr(self, f"{name}_weight_outer_scale_denominator")

    def route_only(self, x, gate, is_padding):
        """this part's routing of x: (src2dst, gammas, tiles, plan) (the Grace band's job comes from it)"""
        plan = self.plan(x.shape[0])
        routing, _ = route(x, gate.weight, gate.bias, self.counters, self.counts, gate.topk, gate.route_scale,
                           self.expert_bounds, plan, *self.scales("w13"), is_padding)
        src2dst, gammas, _, tiles = routing
        return src2dst, gammas, tiles, plan

    def _gemm(self, quantized, name, tiles, plan, projection):
        act, act_scale, row_scale = quantized
        return bmm(act, act_scale, row_scale, getattr(self, f"{name}_weight"), getattr(self, f"{name}_weight_scale"),
                   self.ones, tiles, plan, projection)

    def route_part(self, x, gate, is_padding):
        """this part's routing and quantized rows of x (the first half of forward)"""
        plan = self.plan(x.shape[0])
        routing, quantized = route(x, gate.weight, gate.bias, self.counters, self.counts, gate.topk, gate.route_scale,
                                   self.expert_bounds, plan, *self.scales("w13"), is_padding)
        return routing, quantized, plan

    def forward(self, x, gate, is_padding):
        return self.compute(*self.route_part(x, gate, is_padding))

    def compute(self, routing, quantized, plan):
        """the experts' GEMMs over route_part's layout (the second half of forward)"""
        src2dst, gammas, rows, tiles = routing
        gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
        m = plan.max_ctas
        activated = swiglu_quantize(gate_up, gammas, self.gate_cap, self.up_cap,
                                    (rows, tiles[2 * m + 2:], tiles[:m]), plan.tile, *self.scales("w2"))
        del gate_up
        down = self._gemm(activated, "w2", tiles, plan, 1)
        return down, src2dst, tiles


class SplitExperts(nn.Module):
    """the checkpoint's expert parameters as placeholders whose loaders fill the two parts (permuted, shuffled and
    swizzled as the image's Inkling2Experts does)"""

    def __init__(self, config, quant_format, layer_id):
        super().__init__()
        E = config.n_routed_experts
        pl = _placement()
        order = pl["layers"].get(str(layer_id)) or list(range(E))
        assert sorted(order) == list(range(E)), f"layer {layer_id}: placement is not a permutation"
        H = int(pl["hot"].get(str(layer_id), os.environ.get("INK2_HOT", "192")))
        self.layer_id, self.E, self.H = layer_id, E, H
        self.perm = torch.tensor(order, dtype=torch.long, device="cpu")   # new index -> checkpoint expert
        # INK2_EXPERT_OVERRIDE=dir: layer{L}.safetensors replaces this layer's routed experts (checkpoint names and raw
        # layout, all experts, metadata "format": nvfp4 | mxfp8), possibly in another format than the checkpoint's; the
        # checkpoint's own tensors for the layer are then accepted and dropped
        self._ovr = _override_file(layer_id)
        ckpt_format = quant_format
        if self._ovr is not None:
            quant_format = self._ovr[1]
        self.nvfp4 = quant_format == "nvfp4"
        ckpt_nvfp4 = ckpt_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        shapes = ((2 * inter, hidden), (hidden, inter))
        args = (self.nvfp4, shapes, config.num_experts_per_tok, config.mlp_gate_softcap, config.mlp_up_softcap)
        self.hot = _Part(H, 0, *args, host=False)
        self.cold = _Part(E - H, H, *args, host=True)
        from ink2 import debug as _dbg
        _dbg.register(layer_id, "hot", self.hot)
        _dbg.register(layer_id, "cold", self.cold)
        # the RTX band (ink2/band.py): positions [H, H + W) also run as TR3 on the RTX for decode-sized calls; the
        # Grace part keeps their native copies for prefill; decode reads only the tail [H + W, E) from Grace memory
        from ink2 import band as _band
        W = int(pl.get("warm", {}).get(str(layer_id), 0)) if _band.ENABLE else 0
        self.W, self.band, self.tail = W, None, None
        if W:
            assert 0 < W < E - H, f"layer {layer_id}: warm band {W} does not fit beside {H} hot experts"
            self.tail = _Part.view(self.cold, W)
            self.band = _band.LayerBand(layer_id, H, W, self.perm, config.num_experts_per_tok,
                                        torch.device("cuda", torch.cuda.current_device()))
            from ink2 import graceband as _gb
            _gb.register(layer_id, self)
            _dbg.register(layer_id, "tail", self.tail)
            _dbg.register(layer_id, "band", self.band)
        wdt = torch.uint8 if ckpt_nvfp4 else torch.float8_e4m3fn          # placeholders: the checkpoint's names
        sdt = torch.float8_e4m3fn if ckpt_nvfp4 else torch.uint8
        for name in ("w13", "w2"):
            self._placeholder(f"{name}_weight", wdt, self._load_weight)
            self._placeholder(f"{name}_weight_scale", sdt, self._load_scale)
            if ckpt_nvfp4:
                self._placeholder(f"{name}_weight_amax", torch.float32, self._load_amax)
                self._placeholder(f"{name}_weight_outer_scale_denominator", torch.float32, self._load_denom)
        if self._ovr is not None:
            self._load_override()

    def _placeholder(self, name, dtype, loader):
        p = nn.Parameter(torch.empty(0, dtype=dtype, device="cpu"), requires_grad=False)
        p._ink2_name = name
        set_weight_attrs(p, {"weight_loader": loader})
        self.register_parameter(name, p)

    def _split(self, name, loaded, transform):
        """loaded [E, ...] -> permuted -> parts (transformed per expert, in chunks on the GPU)"""
        dev = self.hot.expert_bounds.device
        for part, lo, hi in ((self.hot, 0, self.H), (self.cold, self.H, self.E)):
            dst = getattr(part, name)
            host = part._host.get(name)
            for c0 in range(lo, hi, 16):
                c1 = min(c0 + 16, hi)
                src = loaded[self.perm[c0:c1]].to(dev, non_blocking=False)
                out = transform(src) if transform is not None else src
                if host is not None:
                    host[c0 - lo:c1 - lo].copy_(out.cpu())
                else:
                    dst[c0 - lo:c1 - lo].copy_(out)

    def _load_override(self):
        from safetensors import safe_open
        path, fmt = self._ovr
        pre = f"model.llm.layers.{self.layer_id}.mlp.experts."
        with safe_open(path, "pt", device="cpu") as f:
            for name in ("w13", "w2"):
                self._split(f"{name}_weight", f.get_tensor(pre + f"{name}_weight"), shuffle_rows)
                self._split(f"{name}_weight_scale", f.get_tensor(pre + f"{name}_weight_scale"),
                            lambda t: swizzle_block_scales(shuffle_rows(t)))
                if fmt == "nvfp4":
                    self._split(f"{name}_weight_amax", f.get_tensor(pre + f"{name}_weight_amax"), None)
                    d = f.get_tensor(pre + f"{name}_weight_outer_scale_denominator").reshape(1)
                    for part in (self.hot, self.cold):
                        getattr(part, f"{name}_weight_outer_scale_denominator").copy_(d)
        _LOG(f"layer {self.layer_id}: experts from {path} ({fmt})")

    def _load_weight(self, param, loaded):
        if self._ovr is not None:
            return
        self._split(param._ink2_name, loaded, shuffle_rows)

    def _load_scale(self, param, loaded):
        if self._ovr is not None:
            return
        self._split(param._ink2_name, loaded, lambda t: swizzle_block_scales(shuffle_rows(t)))

    def _load_amax(self, param, loaded):
        if self._ovr is not None:
            return
        self._split(param._ink2_name, loaded, None)

    def _load_denom(self, param, loaded):
        if self._ovr is not None:
            return
        for part in (self.hot, self.cold):
            getattr(part, param._ink2_name).copy_(loaded.reshape(1))


class SplitMoE(nn.Module):
    """drop-in for the image's Inkling2MoE (one rank, reduce=True only)"""

    def __init__(self, config, quant_format, layer_id):
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = _moe.Inkling2Gate(config)
        self.experts = SplitExperts(config, quant_format, layer_id)
        perm = self.experts.perm

        def gate_loader(param, loaded, _perm=perm):
            param.data.copy_(loaded[_perm].to(param.device))
        for p in (self.gate.weight, self.gate.bias):
            p.weight_loader = gate_loader        # (set_weight_attrs refuses to overwrite the default loader)
        self._warmed = False
        self.register_buffer("_count_acc", None, persistent=False)
        from ink2.follow import FOLLOW
        if _COUNT or FOLLOW is not None:
            dev = torch.device("cuda", torch.cuda.current_device())
            acc = (FOLLOW.counter(layer_id, config.num_hidden_layers, self.experts.E, dev) if FOLLOW is not None
                   else torch.zeros(self.experts.E, dtype=torch.int64, device=dev))
            self.register_buffer("_count_acc", acc, persistent=False)
            if _COUNT:
                _COUNTERS[layer_id] = (perm, self._count_acc)
            if FOLLOW is not None:
                FOLLOW.register(layer_id, self)

    def forward(self, x, reduce=True, push=None, lora=None):
        if lora is not None or push is not None or not reduce:
            raise NotImplementedError("INK2 split experts: one rank, reduced output only")
        if not self._warmed:
            self.experts.hot.warm()
            self.experts.cold.warm()
            if self.experts.tail is not None:
                self.experts.tail.warm()
            self._warmed = True
        import vllm.envs as envs
        from vllm.forward_context import get_forward_context, is_forward_context_available
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        x = x.contiguous()
        ex = self.experts
        topk = self.gate.topk
        if not ex.cold.n or not ex.hot.n:
            down, src2dst, _ = (ex.hot if ex.hot.n else ex.cold)(x, self.gate, is_padding)
            out = combine(down, src2dst, topk, x.dtype)
        elif ex.band is not None and _BAND_MIN_T <= x.shape[0] <= _BAND_MAX:
            out = self._forward_band(x, is_padding, topk)
        else:
            from ink2 import debug as _dbg
            _dbg.sync(f"before layer {ex.layer_id} T={x.shape[0]} (two-tier)")
            if _PARALLEL and x.shape[0] <= _PAR_MAX:
                main, side = torch.cuda.current_stream(), _side_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    cold = ex.cold(x, self.gate, is_padding)
                hot = ex.hot(x, self.gate, is_padding)
                main.wait_stream(side)
            else:
                hot = ex.hot(x, self.gate, is_padding)
                cold = ex.cold(x, self.gate, is_padding)
            # one combine over both parts' rows: each token's 8 rows are summed in fp32 in selection order and rounded
            # once, so the output does not depend on which part holds an expert (placement changes, INK2_FOLLOW)
            acc = self._count_acc if (self._count_acc is not None and x.shape[0] <= _COUNT_MAX) else None
            (dh, sh, th), (dc, sc, tc) = hot, cold
            _dbg.sync(f"hot + cold parts, layer {ex.layer_id} T={x.shape[0]} (two-tier)")
            out = combine2(dh, sh, th, dc, sc, tc, topk, x.dtype, ex.hot.plan(x.shape[0]).tile, ex.H, acc)
            _dbg.sync(f"combine, layer {ex.layer_id} T={x.shape[0]} (two-tier)")
            if acc is not None and _COUNT:
                _count_start(x.device)
        return out

    def _forward_band(self, x, is_padding, topk):
        """decode-sized call with the RTX band: band rows out first (side stream), the RTX computes them while the
        GB300 runs the hot part (main) and the Grace tail (side); one combine adds all three"""
        ex, b, T = self.experts, self.experts.band, x.shape[0]
        from ink2 import graceband as _gb
        from ink2 import debug as _dbg
        _dbg.sync(f"before layer {ex.layer_id} T={T}")
        _dbg.note(ex.layer_id, T, _gb.ENABLE and T >= _gb.MIN_T, x)
        main, side = torch.cuda.current_stream(), _side_stream()
        if _ROUTE3 and not _gb.ENABLE:
            from ink2 import route3 as _r3
            if T <= _r3.MAX_TOKENS:
                (ra, qa, pa), (rc, qc, pc) = _r3.route3(x, self.gate, ex.hot, ex.tail, b, is_padding)
                b.ev_go.record(main)                                # the band slot is written
                _dbg.sync(f"route3, layer {ex.layer_id} T={T}")
                b.run(T)
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    tail = ex.tail.compute(rc, qc, pc)
                hot = ex.hot.compute(ra, qa, pa)
                main.wait_stream(side)
                main.wait_event(b.ev_done)
                _dbg.sync(f"route3 parts + band, layer {ex.layer_id} T={T}")
                acc = self._count_acc if (self._count_acc is not None and T <= _COUNT_MAX) else None
                (dh, sh, th), (dt, st, tt) = hot, tail
                out = combine2(dh, sh, th, dt, st, tt, topk, x.dtype, pa.tile, ex.H + ex.W, acc, extra=b.y_g[:T])
                _dbg.note(ex.layer_id, T, False, x, out)
                return out
        side.wait_stream(main)
        with torch.cuda.stream(side):
            bp = expert_plan(T, topk, b.n_pad, False, ex.cold.shapes)
            if _dbg.ENABLE:
                _dbg.probe_band(b, ex.layer_id, T)
            (src, gam, _, tiles), _ = route(x, self.gate.weight, self.gate.bias, b.counters, b.counts, topk,
                                            self.gate.route_scale, b.bounds, bp, None, None, is_padding)
            if _dbg.ENABLE:
                _dbg.sync(f"band route only, layer {ex.layer_id} T={T}")
                _dbg.check_route(ex.layer_id, T, src, gam, tiles, bp, b.W)
            b.send(x, src, gam, tiles, bp.tile)
            b.ev_go.record(side)
        _dbg.sync(f"band route + send, layer {ex.layer_id} T={T}")
        b.run(T)
        _dbg.sync(f"band run (RTX), layer {ex.layer_id} T={T}")
        grace = _gb.ENABLE and T >= _gb.MIN_T
        with torch.cuda.stream(side):
            if grace:      # the tail's job to the Grace cores; the GB300 skips the tail GEMMs
                gt = _gb.tail()
                src_t, gam_t, tiles_t, plan_t = ex.tail.route_only(x, self.gate, is_padding)
                _dbg.sync(f"tail route_only, layer {ex.layer_id} T={T}")
                gt.send_post(ex.layer_id, x, src_t, gam_t, tiles_t, plan_t.tile, 0)
                _dbg.sync(f"grace send_post, layer {ex.layer_id} T={T}")
            else:
                tail = ex.tail(x, self.gate, is_padding)
                _dbg.sync(f"GPU tail, layer {ex.layer_id} T={T}")
        hot = ex.hot(x, self.gate, is_padding)
        _dbg.sync(f"hot part, layer {ex.layer_id} T={T}")
        main.wait_stream(side)
        main.wait_event(b.ev_done)
        acc = self._count_acc if (self._count_acc is not None and T <= _COUNT_MAX) else None
        dh, sh, th = hot
        if grace:
            gt.wait()
            _dbg.sync(f"grace wait, layer {ex.layer_id} T={T}")
            none = torch.full_like(sh, -1)              # no layout rows in a second GPU part
            out = combine2(dh, sh, th, dh, none, th, topk, x.dtype, ex.hot.plan(T).tile, ex.H + ex.W, acc,
                           extra=b.y_g[:T], extra2=gt.partial(T))
            _dbg.sync(f"combine with the grace partial, layer {ex.layer_id} T={T}")
            _dbg.note(ex.layer_id, T, True, x, out, gt.partial(T))
            return out
        dt, st, tt = tail
        out = combine2(dh, sh, th, dt, st, tt, topk, x.dtype, ex.hot.plan(T).tile, ex.H + ex.W, acc,
                       extra=b.y_g[:T])
        _dbg.note(ex.layer_id, T, False, x, out)
        return out


def install():
    """replace Inkling2MoE in the decoder layers (the layer id comes from the constructing decoder layer)"""
    from vllm.models.inkling2 import model as _model
    orig_init = _model.Inkling2DecoderLayer.__init__

    def __init__(self, config, layer_id, quant_config, prefix):
        orig_moe = _model.Inkling2MoE
        _model.Inkling2MoE = lambda cfg, fmt: SplitMoE(cfg, fmt, layer_id)
        try:
            orig_init(self, config, layer_id, quant_config, prefix)
        finally:
            _model.Inkling2MoE = orig_moe
    _model.Inkling2DecoderLayer.__init__ = __init__
    from ink2 import attn, debug, follow, graceband
    attn.install()
    follow.install_runner_hook()
    graceband.install_runner_check()
    debug.install_profiler_steps()
    # INK2_HOOKS=mod1,mod2: extra measurement modules (each with install()), found in INK2_HOOK_PATH (logit dumps,
    # calibration captures)
    hooks = [h for h in os.environ.get("INK2_HOOKS", "").split(",") if h]
    if hooks:
        import importlib
        for d in reversed(os.environ.get("INK2_HOOK_PATH", "/ink2hooks").split(":")):
            sys.path.insert(0, d)
        for h in hooks:
            importlib.import_module(h).install()
            _LOG(f"hook {h} installed")
    _LOG(f"installed (hot experts per layer: {os.environ.get('INK2_HOT', '192')} unless placed)")
