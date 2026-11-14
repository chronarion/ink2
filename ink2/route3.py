"""One routing launch for an RTX-band call of up to 16 tokens (decode): vLLM's fused route kernel (ops/moe.py,
_fused_route_kernel) extended to place every token into BOTH GPU parts at once (the hot part in GPU memory and the tail in
Grace memory) and to write the RTX band's hand-off slot (the bf16 row, local band ids and routing weights). The router
GEMV and the top-k run once per layer instead of three times: three cooperative route launches per layer cost ~15 +
40 + 17 us at one row (torch-profiler trace, 2026-11-14). The layouts, gammas and quantized rows are those of separate
route() calls (the same helpers, the same top-k), so the experts' outputs are unchanged."""
import torch

from vllm.models.inkling2.ops.linear import _k_group_tile, _prefetch_k_group_tile
from vllm.models.inkling2.ops.moe import (FUSED_ROUTING_MAX_TOKENS, _ROUTERS_DONE, _TOKENS_DONE, _key_values,
                                          _load_bias, _local_experts, _log_sigmoid, _place_token, _quantize_x,
                                          _routing_weights, _row_scales, _spin_until, _store_x, _top_keys)
from vllm.triton_utils import tl, triton

MAX_TOKENS = FUSED_ROUTING_MAX_TOKENS


@triton.jit
def _route3_kernel(X, GateWeight, Bias, Partials, Padding, Counters,
                   BoundsA, MasksA, Src2DstA, GammasA, RowsA, TilesA, QValA, QScaA, QRowA, WAmaxA, WDenA,
                   BoundsC, MasksC, Src2DstC, GammasC, RowsC, TilesC, QValC, QScaC, QRowC, WAmaxC, WDenC,
                   XB, IdsB, GamB, band_lo, band_hi, T, route_scale,
                   H: tl.constexpr, E: tl.constexpr, E_POW2: tl.constexpr, GROUPS: tl.constexpr, TOPK: tl.constexpr,
                   NL_A: tl.constexpr, LP2_A: tl.constexpr, TILE_A: tl.constexpr, MC_A: tl.constexpr,
                   NV_A: tl.constexpr, NL_C: tl.constexpr, LP2_C: tl.constexpr, TILE_C: tl.constexpr,
                   MC_C: tl.constexpr, NV_C: tl.constexpr, HEAD: tl.constexpr, TAIL: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, XBLOCK: tl.constexpr):
    # programs [0, ROUTERS): the router K-group partials (as vLLM's kernel); program ROUTERS + t: token t
    pid = tl.program_id(0)
    N_TILES: tl.constexpr = (E + BN - 1) // BN
    ROUTERS: tl.constexpr = N_TILES * GROUPS
    if pid < ROUTERS:
        _prefetch_k_group_tile(GateWeight, E, H, BN, pid % N_TILES, pid // N_TILES)
    bias = _load_bias(Bias, E, E_POW2)
    lo_a = tl.load(BoundsA)
    hi_a = tl.load(BoundsA + 1)
    lo_c = tl.load(BoundsC)
    hi_c = tl.load(BoundsC + 1)
    tl.extra.cuda.gdc_wait()
    if pid < ROUTERS:
        _k_group_tile(X, GateWeight, Partials, T, E, H, BM, BN, True, pid % N_TILES, 0, pid // N_TILES)
        tl.debug_barrier()
        tl.atomic_add(Counters + _ROUTERS_DONE, 1, sem="release")
    else:
        token = pid - ROUTERS
        tokens = tl.full((1,), token, tl.int64)
        padding = None
        if Padding is not None:
            padding = tl.load(Padding + tokens)
        qa = _quantize_x(X, token.to(tl.int64), H, NV_A, HEAD, TAIL)
        qc = qa
        if NV_C != NV_A:
            qc = _quantize_x(X, token.to(tl.int64), H, NV_C, HEAD, TAIL)
        row = token.to(tl.int64) * H
        for c0 in tl.static_range(0, H, XBLOCK):                 # the band's bf16 copy of the row, meanwhile
            cols = c0 + tl.arange(0, XBLOCK)
            m = cols < H
            tl.store(XB + row + cols, tl.load(X + row + cols, m), m)
        _spin_until(Counters + _ROUTERS_DONE, ROUTERS)
        logits, keys, top = _top_keys(Partials, bias, tokens, T, E, E_POW2, GROUPS, TOPK)
        ids = E - (top & 0xFFFF).to(tl.int32)
        la = tl.reshape(_local_experts(ids, lo_a, hi_a, padding), (TOPK,))
        lc = tl.reshape(_local_experts(ids, lo_c, hi_c, padding), (TOPK,))
        lb = tl.reshape(_local_experts(ids, band_lo, band_hi, padding), (TOPK,))
        ea = tl.arange(0, LP2_A)
        ec = tl.arange(0, LP2_C)
        if T > 1:
            tl.atomic_or(MasksA + la, 1 << token, la >= 0, sem="relaxed")
            tl.atomic_or(MasksC + lc, 1 << token, lc >= 0, sem="relaxed")
            tl.debug_barrier()
            tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="release")
        weights = tl.reshape(_routing_weights(_log_sigmoid(_key_values(logits, keys, top)), route_scale, TOPK),
                             (TOPK,))
        slots = token * TOPK + tl.arange(0, TOPK)
        tl.store(IdsB + slots, lb)                                # the band slot: local band id (or -1), weight
        tl.store(GamB + slots, tl.where(lb >= 0, weights, 0.0))
        if T > 1:
            _spin_until(Counters + _TOKENS_DONE, T)
            masks_a = tl.load(MasksA + ea, ea < NL_A, other=0, cache_modifier=".cg")
            mine_a = tl.load(MasksA + la, la >= 0, other=0, cache_modifier=".cg")
            masks_c = tl.load(MasksC + ec, ec < NL_C, other=0, cache_modifier=".cg")
            mine_c = tl.load(MasksC + lc, lc >= 0, other=0, cache_modifier=".cg")
        else:
            masks_a = tl.max((ea[:, None] == la[None, :]).to(tl.int32), 1)
            mine_a = (la >= 0).to(tl.int32)
            masks_c = tl.max((ec[:, None] == lc[None, :]).to(tl.int32), 1)
            mine_c = (lc >= 0).to(tl.int32)
        dst_a = _place_token(masks_a, mine_a, la, weights, Src2DstA, GammasA, RowsA, TilesA, token, TOPK, TILE_A, MC_A)
        dst_c = _place_token(masks_c, mine_c, lc, weights, Src2DstC, GammasC, RowsC, TilesC, token, TOPK, TILE_C, MC_C)
        last = T == 1
        if T > 1:
            tl.debug_barrier()
            last = tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="relaxed") == 2 * T - 1
        _store_x(QValA, QScaA, QRowA, qa, dst_a, _row_scales(qa[4], WAmaxA, WDenA, la, NV_A), H, NV_A, HEAD, TAIL,
                 TILE_A, True)
        _store_x(QValC, QScaC, QRowC, qc, dst_c, _row_scales(qc[4], WAmaxC, WDenC, lc, NV_C), H, NV_C, HEAD, TAIL,
                 TILE_C, True)
        if last:
            if T > 1:
                tl.store(MasksA + ea, 0, ea < NL_A)
                tl.store(MasksC + ec, 0, ec < NL_C)
            tl.store(Counters + _ROUTERS_DONE, 0)
            tl.store(Counters + _TOKENS_DONE, 0)


def _outputs(tokens, topk, plan, nvfp4, hidden, device):
    """route()'s outputs for one part: (src2dst, gammas, routed, tiles), (quantized rows, block scales, row scales)"""
    n, rows = tokens * topk, plan.rows
    routing = (torch.empty(n, dtype=torch.int32, device=device), torch.empty(rows, dtype=torch.float32, device=device),
               torch.empty(n, dtype=torch.int32, device=device),
               torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device))
    if nvfp4:
        quantized = (torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device),
                     torch.empty((rows, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
                     torch.empty(rows, dtype=torch.float32, device=device))
    else:
        quantized = (torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device),
                     torch.empty((rows, hidden // 32), dtype=torch.uint8, device=device), None)
    return routing, quantized


def route3(x, gate, hot, tail, band, is_padding):
    """x [T, H] bf16, T <= MAX_TOKENS; hot / tail: split.py parts; band: band.py LayerBand (positions [band.H, band.H +
    band.W)). Writes the band slot and returns each GPU part's (routing, quantized, plan) for its compute()."""
    tokens, hidden = x.shape
    assert tokens <= MAX_TOKENS and x.is_contiguous() and gate.weight.is_contiguous()
    assert is_padding is None or (is_padding.shape == (tokens,) and is_padding.dtype == torch.bool)
    experts, topk = gate.weight.shape[0], gate.topk
    plan_a, plan_c = hot.plan(tokens), tail.plan(tokens)
    assert plan_a.tile >= tokens and plan_c.tile >= tokens
    ra, qa = _outputs(tokens, topk, plan_a, hot.nvfp4, hidden, x.device)
    rc, qc = _outputs(tokens, topk, plan_c, tail.nvfp4, hidden, x.device)
    head = 1 << (hidden.bit_length() - 1)
    groups, block_n = triton.cdiv(hidden, 1024), 32
    partials = torch.empty((groups, tokens, experts), dtype=torch.float32, device=x.device)
    routers = triton.cdiv(experts, block_n) * groups
    _route3_kernel[(routers + tokens,)](
        x, gate.weight, gate.bias, partials, is_padding, hot.counters,
        hot.expert_bounds, hot.counts, *ra, *qa, *hot.scales("w13"),
        tail.expert_bounds, tail.counts, *rc, *qc, *tail.scales("w13"),
        band.x_g, band.ids_g, band.gam_g, band.H, band.H + band.W, tokens, gate.route_scale,
        H=hidden, E=experts, E_POW2=triton.next_power_of_2(experts), GROUPS=groups, TOPK=topk,
        NL_A=hot.counts.numel(), LP2_A=triton.next_power_of_2(hot.counts.numel()), TILE_A=plan_a.tile,
        MC_A=plan_a.max_ctas, NV_A=hot.nvfp4, NL_C=tail.counts.numel(),
        LP2_C=triton.next_power_of_2(tail.counts.numel()), TILE_C=plan_c.tile, MC_C=plan_c.max_ctas,
        NV_C=tail.nvfp4, HEAD=head, TAIL=hidden - head, BM=16, BN=block_n, XBLOCK=512,
        num_warps=4, num_stages=4, enable_fp_fusion=False, launch_cooperative_grid=True, launch_pdl=True)
    return (ra, qa, plan_a), (rc, qc, plan_c)
