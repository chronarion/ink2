"""One combine over two parts' expert rows (GPU memory + Grace): the image's _combine_kernel with a second source.
Each token's TOPK slots are summed in fp32 in slot (selection) order, taking a slot's row from whichever part holds that
expert, and rounded once, so the output is independent of the placement. With a counter (Acc, int64 [E] by position)
the kernel also counts the rows routed to each expert: the expert of a layout row is the expert of its tile
(tiles[row // TILE]); part A holds positions [0, NA), part B [NA, E). One program per token counts (column block 0)."""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _combine2_kernel(DownA, SrcA, TilesA, DownB, SrcB, TilesB, Out, Acc, Extra,
                     H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr,
                     TILE: tl.constexpr, NA: tl.constexpr, COUNT: tl.constexpr, EXTRA: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(TOPK):
        ra = tl.load(SrcA + token * TOPK + slot).to(tl.int64)
        rb = tl.load(SrcB + token * TOPK + slot).to(tl.int64)
        in_a = ra >= 0
        in_b = (ra < 0) & (rb >= 0)
        a = tl.load(DownA + ra * H + cols, in_a, other=0).to(tl.float32)
        b = tl.load(DownB + rb * H + cols, in_b, other=0).to(tl.float32)
        acc += a + b
        if COUNT:
            # one program per token counts (column block 0); positions: part A holds [0, NA), part B [NA, E)
            first = block == 0
            ea = tl.load(TilesA + ra // TILE, in_a, other=0).to(tl.int64)
            eb = tl.load(TilesB + rb // TILE, in_b, other=0).to(tl.int64)
            tl.atomic_add(Acc + ea, 1, mask=first & in_a, sem="relaxed")
            tl.atomic_add(Acc + NA + eb, 1, mask=first & in_b, sem="relaxed")
    if EXTRA:   # another tier's fp32 partial of this token (the RTX band), before the single rounding
        acc += tl.load(Extra + token * H + cols)
    tl.store(Out + token * H + cols, acc)


def combine2(down_a, src_a, tiles_a, down_b, src_b, tiles_b, topk, out_dtype, tile, n_a, acc=None, extra=None):
    """acc: int64 [E] counter row of this layer (positions), or None; extra: fp32 [T, H] partial to add, or None"""
    tokens = src_a.numel() // topk
    hidden = down_a.shape[1]
    out = torch.empty((tokens, hidden), dtype=out_dtype, device=down_a.device)
    block = 512
    assert hidden % block == 0 and down_b.shape[1] == hidden
    assert acc is None or (acc.dtype == torch.int64 and acc.is_contiguous())
    _combine2_kernel[(tokens, hidden // block)](down_a, src_a, tiles_a, down_b, src_b, tiles_b, out, acc, extra,
                                                hidden, topk, block, tile, n_a, acc is not None, extra is not None,
                                                num_warps=2, launch_pdl=True)
    return out
