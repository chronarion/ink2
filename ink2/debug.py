"""INK2_DEBUG=1: eager-mode checks for the split MoE (no effect inside CUDA graph capture, off by default).

At each named stage of a MoE call, wait for every GPU (a failure names the stage it surfaced at), then check the route
state of every expert part of every layer: vLLM's route kernels take `counters` and `counts` zeroed and reset them, so
between kernels they must all read zero; a nonzero entry is the footprint of a stray write (or of a route kernel that
did not finish), and the first one is logged with the stage it appeared at. note() logs the first band calls of
layer 0 and the first non-finite MoE input, Grace partial or output."""
import os
import sys

import torch

ENABLE = os.environ.get("INK2_DEBUG", "0") == "1" or os.environ.get("INK2_GRACE_DEBUG", "0") == "1"
_LOG = lambda m: sys.stderr.write(f"INK2_DEBUG {m}\n")
PARTS = []                    # (layer, part name, buffer name, tensor)
_S = {"calls": 0, "bad": False, "dirty": 0, "flat": None}


def register(layer, name, part):
    if ENABLE:
        for buf in ("counts", "counters"):
            PARTS.append((layer, name, buf, getattr(part, buf)))
        _S["flat"] = None


def _active():
    return ENABLE and not torch.cuda.is_current_stream_capturing()


def sync(stage):
    if not _active():
        return
    for d in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(d)
        except Exception as e:
            _LOG(f"device {d} failed at or before stage '{stage}': {e}")
            raise
    if _S["dirty"] >= 3 or not PARTS:
        return
    ts = [t for *_, t in PARTS]
    if not bool(torch.stack([t.ne(0).any() for t in ts]).any()):
        return
    _S["dirty"] += 1
    for layer, name, buf, t in PARTS:
        nz = t.ne(0).nonzero().flatten()
        if nz.numel():
            _LOG(f"nonzero route state after stage '{stage}': layer {layer} {name}.{buf} (size {t.numel()}): "
                 f"entries {nz[:8].tolist()} = {t[nz[:8]].tolist()}")
            if _S["dirty"] == 1:
                break


def note(layer, T, grace, x, out=None, partial=None):
    if not _active():
        return
    if layer == 0 and _S["calls"] < 40:
        _S["calls"] += 1
        _LOG(f"band call T={T} grace={grace}")
    if _S["bad"]:
        return
    for name, t in (("input", x), ("grace partial", partial), ("output", out)):
        if t is not None and not bool(torch.isfinite(t).all()):
            _S["bad"] = True
            bad = (~torch.isfinite(t.float())).nonzero()[:4].tolist()
            _LOG(f"non-finite {name} at layer {layer} T={T} grace={grace}: first at {bad}")
            return


def probe_band(b, layer, T):
    """read the band's pinned slot through its GB300-side views (an invalid mapping faults here, named)"""
    if not _active():
        return
    sync(f"before the band slot probe, layer {layer} T={T}")
    for name in ("x_g", "ids_g", "gam_g", "y_g"):
        v = getattr(b, name)
        float(v.float().sum())
        sync(f"band slot probe {name} (ptr {v.data_ptr():#x}, host {getattr(b, name[:-1] + 'h').data_ptr():#x}), "
             f"layer {layer} T={T}")


def check_route(layer, T, src, gam, tiles, plan, W):
    """the band route's output on the host: layout rows within plan.rows, tile experts within the warm band"""
    if not _active():
        return
    s, t = src.cpu(), tiles.cpu()
    rows, tile, m = plan.rows, plan.tile, plan.max_ctas
    used = s[s >= 0]
    bad = []
    if used.numel() and int(used.max()) >= rows:
        bad.append(f"src max {int(used.max())} >= rows {rows}")
    if used.numel():
        te = t[(used // tile).long()]
        if int(te.min()) < 0 or int(te.max()) >= W:
            bad.append(f"tile experts {te.tolist()[:16]} outside [0, {W})")
    if bad:
        _LOG(f"band route layer {layer} T={T}: rows {rows} tile {tile} max_ctas {m} tiles size {t.numel()} gam size "
             f"{gam.numel()}; src {s.tolist()[:16]}; tiles {t.tolist()[:24]}" + (f"; BAD: {bad}" if bad else ""))
