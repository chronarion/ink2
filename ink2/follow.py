"""Hot experts that follow the text (decode), with guarantees.

A fixed placement ranks experts on a general corpus; one conversation routes more narrowly. Every 8 steps the routing
counts (filled by the combine kernel, combine2.py, at no extra launch) update two decayed rates per expert (fast: 128
counted steps, medium: 1,024), with the corpus counts behind the placement as 128 steps of pseudo-counts. A Grace
expert replaces a GPU-memory one only if, with margins in standard errors, the rows it would serve from GPU memory over
a horizon (512 / 4,096 steps) exceed twice the swap's own cost in expert reads. The corpus-top H - 128 experts of each
layer are never evicted; promoted / evicted experts stay put for 128 / 256 steps; at most 4 swaps per tick. Every swap
is scored from the counters afterwards (ledger); swaps that do not pay raise the margin (governor). A log line every 256
ticks compares the rows read outside GPU memory with what the boot placement would have read on the same rows.

A swap exchanges two experts' weights (three block copies per tensor on the forward's own stream: Grace -> HBM scratch,
HBM -> Grace, scratch -> HBM) and the two gate rows / bias entries, so the routing stays the same function of the input
and every tensor keeps its address (no graph is invalidated); with the single combine the outputs are bit-identical
whatever the placement. A swap that fails half way stops the server.
  INK2_FOLLOW (default 1), INK2_FOLLOW_PRIOR (routing counts json; default: the placement's corpus counts),
  INK2_FOLLOW_PERIOD 8, _SWAPS 4, _TAU_F 128, _TAU_M 1024, _Z 2.0, _KAPPA 128, _TH 512, _TM 4096, _MARGIN 2, _PMAX 128,
  _GRACE 128, _GRACE_OUT 256, _COST_NV 5.8, _COST_MX 4.1 (reads per swap), _SELFTEST=1 (log the swap cost at arm time),
  _STORM n (test only: n random swaps per layer before every step)"""
import json
import math
import os
import sys

import numpy as np
import torch

_E = os.environ.get
# off with an int4 Grace tail (INK2_GRACE=int4): the cores hold their own int4 copy of each tail expert, which a hot <-> tail
# swap of the native weights would not follow
ENABLE = _E("INK2_FOLLOW", "1") == "1" and _E("INK2_GRACE", "") != "int4"
PERIOD = int(_E("INK2_FOLLOW_PERIOD", "8"))
SWAPS = int(_E("INK2_FOLLOW_SWAPS", "4"))
TAU_F = float(_E("INK2_FOLLOW_TAU_F", "128"))
TAU_M = float(_E("INK2_FOLLOW_TAU_M", "1024"))
Z = float(_E("INK2_FOLLOW_Z", "2.0"))
KAPPA = float(_E("INK2_FOLLOW_KAPPA", "128"))
TH = float(_E("INK2_FOLLOW_TH", "512"))
TM = float(_E("INK2_FOLLOW_TM", "4096"))
MARGIN = float(_E("INK2_FOLLOW_MARGIN", "2"))
PMAX = int(_E("INK2_FOLLOW_PMAX", "128"))
GRACE = int(_E("INK2_FOLLOW_GRACE", "128"))
GRACE_OUT = int(_E("INK2_FOLLOW_GRACE_OUT", "256"))
COST = (float(_E("INK2_FOLLOW_COST_NV", "5.8")), float(_E("INK2_FOLLOW_COST_MX", "4.1")))
STORM = int(_E("INK2_FOLLOW_STORM", "0"))
SELFTEST = _E("INK2_FOLLOW_SELFTEST", "0") == "1"
PRIOR = _E("INK2_FOLLOW_PRIOR", "")
COUNT_MAX = int(_E("INK2_COUNT_MAX", "64"))
LOG_TICKS = int(_E("INK2_FOLLOW_LOG", "256"))
_LOG = lambda m: sys.stderr.write(f"INK2_FOLLOW {m}\n")
_WEIGHTS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


class Follow:
    def __init__(self):
        self.moes = {}
        self.acc = None                 # [L, E] int64 on the GPU, rows per position (filled by the combine kernel)
        self.armed = False
        self.dead = False
        self.step = 0                   # every execute_model call with tokens
        self.counted = 0                # calls of <= COUNT_MAX tokens since the last tick (the ones the counters see)
        self.pending = []               # counted steps of the periods whose snapshots are still in flight

    # ---- registration (unchanged interface)
    def counter(self, layer, num_layers, E, device):
        if self.acc is None:
            self.acc = torch.zeros((num_layers, E), dtype=torch.int64, device=device)
        return self.acc[layer]

    def register(self, layer, moe):
        self.moes[layer] = moe

    # ---- arming: first real request
    def arm(self):
        L, E = self.acc.shape
        layers = sorted(self.moes)
        assert layers == list(range(L))
        ex0 = [self.moes[l].experts for l in layers]
        self.H = np.array([ex.H for ex in ex0])
        # experts in an RTX band (ink2/band.py, positions [H, H + W)) never move: candidates come from the tail only
        self.Wb = np.array([getattr(ex, "W", 0) for ex in ex0])
        self.nbytes = np.array([sum(getattr(ex.hot, n)[0].numel() * getattr(ex.hot, n).element_size() for n in _WEIGHTS)
                                if ex.H else 0 for ex in ex0], dtype=np.float64)
        self.cost = np.array([COST[0] if ex.nvfp4 else COST[1] for ex in ex0])
        self.rank0 = np.tile(np.arange(E), (L, 1))          # boot rank of the expert at each position
        self.pos0 = np.tile(np.arange(E), (L, 1))           # position of each boot rank
        self.f = np.zeros((L, E)); self.m = np.zeros((L, E))
        self.ef = 0.0; self.em = 0.0                        # counted steps behind the two rates
        self.born = np.full((L, E), -10 ** 9, dtype=np.int64); self.gone = self.born.copy()
        self.prior = np.zeros((L, E))                       # rows per decode row, by position
        if PRIOR:
            c = json.load(open(PRIOR))
            for l in layers:
                row = np.array(c[str(l)], dtype=np.float64)
                self.prior[l] = (row / max(row.sum() / 8.0, 1.0))[ex0[l].perm.numpy()]
        else:
            _LOG("no INK2_FOLLOW_PRIOR: evidence only (KAPPA has no effect)")
        self.prev = None
        self.host = torch.zeros((L, E), dtype=torch.int64, pin_memory=True)
        self.event = torch.cuda.Event()
        self.margin = MARGIN; self.ema = 0.0; self.closed = 0; self.paid = 0; self.net = 0.0; self.since = 0
        self.ledger = []                                    # [layer, rank_in, rank_out, step, base_in, base_out]
        self.swaps = 0; self.ticks = 0
        self.stat = np.zeros(4)                             # rows, rows outside now, rows outside under the boot placement, steps
        self.rng = np.random.default_rng(0)
        # scratch for one expert of every tensor shape, allocated now: a swap must not be able to fail half way
        self.tmp = {}
        for l in layers:
            hot = ex0[l].hot
            if not hot.n:
                continue
            for n in _WEIGHTS:
                t = getattr(hot, n)[0]
                self.tmp.setdefault((t.dtype, tuple(t.shape)), torch.empty_like(t))
        self.acc.zero_()
        self.stream = torch.cuda.current_stream()
        self.armed = True
        if SELFTEST:
            t = self.selftest()
            _LOG("selftest: stream time per swap " + ", ".join(f"layer {k} {v * 1e6:.0f} us" for k, v in t.items()))
        _LOG(f"armed: period {PERIOD}, {SWAPS} swaps per tick, tau {TAU_F:.0f}/{TAU_M:.0f}, z {Z}, kappa {KAPPA:.0f}, "
             f"horizons {TH:.0f}/{TM:.0f}, margin {MARGIN}, pool {PMAX}, cost {COST} reads")

    # ---- the swap: stream-ordered on the forward's stream, between forwards; no host->device copies
    def _swap(self, layer, s, c):
        moe = self.moes[layer]; ex = moe.experts
        hot, cold, H = ex.hot, ex.cold, ex.H
        j = c - H
        try:
            for name in _WEIGHTS:
                h = getattr(hot, name)[s]
                g_host = cold._host[name][j]
                tmp = self.tmp[(h.dtype, tuple(h.shape))]
                tmp.copy_(g_host, non_blocking=True)        # Grace -> scratch (copy engine, registered memory)
                g_host.copy_(h, non_blocking=True)          # GPU memory -> Grace
                h.copy_(tmp, non_blocking=True)             # scratch -> GPU memory
            if ex.nvfp4:
                for name in ("w13_weight_amax", "w2_weight_amax"):
                    a, b = getattr(hot, name), getattr(cold, name)
                    t = a[s].clone(); a[s].copy_(b[j]); b[j].copy_(t)
            for t in (moe.gate.weight.data, moe.gate.bias.data, moe._count_acc):
                x = t[s].clone(); t[s].copy_(t[c]); t[c].copy_(x)       # python-int indexing: kernels only, no sync
        except BaseException:
            self.dead = True                                # weights and gate may disagree now: never serve from this
            raise
        p = ex.perm
        p[s], p[c] = p[c].item(), p[s].item()
        for a in (self.f, self.m, self.prior, self.rank0, self.born, self.gone, self.prev):
            if a is not None:
                a[layer, [s, c]] = a[layer, [c, s]]
        self.pos0[layer, self.rank0[layer, s]] = s; self.pos0[layer, self.rank0[layer, c]] = c
        self.born[layer, s] = self.step; self.gone[layer, c] = self.step
        self.swaps += 1

    # ---- per step
    def tick(self, num_tokens):
        if self.dead:
            raise RuntimeError("INK2_FOLLOW: a swap failed half way; restart the server")
        if not self.moes:
            return
        if not self.armed:
            self.arm()
        assert torch.cuda.current_stream() == self.stream, "INK2_FOLLOW: execute_model runs on another stream"
        self.step += 1
        self.counted += num_tokens <= COUNT_MAX
        if STORM:
            self._storm()
        if self.step % PERIOD:
            return
        if self.prev is None:                               # first tick: baseline
            self.prev = self.acc.cpu().numpy().copy()       # one sync, once
        elif self.pending and self.event.query():
            n = self.pending.pop(0)
            for layer, s, c in self._decide(self.host.numpy().copy(), n):
                self._open(layer, s, c)
                self._swap(layer, s, c)
        elif self.pending:
            return                                          # snapshot still in flight: this period joins the next one
        self.host.copy_(self.acc, non_blocking=True)        # after this tick's swaps: same positions as the counters
        self.event.record()
        self.pending.append(self.counted); self.counted = 0
        self.ticks += 1

    def _open(self, layer, s, c):
        # baselines are filled from the next snapshot (taken after the swap); positions follow the experts
        self.ledger.append([layer, int(self.rank0[layer, c]), int(self.rank0[layer, s]), self.step, None, None])

    def _decide(self, snap, n):
        L, E = snap.shape
        delta = np.maximum(snap - self.prev, 0).astype(np.float64)
        self.prev = snap
        self._ledger(snap)
        rows = delta.sum()
        if n <= 0 or rows <= 0:
            return []                                       # nothing counted (prefill, mixed batches): no decay, no swaps
        df, dm = math.exp(-n / TAU_F), math.exp(-n / TAU_M)
        self.f = self.f * df + delta; self.m = self.m * dm + delta
        self.ef = self.ef * df + n; self.em = self.em * dm + n
        streams = rows / (L * 8.0 * n)                      # decode rows per counted step
        g = self.prior * streams                            # corpus rate per step, by position
        pm = self.m + KAPPA * g
        wf, wm = self.ef, self.em + KAPPA
        keep = np.minimum((pm + Z * np.sqrt(pm + 1.0)) / wm, (self.f + Z * np.sqrt(self.f + 1.0)) / wf)
        lf = np.maximum(self.f - Z * np.sqrt(self.f), 0.0) / wf
        lm = np.maximum(pm - Z * np.sqrt(pm), 0.0) / wm
        # shadow comparison on the same rows: outside GPU memory now / under the boot placement
        cold_now = sum(delta[i, self.H[i]:].sum() * self.nbytes[i] for i in range(L))
        cold_boot = sum(delta[i][self.rank0[i] >= self.H[i]].sum() * self.nbytes[i] for i in range(L))
        self.stat += (sum(delta[i].sum() * self.nbytes[i] for i in range(L)), cold_now, cold_boot, n)
        cands = []
        now = self.step
        for i in range(L):
            H = int(self.H[i])
            Hc = H + int(self.Wb[i])
            if H == 0 or Hc >= E:
                continue
            kv = keep[i, :H].copy()
            kv[self.rank0[i, :H] < H - PMAX] = np.inf       # protected core: the boot-top H - PMAX experts
            kv[now - self.born[i, :H] < GRACE] = np.inf
            val = np.maximum(lf[i, Hc:] * TH, lm[i, Hc:] * TM)
            val[now - self.gone[i, Hc:] < GRACE_OUT] = -1.0
            k = min(4, H, E - Hc)
            hot = np.argpartition(kv, k - 1)[:k]; hot = hot[np.argsort(kv[hot])]
            cold = np.argpartition(-val, k - 1)[:k]; cold = cold[np.argsort(-val[cold])]
            for s, cj in zip(hot, cold):
                c = Hc + int(cj)
                if not np.isfinite(kv[s]) or val[cj] <= 0.0:
                    break
                gain = max((lf[i, c] - kv[s]) * TH, (lm[i, c] - kv[s]) * TM) - self.margin * self.cost[i]
                if gain > 0.0:
                    cands.append((gain * self.nbytes[i], i, int(s), c))
        cands.sort(reverse=True)
        if self.ticks % LOG_TICKS == 0 and self.stat[3] > 0:
            r, cn, cb, st = self.stat
            _LOG(f"tick {self.ticks}: rows outside GPU memory {cn / r:.3f} (boot placement on the same rows {cb / r:.3f}), "
                 f"{cn / st / 1e9:.2f} vs {cb / st / 1e9:.2f} GB per step (row bound); swaps {self.swaps}, ledger "
                 f"{self.paid}/{self.closed} paid, mean {self.net / max(1, self.closed):+.1f} reads, margin {self.margin:.1f}")
            self.stat[:] = 0
        return [(i, s, c) for _, i, s, c in cands[:SWAPS]]

    def _ledger(self, snap):
        """score the swaps from the counters: rows the promoted expert got since, minus rows the evicted one got"""
        keep = []
        for e in self.ledger:
            layer, rin, rout, t0, bin_, bout = e
            vin, vout = snap[layer, self.pos0[layer, rin]], snap[layer, self.pos0[layer, rout]]
            if bin_ is None:
                e[4], e[5] = vin, vout; keep.append(e); continue
            moved = self.pos0[layer, rin] >= self.H[layer] or self.pos0[layer, rout] < self.H[layer]
            if self.step - t0 < TH and not moved:
                keep.append(e); continue
            pay = float(vin - bin_) - float(vout - bout) - self.cost[layer]
            self.closed += 1; self.paid += pay > 0; self.net += pay
            self.ema = 0.9 * self.ema + 0.1 * pay; self.since += 1
        self.ledger = keep
        if self.since >= 16:                                # governor: swaps must pay, or they get rarer
            self.since = 0
            if self.ema < 0.0:
                self.margin = min(self.margin * 1.5, 64.0)
            elif self.ema > 2.0 * COST[0]:
                self.margin = max(self.margin / 1.25, MARGIN)

    def _storm(self):
        """test only: STORM random swaps per layer before every step (every forward runs right after swaps)"""
        if self.prev is None:
            return
        for layer in sorted(self.moes):
            H = int(self.H[layer]); E = self.acc.shape[1]
            for _ in range(STORM):
                self._swap(layer, int(self.rng.integers(0, H)), int(self.rng.integers(H, E)))

    def selftest(self, n=200):
        """seconds of stream time per swap, per layer kind: call once on an idle server, set COST_* from it
        (COST = t_swap / (expert bytes / C2C rate))"""
        out = {}
        for layer in (0, 47):
            torch.cuda.synchronize(); ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            H = int(self.H[layer]); ev0.record()
            for i in range(n):
                a, b = i % H, H + i % (self.acc.shape[1] - H)
                self._swap(layer, a, b); self._swap(layer, a, b)        # there and back: the placement is unchanged
            ev1.record(); ev1.synchronize()
            out[layer] = ev0.elapsed_time(ev1) / (2 * n) * 1e-3
        return out


FOLLOW = Follow() if ENABLE else None


def install_runner_hook():
    if FOLLOW is None:
        return
    from vllm.v1.worker.gpu import model_runner as mr
    cls = mr.GPUModelRunner
    orig = cls.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        n = scheduler_output.total_num_scheduled_tokens
        # boot warm-up requests are real execute_model calls (v1/worker/gpu/warmup.py:139, 643, 705); their ids
        # (_warmup_*, _context_warmup_*, _v2_mixed_warmup_*, _dummy_req_*) all start with an underscore
        real = n and not kwargs.get("dummy_run", False) and not any(
            r.startswith("_") for r in scheduler_output.num_scheduled_tokens)
        if real:
            FOLLOW.tick(n)                                  # exceptions propagate: a broken promotion must stop the server
        return orig(self, scheduler_output, *args, **kwargs)
    cls.execute_model = execute_model
