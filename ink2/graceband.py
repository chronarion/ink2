"""The Grace-core band: the tail experts (after hot and warm) computed on the Grace CPU's cores in place, so the GB300 does
not stream them over NVLink-C2C (as Hecate's cold band). The host pool and its NEON kernels are cpuband.cpp
(~/llm/inkling2/cpuband, mounted at /ink2cpuband); the GPU side posts a job per decode-sized MoE call through one pinned
slot inside vLLM's CUDA graphs and the combine adds the cores' fp32 partial (cpuband/gracetail.py).

  INK2_GRACE=native (the tail in its published NVFP4 / MXFP8 bytes, read where they lie) | int4 (GPTQ int4 tail files,
      INK2_GRACE_INT4=dir with layer{L}.safetensors from quant/gptq_tail.py) | unset (the tail is read in place by the GPU)
  INK2_GRACE_MIN_T (4): calls of fewer rows keep the in-place read (the cores cannot hide a pick behind a short window)
  INK2_GRACE_THREADS (56), INK2_GRACE_CORE0 (16): the pool's cores; vLLM's own threads stay on cores 0 .. CORE0 - 1
  INK2_GRACE_TIMEOUT_MS (1000): a job that takes longer sets the slot's error word and the next step stops the server
      (a hang guard only: Hecate's waits have no timeout; 20 ms stopped a C16 run on 11-14)"""
import os
import sys

import torch

MODE = os.environ.get("INK2_GRACE", "")
ENABLE = MODE in ("native", "int4")
MIN_T = int(os.environ.get("INK2_GRACE_MIN_T", "4"))
THREADS = int(os.environ.get("INK2_GRACE_THREADS", "56"))
CORE0 = int(os.environ.get("INK2_GRACE_CORE0", "16"))
TIMEOUT_MS = float(os.environ.get("INK2_GRACE_TIMEOUT_MS", "1000"))
_LOG = lambda m: sys.stderr.write(f"INK2_GRACE {m}\n")
_STATE = {"tail": None, "layers": [], "started": False}


def register(layer, experts):
    """called by SplitExperts at construction; the layer joins the pool at the first forward (weights loaded by then)"""
    if ENABLE and experts.tail is not None:
        _STATE["layers"].append((layer, experts))


def tail():
    """the GraceTail, with every registered layer added and the pool started (once, at the first forward)"""
    if _STATE["started"]:
        return _STATE["tail"]
    sys.path.insert(0, os.environ.get("INK2_CPUBAND_HOME", "/ink2cpuband"))
    os.environ.setdefault("CPUBAND_SO", "/root/.cache/vllm/cpuband/libcpuband.so")
    from gracetail import GraceTail
    dev = _STATE["layers"][0][1].hot.w13_weight.device      # the GB300 (the RTX may be the band's device too)
    with torch.cuda.device(dev):
        t = GraceTail(threads=THREADS, core0=CORE0, timeout_ms=TIMEOUT_MS, build_dir="/root/.cache/vllm/cpuband_gpu")
    _LOG(f"slot view on {t.dev.device}, counters on {t.dev_seq.device}, current device {torch.cuda.current_device()}")
    for layer, ex in _STATE["layers"]:
        if MODE == "int4":
            path = os.path.join(os.environ["INK2_GRACE_INT4"], f"layer{layer}.safetensors")
            t.add_layer_int4(layer, [path], ex.perm[ex.H + ex.W:].tolist())
        else:
            t.add_layer(layer, ex.cold, first=ex.W)
    t.start()
    _STATE.update(tail=t, started=True)
    _LOG(f"{MODE} tail of {len(_STATE['layers'])} layers on {THREADS} cores from {CORE0}, calls of >= {MIN_T} rows")
    return t


def pin_engine_threads():
    """keep this process's threads (vLLM's) on cores 0 .. CORE0 - 1: a busy thread on a pool core stalls its barriers"""
    if ENABLE:
        try:
            os.sched_setaffinity(0, set(range(CORE0)))
        except OSError as e:
            _LOG(f"could not pin to cores 0-{CORE0 - 1}: {e}")


def install_runner_check():
    """the pool starts right after the model is loaded (its setup allocates pinned memory and reads the tail's scales:
    neither may happen inside a CUDA graph capture); after every step, a nonzero slot error word (bad job, or a wait
    that timed out) stops the server"""
    if not ENABLE:
        return
    from vllm.v1.worker.gpu import model_runner as mr
    cls = mr.GPUModelRunner
    orig_load = cls.load_model

    def load_model(self, *args, **kwargs):
        out = orig_load(self, *args, **kwargs)
        if _STATE["layers"]:
            tail()
        return out
    cls.load_model = load_model
    orig = cls.execute_model

    def execute_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        t = _STATE["tail"]
        if t is not None:
            err = t.check()
            if err:
                what = [m for bit, m in ((1, "bad job header"), (2, "tail expert id out of range"),
                                         (0x100, f"a GPU wait passed {TIMEOUT_MS:g} ms")) if err & bit]
                try:
                    st = t.cb.stats()
                    hdr = t.slot.hdr.tolist()
                except Exception as e:          # the report must not hide the error
                    st, hdr = repr(e), None
                msg = f"error {err:#x} ({', '.join(what) or 'unknown bits'}); slot layer/rows {hdr}; pool stats {st}"
                _LOG(msg)
                try:                            # kept beside vLLM's cache (a host mount): the container is often gone
                    import time
                    with open("/root/.cache/vllm/ink2-grace-errors.log", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
                except OSError:
                    pass
                raise RuntimeError(f"INK2_GRACE: the Grace band reported error {err:#x} "
                                   f"({', '.join(what) or 'unknown bits'}); restart the server")
        return out
    cls.execute_model = execute_model

