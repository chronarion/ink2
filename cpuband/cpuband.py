"""ctypes binding of the Inkling 2 Grace-core band (cpuband.cpp), usable from inside the vLLM container process.

    import cpuband as cb
    lib = cb.load()                                    # builds build/libcpuband.so with g++ if it is missing or stale
    slot = cb.Slot.alloc()                             # tests; the engine passes its own pinned, GPU-mapped memory:
    slot = cb.Slot(ptr)                                #   cuMemHostAlloc(DEVICEMAP) or a cudaHostRegister'd buffer
    cb.add_layer(L, "nvfp4", w13, s13, w2, s2, amax13, amax2)   # lossless profile: the tail part's host tensors,
                                                                 # vLLM's GPU layout (MXFP8 the same, no amax)
    cb.add_layer_int4(L, *cb.load_hecate_int4(files, ids))     # fast profile: Hecate's GPTQ int4 files
    cb.start(slot, threads=56, core0=16)
    ... the GB300 posts jobs (slot.layer / rows / ids / gam / x, then ISSUE) and waits for DONE ...
    cb.stop()

Tensors are CPU tensors [n, rows, cols] (contiguous per expert; the band keeps raw pointers: keep them alive)."""
import ctypes
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cpuband.cpp")
SO = os.environ.get("CPUBAND_SO", os.path.join(HERE, "build", "libcpuband.so"))
FLAGS = ["-O3", "-mcpu=neoverse-v2", "-std=c++17", "-shared", "-fPIC"]
H, I, TOPK, MAXR = 4608, 2304, 8, 64
FMT = {"nvfp4": 0, "mxfp8": 1, "int4": 2}
_lib = None


def build(so=SO):
    os.makedirs(os.path.dirname(so), exist_ok=True)
    tmp = so + f".tmp{os.getpid()}"
    subprocess.check_call(["g++", *FLAGS, "-o", tmp, SRC, "-lpthread"])
    os.replace(tmp, so)


def load(so=SO, rebuild=None):
    global _lib
    if _lib is not None:
        return _lib
    if rebuild or not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(SRC):
        build(so)
    L = ctypes.CDLL(so)
    i64, i32, u32, f64 = ctypes.c_int64, ctypes.c_int, ctypes.c_uint32, ctypes.c_double
    L.cb_slot_layout.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    L.cb_add_layer.argtypes = [i32, i32, i32] + [i64] * 10 + [f64]
    L.cb_add_layer.restype = i32
    L.cb_add_layer_i4.argtypes = [i32, i32] + [i64] * 12
    L.cb_add_layer_i4.restype = i32
    L.cb_set_amax.argtypes = [i32, i32, f64, f64, f64]
    L.cb_set_amax.restype = i32
    L.cb_start.argtypes = [i64, i32, i32, i32]
    L.cb_start.restype = i32
    L.cb_stats.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    L.cb_post.argtypes = [i64, u32]
    L.cb_wait.argtypes = [i64, u32, i64]
    L.cb_wait.restype = i32
    L.cb_run.argtypes = [i64, u32, i64]
    L.cb_run.restype = i64
    L.cb_test_situ.argtypes = [i64, i64, i64, i32, i64]
    L.cb_last_act.argtypes = [i32, i32, i64]
    _lib = L
    return L


def layout():
    o = (ctypes.c_int64 * 10)()
    load().cb_slot_layout(o)
    keys = ("issue", "done", "err", "layer", "rows", "ids", "gam", "x", "out", "size")
    return dict(zip(keys, list(o)))


class Slot:
    """the job slot: views into pinned host memory at ptr (layout from cpuband.cpp's struct Slot)"""

    def __init__(self, ptr, keep=None):
        import torch
        self.ptr, self._keep, self.L = int(ptr), keep, layout()
        assert self.ptr % 256 == 0, "slot must be 256-byte aligned"
        L = self.L

        def view(off, n, dt):
            return torch.frombuffer((ctypes.c_byte * (n * torch.empty(0, dtype=dt).element_size())).from_address(self.ptr + off), dtype=dt)
        self.issue = view(L["issue"], 1, torch.int32)
        self.done = view(L["done"], 1, torch.int32)
        self.err = view(L["err"], 1, torch.int32)
        self.hdr = view(L["layer"], 2, torch.int32)              # layer, rows
        self.ids = view(L["ids"], MAXR * TOPK, torch.int32).view(MAXR, TOPK)
        self.gam = view(L["gam"], MAXR * TOPK, torch.float32).view(MAXR, TOPK)
        self.x = view(L["x"], MAXR * H, torch.bfloat16).view(MAXR, H)
        self.out = view(L["out"], MAXR * H, torch.float32).view(MAXR, H)
        self.seq = int(self.done[0])

    @classmethod
    def alloc(cls):
        import torch
        n = layout()["size"]
        buf = torch.zeros(n + 256, dtype=torch.uint8)
        p = buf.data_ptr()
        return cls((p + 255) // 256 * 256, keep=buf)

    def set_job(self, layer, x, ids, gam):
        """x [R, H] bf16, ids [R, 8] int (local tail id or -1), gam [R, 8] float"""
        R = x.shape[0]
        self.hdr[0], self.hdr[1] = int(layer), int(R)
        self.x[:R] = x
        self.ids[:R] = ids
        self.gam[:R] = gam

    def run(self, timeout_s=30.0):
        """post the job (release ISSUE = seq + 1) and wait for DONE; returns the round trip in ns"""
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        ns = load().cb_run(self.ptr, self.seq, int(timeout_s * 1e9))
        if ns < 0:
            raise RuntimeError(f"cpuband: job {self.seq} timed out")
        return ns


def _p(t):
    return 0 if t is None else t.data_ptr()


def _es(t):
    return t.stride(0) * t.element_size()


def add_layer(layer, fmt, w13, s13, w2, s2, amax13=None, amax2=None, denom=2688.0):
    """the layer's tail experts (local ids 0 .. n - 1) in vLLM's layout: w13 [n, 4608, 2304 | 4608] u8 / e4m3,
    s13 [n, 4608, 288 | 144] (swizzled), w2 [n, 4608, 1152 | 2304], s2 [n, 4608, 144 | 72]; NVFP4 amax [n] fp32"""
    f = FMT[fmt]
    n = w13.shape[0]
    for t in (w13, s13, w2, s2):
        assert t.device.type == "cpu" and t.shape[0] == n and t[0].is_contiguous()
    if f == 0:
        amax13 = amax13.float().contiguous().cpu(); amax2 = amax2.float().contiguous().cpu()
    r = load().cb_add_layer(layer, f, n, _p(w13), _es(w13), _p(s13), _es(s13), _p(w2), _es(w2), _p(s2), _es(s2),
                            _p(amax13), _p(amax2), float(denom))
    if r:
        raise RuntimeError("cpuband: add_layer failed (pool running or bad arguments)")


def start(slot, threads=56, core0=16, wait_mode=0):
    if load().cb_start(slot.ptr, threads, core0, wait_mode):
        raise RuntimeError("cpuband: start failed")


def stop():
    load().cb_stop()


def stats():
    o = (ctypes.c_int64 * 12)()
    load().cb_stats(o)
    k = ("jobs", "picks", "entries", "ns_total", "ns_x", "ns_a", "ns_c", "bad", "last_ns", "last_x", "last_a", "last_c")
    return dict(zip(k, list(o)))


# ---------------------------------------------------------------------------------------------------------------------
# fast profile: Hecate's GPTQ int4 (tools/quant/gptq_int4.py): uniform int4 q in -8..7 stored as q + 8, one fp32 scale
# per output row (no groups), act-order undone before packing (codes in natural column order), packed in the moe4
# fragment order; files layer{L}*.safetensors with "{expert}.{gate,up,down}.{q,s}". The band keeps row-major nibbles
# (low nibble = even column), converted at load exactly as Hecate's tier_split does for its Grace pool.

def pack_u4(codes, SPW=4):
    """gptq_int4.pack_u4 (verbatim): codes [N, K] uint8 0..15 -> moe4 fragment-order bytes [N * K / 2]"""
    N, K = codes.shape
    c = codes.view(N // (16 * SPW), SPW, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 8, 6, 2).contiguous().view(-1, 2)
    return (c[:, 0] | (c[:, 1] << 4)).contiguous()


def unpack_moe4(q, N, K):
    """tier_split._unpack_moe4 (verbatim): moe4 fragment-order int4 -> codes [N, K] uint8"""
    import torch
    c = torch.stack((q & 15, q >> 4), dim=-1).view(N // 64, K // 64, 4, 8, 4, 4, 2, 2, 2)
    return c.permute(0, 2, 8, 3, 1, 5, 7, 4, 6).reshape(N, K)


def rowmajor_i4(q, N, K):
    """tier_split._rowmajor_i4 (verbatim): fragment order -> row-major [N, K / 2], low nibble = even column"""
    c = unpack_moe4(q.view(-1), N, K)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()


def load_hecate_int4(paths, ids, hidden=H, inter=I):
    """experts `ids` (in local tail order) from Hecate-format files -> (gate_q, up_q, down_q, gate_s, up_s, down_s):
    row-major uint8 [n, I, H/2], [n, I, H/2], [n, H, I/2] and fp32 [n, I], [n, I], [n, H] (CPU tensors)"""
    import torch
    from safetensors import safe_open
    files = [safe_open(p, "pt", device="cpu") for p in ([paths] if isinstance(paths, str) else paths)]
    where = {k: f for f in files for k in f.keys()}
    get = lambda k: where[k].get_tensor(k)
    n = len(ids)
    gq = torch.empty(n, inter, hidden // 2, dtype=torch.uint8); uq = torch.empty_like(gq)
    dq = torch.empty(n, hidden, inter // 2, dtype=torch.uint8)
    gs = torch.empty(n, inter, dtype=torch.float32); us = torch.empty_like(gs); ds = torch.empty(n, hidden, dtype=torch.float32)
    for i, e in enumerate(ids):
        gq[i] = rowmajor_i4(get(f"{e}.gate.q"), inter, hidden); gs[i] = get(f"{e}.gate.s").float()
        uq[i] = rowmajor_i4(get(f"{e}.up.q"), inter, hidden); us[i] = get(f"{e}.up.s").float()
        dq[i] = rowmajor_i4(get(f"{e}.down.q"), hidden, inter); ds[i] = get(f"{e}.down.s").float()
    return gq, uq, dq, gs, us, ds


def add_layer_int4(layer, gate_q, up_q, down_q, gate_s, up_s, down_s):
    """the layer's tail (local ids 0 .. n - 1) as row-major int4 (load_hecate_int4); the band keeps raw pointers"""
    n = gate_q.shape[0]
    for t, shape in ((gate_q, (I, H // 2)), (up_q, (I, H // 2)), (down_q, (H, I // 2)), (gate_s, (I,)), (up_s, (I,)), (down_s, (H,))):
        assert t.device.type == "cpu" and t.shape[0] == n and tuple(t.shape[1:]) == shape and t[0].is_contiguous(), (t.shape, shape)
    assert gate_s.dtype == up_s.dtype == down_s.dtype == __import__("torch").float32
    r = load().cb_add_layer_i4(layer, n, _p(gate_q), _es(gate_q), _p(up_q), _es(up_q), _p(down_q), _es(down_q),
                               _p(gate_s), _es(gate_s), _p(up_s), _es(up_s), _p(down_s), _es(down_s))
    if r:
        raise RuntimeError("cpuband: add_layer_int4 failed (pool running or bad arguments)")
