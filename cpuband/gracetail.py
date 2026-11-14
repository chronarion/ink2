"""The Grace band's GPU side (experimental, off by default; ink2/graceband.py): jobs for the Grace cores posted from
inside vLLM's CUDA graphs through one pinned slot, following Hecate's release / wait pattern and ink2/band.py's send
kernel. Tested on the GB300: GPU-posted jobs (eager, 1-64 rows, and replayed CUDA graphs) equal host-posted ones bit for
bit, a timed-out wait sets the error word; the cores' arithmetic (cpuband.cpp) is tested against float64 references.

Per decode-sized MoE call (T <= 64 tokens), per layer, all inside the captured graph:

  side stream (forked from main as today's tail part):
    route(x, bounds = [H + W, 384))            the tail part's own route (as band.py does for the warm band), or the
                                               fused tier route of track A1; gives src2dst / gammas / tiles
    cb_send_k  <<<T, 256>>>                    row t -> slot.x[t] (bf16), its tail picks -> slot.ids[t][k] (local tail
                                               id of the layout row's tile, -1 elsewhere), gammas -> slot.gam[t][k]
    cb_post_k  <<<1, 1>>>                      seq = ++dev_seq (device memory: every replay advances it); slot.layer,
                                               slot.rows = T; fence.sys; st.release.sys slot.issue = seq
  ... the hot part (GB300 HBM) and the RTX band run meanwhile; the Grace pool computes the tail ...
  main stream (after joining the side stream):
    cb_wait_k  <<<1, 1>>>   or the prologue of combine2:
                                               ld.acquire.sys slot.done until done - seq >= 0 (wrap-safe) or
                                               %globaltimer passes the timeout: then slot.err |= 0x100, *dev_skip = 1
    combine2(..., extra = RTX partial, extra2 = slot.out device view [T, 4608], unless *dev_skip)

The host checks slot.err after every step (vLLM runner hook, e.g. where ink2/follow.py hooks execute_model) and stops the
server on a nonzero value: a timed-out wait means the step's output lacked the tail experts (never silent).

Ordering: the send kernel's stores to pinned host memory (system scope) complete with the kernel; cb_post_k's
__threadfence_system() + st.release.sys orders them before ISSUE for the CPU (Hecate's rtx_graph release_k pattern,
in service since 1.6). The CPU's acquire of ISSUE, its stores of out[], and its release of DONE (after the pool's gather
barrier) pair with the GPU's ld.acquire.sys of DONE. One slot is enough: layer L + 1 posts only after layer L's combine
(its input), so a job never overlaps the previous one's read of out[].

CUDA graphs: T (rows) and the layer index are constants of each captured graph (vLLM captures per batch size); the
sequence number lives in device memory (dev_seq), so replays post fresh jobs, and the CPU pool detects a job by ISSUE
changing (any value). Graph capture itself executes nothing; the eager warmup runs do post jobs, so the pool must be
running before the first forward (start it in the model loader, after the tail weights are in host memory).
"""
import os

import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

// the slot as laid out by cpuband.cpp (offsets from cb_slot_layout(); asserted at load)
struct Slot { unsigned issue; char p0[124]; unsigned done; char p1[124]; unsigned err; char p2[124];
              int layer; int rows; int pad[30]; int ids[64][8]; float gam[64][8]; char p3[0]; };

// one CTA per token: its bf16 row, its tail picks and routing weights (src: layout row of each (token, slot) in the tail
// part or -1; tiles: expert of each layout tile; gam: gamma per layout row)
// (first: the part's expert index of local tail id 0; experts below it belong to another tier -> -1)
__global__ void cb_send_k(const __nv_bfloat16* __restrict__ x, const int* __restrict__ src, const float* __restrict__ gam,
                          const int* __restrict__ tiles, int tile, int first, unsigned char* slot, long x_off, int H)
{
    const int t = blockIdx.x;
    Slot* S = (Slot*) slot;
    if (threadIdx.x < 8)
    {
        const int r = src[t * 8 + threadIdx.x];
        const int id = r >= 0 ? tiles[r / tile] - first : -1;
        S->ids[t][threadIdx.x] = id >= 0 ? id : -1;
        S->gam[t][threadIdx.x] = id >= 0 ? gam[r] : 0.f;
    }
    const uint4* xs = (const uint4*) (x + (long) t * H);
    uint4* xd = (uint4*) (slot + x_off + (long) t * H * 2);
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x) xd[i] = xs[i];
}

// one thread: header, then ISSUE = ++seq with release semantics at system scope
__global__ void cb_post_k(unsigned char* slot, unsigned* dev_seq, int layer, int rows)
{
    Slot* S = (Slot*) slot;
    const unsigned s = *dev_seq + 1;
    *dev_seq = s;
    S->layer = layer; S->rows = rows;
    __threadfence_system();
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(&S->issue), "r"(s) : "memory");
}

// one thread: wait for DONE == seq (wrap-safe) with a %globaltimer timeout; on timeout set the error word and skip
__global__ void cb_wait_k(unsigned char* slot, const unsigned* dev_seq, unsigned* dev_skip, unsigned long long timeout_ns)
{
    Slot* S = (Slot*) slot;
    const unsigned want = *dev_seq;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;)
    {
        unsigned v;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(&S->done) : "memory");
        if ((int) (v - want) >= 0) { *dev_skip = 0; return; }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 > timeout_ns) { atomicOr(&S->err, 0x100u); *dev_skip = 1; __threadfence_system(); return; }
        __nanosleep(64);
    }
}

void send(torch::Tensor x, torch::Tensor src, torch::Tensor gam, torch::Tensor tiles, int64_t tile, int64_t first, int64_t slot_dev, int64_t x_off)
{
    const int T = x.size(0), H = x.size(1);
    cb_send_k<<<T, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>((const __nv_bfloat16*) x.data_ptr(), src.data_ptr<int>(),
        gam.data_ptr<float>(), tiles.data_ptr<int>(), (int) tile, (int) first, (unsigned char*) slot_dev, x_off, H);
}
void post(int64_t slot_dev, torch::Tensor dev_seq, int64_t layer, int64_t rows)
{
    cb_post_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>((unsigned char*) slot_dev, (unsigned*) dev_seq.data_ptr(), (int) layer, (int) rows);
}
void wait(int64_t slot_dev, torch::Tensor dev_seq, torch::Tensor dev_skip, int64_t timeout_ns)
{
    cb_wait_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>((unsigned char*) slot_dev, (const unsigned*) dev_seq.data_ptr(),
        (unsigned*) dev_skip.data_ptr(), (unsigned long long) timeout_ns);
}
"""

_ext = None


def ext(build_dir):
    global _ext
    if _ext is None:
        os.makedirs(build_dir, exist_ok=True)       # load_inline writes its sources there and does not create it
        _ext = load_inline(name="cpuband_gpu", cpp_sources="void send(torch::Tensor x, torch::Tensor src, torch::Tensor gam, "
                           "torch::Tensor tiles, int64_t tile, int64_t first, int64_t slot_dev, int64_t x_off);\n"
                           "void post(int64_t slot_dev, torch::Tensor dev_seq, int64_t layer, int64_t rows);\n"
                           "void wait(int64_t slot_dev, torch::Tensor dev_seq, torch::Tensor dev_skip, int64_t timeout_ns);",
                           cuda_sources=_SRC, functions=["send", "post", "wait"], build_directory=build_dir,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"])
    return _ext


class GraceTail:
    """per process: one slot, the pool, the device counters. Per layer: register its tail with add_layer.

        tail = GraceTail(threads=56, core0=16)                  # model load, before any forward
        tail.add_layer(L, part_cold, first=W)                   # lossless profile: the cold _Part's host tensors,
                                                                # experts [W, n) = tail (NVFP4 / MXFP8 as published)
        tail.add_layer_int4(L, files, perm[H + W:])             # fast profile: Hecate-format GPTQ int4 files
        tail.start()
        ... in SplitMoE.forward, T <= 64:
        with torch.cuda.stream(side):
            routing = route(x, ..., bounds=tail_bounds, ...)     # src2dst, gammas, rows, tiles of the tail part
            tail.send_post(L, x, src2dst, gammas, tiles, plan.tile, first=0)   # (W with the cold part's own route)
        ... hot part, RTX band ...
        main.wait_stream(side); tail.wait(); out = combine2(..., extra2=tail.partial(T), skip=tail.dev_skip)
    """

    def __init__(self, threads=56, core0=16, wait_mode=0, timeout_ms=20, build_dir="/root/.cache/vllm/cpuband"):
        import cpuband as cb
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        self.cb, self.threads, self.core0, self.wait_mode = cb, threads, core0, wait_mode
        L = cb.layout()
        assert (L["issue"], L["done"], L["err"], L["layer"], L["rows"], L["ids"], L["gam"]) == (0, 128, 256, 384, 388, 512, 2560), L
        self.host = torch.zeros(L["size"], dtype=torch.uint8, device="cpu").pin_memory()   # page-aligned, portable
        self.slot = cb.Slot(self.host.data_ptr(), keep=self.host)
        self.dev = get_accelerator_view_from_cpu_tensor(self.host)                       # GB300-side view
        self.x_off, self.out_off = L["x"], L["out"]
        self.dev_seq = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.dev_skip = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.timeout_ns = int(timeout_ms * 1e6)
        self.k = ext(build_dir)
        self._keep = []

    def add_layer(self, layer, cold, first):
        """cold: the layer's host _Part (split.py: _host tensors [n, rows, cols] in vLLM's layout, amax on the GPU);
        experts [first, n) of it are the tail, local ids 0 .. n - first - 1 (= tiles' expert index - first)"""
        h = cold._host
        t = [h[k][first:] for k in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")]
        a13 = a2 = None
        if cold.nvfp4:
            a13 = cold.w13_weight_amax[first:].float().cpu(); a2 = cold.w2_weight_amax[first:].float().cpu()
            denom = float(cold.w13_weight_outer_scale_denominator.item())
        else:
            denom = 1.0
        self._keep += [t, a13, a2]
        self.cb.add_layer(layer, "nvfp4" if cold.nvfp4 else "mxfp8", *t, a13, a2, denom)

    def add_layer_int4(self, layer, files, tail_ids):
        """fast profile: the layer's tail as Hecate-format GPTQ int4 (gptq_int4.py's files, "{e}.{gate,up,down}.{q,s}"),
        tail_ids = checkpoint expert ids in local tail order (placement perm[H + W:]); converted to row-major nibbles
        (Hecate's _rowmajor_i4) in pageable host memory, 16 MB per expert; the jobs and the slot are unchanged"""
        t = self.cb.load_hecate_int4(files, tail_ids)
        self._keep.append(t)
        self.cb.add_layer_int4(layer, *t)

    def start(self):
        self.cb.start(self.slot, self.threads, self.core0, self.wait_mode)

    def send_post(self, layer, x, src, gam, tiles, tile, first):
        """src / gam / tiles of a route over a part whose expert index `first` is local tail id 0 (0 for a route with
        the tail's own bounds [H + W, 384); W for the cold part's route, whose [0, W) are the RTX band's experts)"""
        self.k.send(x, src, gam, tiles, tile, first, self.dev.data_ptr(), self.x_off)
        self.k.post(self.dev.data_ptr(), self.dev_seq, layer, x.shape[0])

    def wait(self):
        self.k.wait(self.dev.data_ptr(), self.dev_seq, self.dev_skip, self.timeout_ns)

    def partial(self, T):
        """the fp32 partial [T, 4608] as a GB300-side view of the slot's out[]"""
        return self.dev[self.out_off:self.out_off + T * 4608 * 4].view(torch.float32).view(T, 4608)

    def check(self):
        """host side, after each step: nonzero = a job failed (bad header / id: CPU bits 1 / 2; wait timed out: 0x100)"""
        return int(self.slot.err[0])
