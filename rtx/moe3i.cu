// moe3i: the RTX PRO 6000 (sm_120) band of EXL3 / TR3 trellis experts for Inkling 2 Small's MoE layers (decode shape).
// Derived from Hecate's moe3.cu (runtime/tier/moe3/moe3.cu), adapted to Inkling's H 4608 / I 2304.
//
// Changes against moe3:
//   * Inkling's activation: SiTU with gate / up softcaps and the routing weight (gamma) inside the activation, as vLLM's
//     ops/activation.py (_situ): gate and up are rounded to bf16 (the GEMM's output type there), then
//     bf16(((softcap(g, 4) * sigmoid(g)) * softcap(u, 25)) * gamma) with the same PTX approximations (ex2 / rcp / tanh
//     .approx), and that bf16 value goes into the down projection (no FP4 requant). The fin kernel then sums the
//     experts' down outputs without a weight.
//   * Mixed bit widths (TR3v4): every (expert, projection) tensor has its own K in {3, 4, 5} (16 K int16 per 16x16
//     tile, ExLlamaV3's layout and mul1 codebook). One launch handles any mix; the stream-K split is balanced by bytes
//     (units x K), and the ring slot is sized for the band's largest K (KMAX 3: 3 stages, KMAX 4 / 5: 2 stages, so two
//     blocks still fit an SM).
//   * Layer-shared residual-side vectors (TR3v4): with gate_up_suh shared by every expert of the layer, each row is
//     rotated once (not once per pick) and gate and up read the same rotated row; with down_svh shared, the experts'
//     raw down sums of a row meet in one accumulator and the output transform (H128, svh) runs once per row (it is
//     linear, so sum-then-transform = transform-then-sum). Per-expert vectors (moe3's behaviour) stay available; the two
//     sides are selected independently. No vector is assumed to be +-1 (scales folded in are fine).
//   * Up to NBMAX = 256 band experts per layer (moe3: 128) and up to MAXR = 64 rows.
//   * Row gather: every kernel works on picks (row, band expert), not on (expert, every row). A per-block plan groups
//     the picks of the call by band expert (rows in order); the stream-K kernels decode an expert's weights once per
//     16 picks ("virtual expert") and feed its picks as the n side of the MMA (two n8 fragments). rot / act / fin
//     touch only the picks (or, with shared vectors, the rows).
//   * Deterministic sums: the rotated inputs' per-chunk partials are summed in a fixed order; the stream-K partials of a
//     GEMV (and, with a shared down_svh, the experts' down sums of a row) meet in 64-bit integer atomics on fixed-point
//     values (x 2^32, exact for |partial| >= 2^-8, range +-2^31; raw sums here are ~1e5-1e6): integer addition does not
//     depend on the order, so the same input gives the same output bit for bit, and the total is the exact sum of the
//     fp32 partials, rounded once when act / fin read it. -DMOE3I_ATOMIC: moe3's fp32 atomics (order not fixed).
//   * Row count from the host (graph per row count) or from a device word (one graph for every row count <= Rmax).
// Intermediate-channel reorderings baked into an expert's weights (gate / up output order = down input order) need
// nothing here: the activation is elementwise and the expert's private vectors are stored in the same order.
//
// Chain (PDL): rot -> A (gate + up) -> act -> B (down) -> fin. Scratch self-cleans (gu by act, ds by fin).
// Build options: -DMOE3I_FPP=8 (one MMA fragment per weight pass: ~3% faster with one pick per expert, but > 8 picks of
// an expert stream its weights again); -DMOE3I_CENTER (codebook h - 1536 in the MMA: half the fp32 accumulation error,
// ~3-5% slower). Diagnostics only: -DMOE3I_EXACT_ACT (float64 tanh / sigmoid), -DMOE3I_NO_ROUND (no bf16 roundings).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace {

constexpr int H = 4608, I = 2304, TOPK = 8, MAXR = 64, MAXP = MAXR * TOPK, NBMAX = 256;
constexpr int WPB = 8, NT = WPB * 32;
#ifndef MOE3I_FPP
#define MOE3I_FPP 16
#endif
constexpr int FPP = MOE3I_FPP, FR = FPP / 8;           // picks per weight pass (FR n8 MMA fragments)
constexpr int MAXV = NBMAX + MAXP / FPP;               // virtual experts (expert, 16-pick group) per call
constexpr int HC = H / 128, IC = I / 128;              // 128-chunks of H and I
static_assert(NT == NBMAX, "one plan thread per band expert");
// the stream-K sums: 64-bit fixed point (deterministic, default) or fp32 (-DMOE3I_ATOMIC)
#ifdef MOE3I_ATOMIC
typedef float acc_t;
__device__ __forceinline__ void acc_add(float* p, float v) { atomicAdd(p, v); }
__device__ __forceinline__ float4 acc_take4(float* p)          // read 4 sums and clear them
{
    const float4 v = __ldcg((const float4*) p);
    *(float4*) p = make_float4(0.f, 0.f, 0.f, 0.f);
    return v;
}
#else
typedef unsigned long long acc_t;
__device__ __forceinline__ void acc_add(acc_t* p, float v) { atomicAdd(p, (unsigned long long) __float2ll_rn(v * 4294967296.0f)); }
__device__ __forceinline__ float4 acc_take4(acc_t* p)          // read 4 sums and clear them
{
    const ulonglong2 a = __ldcg((const ulonglong2*) p), b = __ldcg((const ulonglong2*) (p + 2));
    *(ulonglong2*) p = make_ulonglong2(0ull, 0ull);
    *(ulonglong2*) (p + 2) = make_ulonglong2(0ull, 0ull);
    constexpr float S = 2.3283064365386963e-10f;                // 2^-32 (exact)
    return make_float4(__ll2float_rn((long long) a.x) * S, __ll2float_rn((long long) a.y) * S,
                       __ll2float_rn((long long) b.x) * S, __ll2float_rn((long long) b.y) * S);
}
#endif
constexpr int KLO = 3, KHI = 5;                        // trellis bit widths handled
__host__ __device__ constexpr int tile_words(int K) { return 8 * K; }          // uint32 per 16x16 tile (16 K int16)
__host__ __device__ constexpr int unit_words(int K) { return 16 * tile_words(K); }   // a unit: 4 k-tiles x 4 n-tiles
__host__ __device__ constexpr int ring_stages(int kmax) { return kmax <= 3 ? 3 : 2; } // two blocks per SM (100 KB)
constexpr int UPM = (I / 64) * (H / 64);               // units per matrix (gate, up: 36 x 72; down: 72 x 36)
static_assert(UPM == (H / 64) * (I / 64), "gate / up and down have the same unit count");
constexpr float R_SCALE = 0.088388347648f;             // 1 / sqrt(128)
constexpr float GATE_CAP = 4.0f, UP_CAP = 25.0f;       // config mlp_gate_softcap / mlp_up_softcap

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int shift)
{
    const uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}
// two mul1 codebook entries without the affine: fp16 patterns 0x6400 + byte sum of (idx * 0x83DCD12D). The codebook
// does not depend on K (K only sets how many bits the trellis advances per weight)
__device__ __forceinline__ uint32_t dec2(uint32_t i0, uint32_t i1)
{
    const uint32_t s0 = __dp4a(i0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(i1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
#ifdef MOE3I_CENTER
    // centered codebook: h - 1536 (exact in fp16), so the MMA accumulates sum((h - 1536) x) instead of a sum dominated by
    // 1536 * sum(x); the bias term absorbs the offset (k_bias() below)
    const uint32_t hv = __byte_perm(s0, s1, 0x5410);
    const half2 c = __hsub2(*(const half2*) &hv, __halves2half2(__ushort_as_half(0x6600), __ushort_as_half(0x6600)));
    return *(const uint32_t*) &c;
#else
    return __byte_perm(s0, s1, 0x5410);
#endif
}

// A lane's 8 weights of a 16x16 tile (tile positions lane * 8 .. + 7, ExLlamaV3's tensor-core order) are 16-bit
// windows of the tile's circular K-bit trellis stream ending at bits (lane * 8 + 257 + j) * K, j = 0..7 (exl3_dq.cuh
// dq8). They are exactly an m16n8k16 A fragment of the transposed tile. Windows 4..7 span 16 + 3K <= 31 bits, so one
// funnel shift yields all four; for K <= 4 all eight windows lie in one pair of words (moe3's 3-bit decode), for K = 5
// the low four take a second pair.
struct LaneDec { int ia, ib, s, ia2, ib2, s2; };
template <int K>
__device__ __forceinline__ LaneDec lane_dec(int lane)
{
    constexpr int TW = tile_words(K);
    const int b1 = (lane * 8 + 257) * K, b2 = b1 + 7 * K;     // end of window 0 / of window 7
    LaneDec d;
    d.ib = ((b2 - 1) / 32) % TW;
    d.s = ((b2 - 1) / 32 + 1) * 32 - b2;
    if constexpr (K <= 4)
    {
        d.ia = ((b1 - 16) / 32) % TW;
        d.ia2 = d.ib2 = 0; d.s2 = d.s + 4 * K;
    }
    else
    {
        d.ia = ((b2 - 16 - 3 * K) / 32) % TW;                  // first word of window 4
        const int e = b2 - 4 * K;                               // end of window 3
        d.ia2 = ((b1 - 16) / 32) % TW;
        d.ib2 = ((e - 1) / 32) % TW;
        d.s2 = ((e - 1) / 32 + 1) * 32 - e;
    }
    return d;
}
template <int K>
__device__ __forceinline__ void dq8_a(const uint32_t* tw, const LaneDec& d, uint32_t* af)
{
    const uint32_t a = tw[d.ia], b = tw[d.ib];
    const uint32_t w7 = fshift(b, a, d.s), w6 = w7 >> K, w5 = w6 >> K, w4 = w5 >> K;
    uint32_t w3;
    if constexpr (K <= 4) w3 = fshift(b, a, d.s2);
    else w3 = fshift(tw[d.ib2], tw[d.ia2], d.s2);
    const uint32_t w2 = w3 >> K, w1 = w2 >> K, w0 = w1 >> K;
    af[0] = dec2(w0 & 0xffff, w1 & 0xffff);            // f0[0]: n g,     k 2c, 2c+1
    af[2] = dec2(w2 & 0xffff, w3 & 0xffff);            // f0[1]: n g,     k 2c+8, 2c+9
    af[1] = dec2(w4 & 0xffff, w5 & 0xffff);            // f1[0]: n g + 8, k 2c, 2c+1
    af[3] = dec2(w6 & 0xffff, w7 & 0xffff);            // f1[1]: n g + 8, k 2c+8, 2c+9
}

__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
    float h0 = (v0 + v1) + (v2 + v3), h1 = (v0 - v1) + (v2 - v3), h2 = (v0 + v1) - (v2 + v3), h3 = (v0 - v1) - (v2 - v3);
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, i), p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, i), p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        const bool hi = lane & i;
        h0 = (hi ? -h0 : h0) + p0; h1 = (hi ? -h1 : h1) + p1; h2 = (hi ? -h2 : h2) + p2; h3 = (hi ? -h3 : h3) + p3;
    }
    v0 = h0 * R_SCALE; v1 = h1 * R_SCALE; v2 = h2 * R_SCALE; v3 = h3 * R_SCALE;
}
__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int i = 16; i; i >>= 1) v += __shfl_xor_sync(0xffffffffu, v, i);
    return v;
}
__device__ __forceinline__ float k_inv() { return __half2float(__ushort_as_half((unsigned short) 0x1eee)); }
#ifdef MOE3I_CENTER
__device__ __forceinline__ float k_bias() { return __half2float(__ushort_as_half((unsigned short) 0xc931)) + 1536.0f * k_inv(); }
#else
__device__ __forceinline__ float k_bias() { return __half2float(__ushort_as_half((unsigned short) 0xc931)); }
#endif
__device__ __forceinline__ float4 ld_h4(const half* p)            // 4 fp16 -> fp32
{
    const uint2 v = *(const uint2*) p;
    const float2 a = __half22float2(*(const half2*) &v.x), b = __half22float2(*(const half2*) &v.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

// ---- Inkling's activation (vllm/models/inkling2/ops/activation.py, _situ / _softcap / _sigmoid_approx) -------------
#ifdef MOE3I_NO_ROUND      // diagnostic build: no bf16 rounding of gate / up / activation
__device__ __forceinline__ float bf16r(float v) { return v; }
#else
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
#endif
__device__ __forceinline__ float sigmoid_approx(float x)
{
    float r;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float tanh_approx(float x)
{
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
#ifdef MOE3I_EXACT_ACT    // diagnostic build: exact tanh / sigmoid (float64) instead of vLLM's .approx PTX
#define sigmoid_approx(x) ((float) (1.0 / (1.0 + exp(-(double) (x)))))
#define tanh_approx(x) ((float) tanh((double) (x)))
#endif
__device__ __forceinline__ float softcap4(float x) { return __fmul_rn(GATE_CAP, tanh_approx(__fmul_rn(x, 0.25f))); }
__device__ __forceinline__ float softcap25(float x)
{
    float q = __fmul_rn(x, 0.04f);                                          // _div_rn(x, 25): exact for every fp32 x
    const float res = fmaxf(__fmaf_rn(q, 25.0f, -x), -3.4028234663852886e38f);
    q = __fmaf_rn(-res, 0.04f, q);
    return __fmul_rn(UP_CAP, tanh_approx(q));
}
// g, u: the fp32 gate / up outputs; returns the fp32 value of the bf16 activation fed to the down projection
__device__ __forceinline__ float situ(float g, float u, float gamma)
{
    g = bf16r(g); u = bf16r(u);                                             // the GEMM's bf16 output
    const float s = sigmoid_approx(g);
    return bf16r(__fmul_rn(__fmul_rn(__fmul_rn(softcap4(g), s), softcap25(u)), gamma));
}

struct Tabs
{
    const int64_t* tr[3];    // per local id: trellis of gate, up, down
    const int64_t* su[3];    // per local id: input-side vectors (gate / up: unused when gsu is set)
    const int64_t* sv[3];    // per local id: output-side vectors (down: unused when dsv is set)
    const int8_t* kb;        // [3][nb] bit width per (projection, local id)
    const half* gsu;         // layer-shared gate / up input vector [H] (TR3v4 gate_up_suh), or null
    const half* dsv;         // layer-shared down output vector [H] (TR3v4 down_svh), or null
};
struct Scr
{
    half* xr;        // rotated gate / up inputs: [MAXP][2][H] per pick and projection, or [MAXR][H] per row (shared)
    float* xsp;      // their per-chunk sums: [MAXP][2][HC] or [MAXR][HC]
    acc_t* gu;       // [MAXP][2][I] raw gate / up sums (zero between calls)
    half* ar;        // [MAXP][I] rotated down input per pick
    float* asp;      // [MAXP][IC] its per-chunk sums
    acc_t* ds;       // raw down sums (zero between calls): [MAXP][H] per pick, or [MAXR][H] per row (shared)
};

__device__ __forceinline__ int nrows(const int* rows_dev, int rmax)
{
    if (!rows_dev) return rmax;
    const int r = *(volatile const int*) rows_dev;
    return r < 0 ? 0 : (r > rmax ? rmax : r);
}

// ---- the plan: picks of the call grouped by band expert --------------------------------------------------------------
// pick i = row * 8 + slot, valid when 0 <= ids[i] < nb. Experts in increasing local id, the picks of one expert in row
// order (deterministic, so every block sees the same groups); an expert with c picks is ceil(c / 16) virtual experts.
// MODE 1 / 2 also prices the stream phase (A: gate + up, B: down): a unit of a K-bit tensor costs K, and vc[v] is the
// cost at which virtual expert v starts, so the stream-K split can hand every warp the same number of bytes.
struct Plan
{
    int nv, np, cost;
    unsigned long long rows[NBMAX];    // rows that picked local expert e
    short offs[NBMAX];
    short ve[MAXV], voff[MAXV];
    unsigned char vcnt[MAXV];
    unsigned char vk[MAXV][2];         // bit widths of the phase's matrices (A: gate, up; B: down)
    int vc[MAXV];
    short plist[MAXP];
    int wsum[WPB], wcost[WPB];
};
template <int MODE>
__device__ void make_plan(const int* __restrict__ ids, int R, int nb, const int8_t* __restrict__ kb, Plan& pl)
{
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, P = R * TOPK;
    int k0 = 0, k1 = 0;                                     // this thread's expert's bit widths, loaded first (latency)
#ifdef MOE3I_X_NOKB      // experiment: constant K 3 (no table loads)
    if (MODE == 1) { k0 = 3; k1 = 3; }
    if (MODE == 2) k0 = 3;
#else
    if (MODE == 1 && t < nb) { k0 = kb[t]; k1 = kb[nb + t]; }
    if (MODE == 2 && t < nb) k0 = kb[2 * nb + t];
#endif
    pl.rows[t] = 0ull;
    __syncthreads();
    for (int i = t; i < P; i += NT)
    {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) nb) atomicOr(&pl.rows[e], 1ull << (i / TOPK));
    }
    __syncthreads();
    const int c = __popcll(pl.rows[t]), nvt = (c + FPP - 1) / FPP, pk = c | (nvt << 16);
    const int vcost = c ? UPM * (k0 + k1) : 0;
    const int ec = nvt * vcost;
    int inc = pk, incc = ec;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        const int y = __shfl_up_sync(0xffffffffu, inc, d);
        if (lane >= d) inc += y;
        if constexpr (MODE != 0) { const int yc = __shfl_up_sync(0xffffffffu, incc, d); if (lane >= d) incc += yc; }
    }
    if (lane == 31) { pl.wsum[w] = inc; if constexpr (MODE != 0) pl.wcost[w] = incc; }
    __syncthreads();
    int pre = 0, prec = 0;
    for (int q = 0; q < w; ++q) { pre += pl.wsum[q]; if constexpr (MODE != 0) prec += pl.wcost[q]; }
    const int ex = pre + inc - pk, off = ex & 0xffff, vb = ex >> 16, exc = prec + incc - ec;
    pl.offs[t] = (short) off;
    for (int f = 0; f < nvt; ++f)
    {
        pl.ve[vb + f] = (short) t; pl.voff[vb + f] = (short) (off + f * FPP);
        pl.vcnt[vb + f] = (unsigned char) min(FPP, c - f * FPP);
        if constexpr (MODE != 0)
        {
            pl.vk[vb + f][0] = (unsigned char) k0; pl.vk[vb + f][1] = (unsigned char) k1;
            pl.vc[vb + f] = exc + f * vcost;
        }
    }
    if (t == NT - 1) { const int tot = pre + inc; pl.np = tot & 0xffff; pl.nv = tot >> 16; pl.cost = prec + incc; }
    __syncthreads();
    for (int i = t; i < P; i += NT)
    {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) nb)
        {
            const int r = i / TOPK;
            pl.plist[pl.offs[e] + __popcll(pl.rows[e] & ((1ull << r) - 1ull))] = (short) i;
        }
    }
    __syncthreads();
}

// first unit whose start cost is >= c (units of a K-bit matrix cost K each; matrices in order (v, p))
template <int MATS>
__device__ __forceinline__ int unit_at(const Plan& pl, int c)
{
    if (c >= pl.cost) return pl.nv * MATS * UPM;
    int lo = 0, hi = pl.nv - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (pl.vc[mid] <= c) lo = mid; else hi = mid - 1; }
    int off = c - pl.vc[lo];
    #pragma unroll
    for (int p = 0; p < MATS; ++p)
    {
        const int K = pl.vk[lo][p];
        if (off < UPM * K) return (lo * MATS + p) * UPM + (off + K - 1) / K;   // == next matrix's start if past the end
        off -= UPM * K;
    }
    return (lo + 1) * MATS * UPM;
}

// ---------------------------------------------------------------------------------------------------------------------
// rot: xr = fp16(H128(x * suh)) and the chunks' sums. Shared gate / up vector: warp job = (row with a band pick,
// 128-chunk), once for both projections. Per-expert vectors: warp job = (pick, projection, 128-chunk).
__global__ void __launch_bounds__(NT) moe3i_rot_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const int* rows_dev, int rmax, int nb, Tabs tb, Scr sc)
{
    __shared__ Plan pl;
    pdl_wait();                                            // x / ids written, the previous call's fin done
    pdl_launch();
    const int R = nrows(rows_dev, rmax), lane = threadIdx.x & 31;
    auto rotate = [&](const __nv_bfloat16* xrow, const half* s, int col, half* dst, float* sum)
    {
        const uint2 xv = *(const uint2*) (xrow + col);
        const float2 x01 = __bfloat1622float2(*(const __nv_bfloat162*) &xv.x), x23 = __bfloat1622float2(*(const __nv_bfloat162*) &xv.y);
        const float4 sv = ld_h4(s + col);
        float v0 = x01.x * sv.x, v1 = x01.y * sv.y, v2 = x23.x * sv.z, v3 = x23.y * sv.w;
        had128(v0, v1, v2, v3, lane);
        const half2 o01 = __floats2half2_rn(v0, v1), o23 = __floats2half2_rn(v2, v3);
        uint2 ov; ov.x = *(const uint32_t*) &o01; ov.y = *(const uint32_t*) &o23;
        *(uint2*) (dst + col) = ov;
        const float2 f01 = __half22float2(o01), f23 = __half22float2(o23);
        const float t = warp_sum((f01.x + f01.y) + (f23.x + f23.y));
        if (lane == 0) *sum = t;
    };
    if (tb.gsu)
    {
        for (int q = blockIdx.x * WPB + (threadIdx.x >> 5); q < R * HC; q += gridDim.x * WPB)
        {
            const int r = q / HC, c = q % HC;
            bool any = false;
            #pragma unroll
            for (int k = 0; k < TOPK; ++k) any |= (unsigned) ids[r * TOPK + k] < (unsigned) nb;
            if (!any) continue;
            rotate(x + (size_t) r * H, tb.gsu, c * 128 + lane * 4, sc.xr + (size_t) r * H, sc.xsp + (size_t) r * HC + c);
        }
        return;
    }
    make_plan<0>(ids, R, nb, tb.kb, pl);
    const int jobs = pl.np * 2 * HC;
    for (int q = blockIdx.x * WPB + (threadIdx.x >> 5); q < jobs; q += gridDim.x * WPB)
    {
        const int i = pl.plist[q / (2 * HC)], p = (q / HC) & 1, c = q % HC, e = ids[i], r = i / TOPK;
        const half* suh = (const half*) (p ? tb.su[1][e] : tb.su[0][e]);
        rotate(x + (size_t) r * H, suh, c * 128 + lane * 4, sc.xr + ((size_t) i * 2 + p) * H, sc.xsp + ((size_t) i * 2 + p) * HC + c);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// stream-K GEMV over units (virtual expert v, matrix p, group gr of 4 n-tiles = 64 output channels, k-block kb of 4
// k-tiles = 64 inputs). The weights are the MMA's A operand (16 output channels per tile), the <= 16 picks of v its n side.
template <int MATS, int N, int K>
struct Phase
{
    static constexpr int G = N / 64, KBN = K / 64, UPE = MATS * UPM, NTN = N / 16;
    static_assert(G * KBN == UPM, "unit count");
};

// flush of a warp's raw sums into dst(pick, p) + col: lane holds (channel gr*64 + t*16 + g (+8), slots 2c, 2c+1) per fragment
template <typename Dst>
__device__ __forceinline__ void red_flush(const Plan& pl, int v, int p, int gr, float (*c)[4][4], Dst dst)
{
    const int lane = threadIdx.x & 31, g = lane >> 2, s0 = (lane & 3) * 2, cnt = pl.vcnt[v], vo = pl.voff[v];
    #pragma unroll
    for (int f = 0; f < FR; ++f)
    {
        if (f * 8 >= cnt) break;
        const int a = f * 8 + s0, b = a + 1;
        acc_t* da = a < cnt ? dst(pl.plist[vo + a], p) : nullptr;
        acc_t* db = b < cnt ? dst(pl.plist[vo + b], p) : nullptr;
        #pragma unroll
        for (int t = 0; t < 4; ++t)
        {
            const int col = gr * 64 + t * 16 + g;
            if (da) { acc_add(da + col, c[f][t][0]); acc_add(da + col + 8, c[f][t][2]); }
            if (db) { acc_add(db + col, c[f][t][1]); acc_add(db + col + 8, c[f][t][3]); }
        }
    }
}

// copy unit u (K-bit matrix, 16 tiles of 8K words: 4K x 16 B, K chunks per lane) into a ring slot
template <int K, int NTN>
__device__ __forceinline__ void issue_unit(const uint32_t* base, int gr, int kb, uint32_t* dst, int lane)
{
    constexpr int TW = tile_words(K);
    #pragma unroll
    for (int c3 = 0; c3 < K; ++c3)
    {
        const int c = lane + 32 * c3, q = c / TW, off = c % TW;        // TW chunks of 16 B per k-tile row of 4 n-tiles
        cp16(su32(dst + q * 4 * TW + off * 4), base + ((size_t) (kb * 4 + q) * NTN + gr * 4) * TW + off * 4);
    }
}

template <int MATS, int N, int K_, int STAGES, int KMAX, typename XPtr, typename Dst>
__device__ __forceinline__ void stream3(const Plan& pl, const int64_t* const* trp, XPtr xptr, Dst dst, uint32_t* ring)
{
    using P = Phase<MATS, N, K_>;
    constexpr int SLOT = unit_words(KMAX);
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * WPB + (threadIdx.x >> 5), nw = gridDim.x * WPB;
#ifdef MOE3I_UNIFORM_SPLIT     // experiment: split by unit count (moe3), not by bytes
    const int U = pl.nv * P::UPE, per = (U + nw - 1) / nw, u0 = min(U, gw * per), n = min(U, u0 + per) - u0;
#else
    // every warp the same cost quota, a multiple of KMAX (so a K-uniform band splits into whole equal unit counts, as
    // moe3's split by units); the last warps may get less or nothing
    const int q = (((pl.cost + nw - 1) / nw + KMAX - 1) / KMAX) * KMAX;
    const int c0 = min(pl.cost, gw * q), c1 = min(pl.cost, (gw + 1) * q);
    const int u0 = unit_at<MATS>(pl, c0), n = unit_at<MATS>(pl, c1) - u0;
#endif
    auto issue = [&](int i)
    {
        if (i < n)
        {
            const int u = u0 + i, v = u / P::UPE, p = (u % P::UPE) / UPM, gr = (u % UPM) / P::KBN, kb = u % P::KBN;
            const int64_t* tp = (MATS == 1 || p == 0) ? trp[0] : trp[MATS - 1];
            const uint32_t* base = (const uint32_t*) tp[pl.ve[v]];
            uint32_t* d = ring + (i % STAGES) * SLOT;
            const int K = pl.vk[v][p];
            if (K == 3) issue_unit<3, P::NTN>(base, gr, kb, d, lane);
            else if (KMAX >= 4 && K == 4) issue_unit<(KMAX >= 4 ? 4 : 3), P::NTN>(base, gr, kb, d, lane);
            else if (KMAX >= 5) issue_unit<(KMAX >= 5 ? 5 : 3), P::NTN>(base, gr, kb, d, lane);
        }
        cp_commit();
    };
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) issue(i);
    pdl_wait(); pdl_launch();
    if (n <= 0) { cp_wait<0>(); return; }
    int u = u0, v = u / P::UPE, p = (u % P::UPE) / UPM, gr = (u % UPM) / P::KBN, kb = u % P::KBN;
    const int xn = lane >> 2, kq = (lane & 3) * 2;
    // the lane's B column per fragment: pick (f * 8 + xn) of virtual expert v
    const uint32_t* xr[FR];
    bool live[FR];
    int cnt = 0, kbits = 0;
    LaneDec dec;
    auto bind = [&]()
    {
        cnt = pl.vcnt[v];
        const int kn = pl.vk[v][p];
        if (kn != kbits)                                // the lane's decode constants follow the matrix's bit width
            dec = kn == 3 ? lane_dec<3>(lane) : (kn == 4 ? lane_dec<(KMAX >= 4 ? 4 : 3)>(lane) : lane_dec<(KMAX >= 5 ? 5 : 3)>(lane));
        kbits = kn;
        #pragma unroll
        for (int f = 0; f < FR; ++f)
        {
            const int s = f * 8 + xn;
            live[f] = s < cnt;
            xr[f] = (const uint32_t*) (xptr(live[f] ? (int) pl.plist[pl.voff[v] + s] : 0, p) + kq);
        }
    };
    bind();
    float c[FR][4][4];
    #pragma unroll
    for (int f = 0; f < FR; ++f)
        #pragma unroll
        for (int t = 0; t < 4; ++t) c[f][t][0] = c[f][t][1] = c[f][t][2] = c[f][t][3] = 0.f;
    for (int i = 0; i < n; ++i)
    {
        issue(i + STAGES - 1);
        cp_wait<STAGES - 1>();
        __syncwarp();
        const uint32_t* st = ring + (i % STAGES) * SLOT;
        // one unit: 4 k-tiles x 4 n-tiles of a K-bit matrix, decoded once and used by NF fragments (warp-uniform choice)
        auto unit = [&](auto k_tag, auto nf_tag)
        {
            constexpr int K = decltype(k_tag)::value, NF = decltype(nf_tag)::value, TW = tile_words(K);
            const LaneDec& d = dec;
            #pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                const int k0 = (kb * 4 + q) * 16;
                uint32_t xb[NF][2];
                #pragma unroll
                for (int f = 0; f < NF; ++f)
                {
                    xb[f][0] = live[f] ? __ldg(xr[f] + k0 / 2) : 0u;
                    xb[f][1] = live[f] ? __ldg(xr[f] + k0 / 2 + 4) : 0u;
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                {
                    uint32_t af[4];
                    dq8_a<K>(st + (q * 4 + t) * TW, d, af);
                    #pragma unroll
                    for (int f = 0; f < NF; ++f) mma16816(c[f][t], af, xb[f][0], xb[f][1]);
                }
            }
        };
        using I1 = std::integral_constant<int, 1>;
        using IF = std::integral_constant<int, FR>;
        const bool two = FR > 1 && cnt > 8;
        if (kbits == 3) { if (two) unit(std::integral_constant<int, 3>{}, IF{}); else unit(std::integral_constant<int, 3>{}, I1{}); }
        else if (KMAX >= 4 && kbits == 4)
        {
            using K4 = std::integral_constant<int, (KMAX >= 4 ? 4 : 3)>;
            if (two) unit(K4{}, IF{}); else unit(K4{}, I1{});
        }
        else if (KMAX >= 5)
        {
            using K5 = std::integral_constant<int, (KMAX >= 5 ? 5 : 3)>;
            if (two) unit(K5{}, IF{}); else unit(K5{}, I1{});
        }
        __syncwarp();
        ++kb;
        if (kb == P::KBN || i == n - 1)
        {
            red_flush(pl, v, p, gr, c, dst);
            #pragma unroll
            for (int f = 0; f < FR; ++f)
                #pragma unroll
                for (int t = 0; t < 4; ++t) c[f][t][0] = c[f][t][1] = c[f][t][2] = c[f][t][3] = 0.f;
            if (kb == P::KBN)
            {
                kb = 0;
                if (++gr == P::G) { gr = 0; if (++p == MATS) { p = 0; ++v; } if (i < n - 1) bind(); }
            }
        }
    }
    cp_wait<0>();
}

template <int KMAX>
__global__ void __launch_bounds__(NT, 2) moe3i_a_kernel(const int* __restrict__ ids, const int* rows_dev, int rmax, int nb,
    Tabs tb, Scr sc)
{
    __shared__ Plan pl;
    extern __shared__ __align__(16) uint32_t ring_dyn[];
    constexpr int STAGES = ring_stages(KMAX);
    make_plan<1>(ids, nrows(rows_dev, rmax), nb, tb.kb, pl);
    const int64_t* trp[2] = {tb.tr[0], tb.tr[1]};
    const bool shared = tb.gsu != nullptr;
    stream3<2, I, H, STAGES, KMAX>(pl, trp,
        [&](int pick, int p) { return shared ? sc.xr + (size_t) (pick / TOPK) * H : sc.xr + ((size_t) pick * 2 + p) * H; },
        [&](int pick, int p) { return sc.gu + ((size_t) pick * 2 + p) * I; },
        ring_dyn + (threadIdx.x >> 5) * STAGES * unit_words(KMAX));
}

// act: warp job (pick, 128-chunk of I): gate / up = svh * H128(k_inv * acc + k_bias * sum(x)); Inkling's SiTU with gamma;
// times down suh, H128 -> the rotated down input (fp16) and the chunk's sum. Clears the gate / up scratch.
__global__ void __launch_bounds__(NT) moe3i_act_kernel(const int* __restrict__ ids, const float* __restrict__ gam,
    const int* rows_dev, int rmax, int nb, Tabs tb, Scr sc)
{
    __shared__ Plan pl;
    make_plan<0>(ids, nrows(rows_dev, rmax), nb, tb.kb, pl);
    pdl_wait();
    pdl_launch();
    const int lane = threadIdx.x & 31, jobs = pl.np * IC;
    const float ki = k_inv(), kbias = k_bias();
    const bool shared = tb.gsu != nullptr;
    for (int q = blockIdx.x * WPB + (threadIdx.x >> 5); q < jobs; q += gridDim.x * WPB)
    {
        const int i = pl.plist[q / IC], cc = q % IC, e = ids[i];
        const float gamma = gam[i];
        const int col = cc * 128 + lane * 4;
        float g[4], u[4];
        float xsum = 0.f;
        if (shared)
        {
            const float* xs = sc.xsp + (size_t) (i / TOPK) * HC;
            xsum = warp_sum(__ldcg(xs + lane) + (lane < HC - 32 ? __ldcg(xs + 32 + lane) : 0.f));
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            if (!shared)
            {
                const float* xs = sc.xsp + ((size_t) i * 2 + p) * HC;
                xsum = warp_sum(__ldcg(xs + lane) + (lane < HC - 32 ? __ldcg(xs + 32 + lane) : 0.f));
            }
            const float4 v = acc_take4(sc.gu + ((size_t) i * 2 + p) * I + col);
            const float bs = kbias * xsum;
            float* o = p ? u : g;
            o[0] = fmaf(ki, v.x, bs); o[1] = fmaf(ki, v.y, bs); o[2] = fmaf(ki, v.z, bs); o[3] = fmaf(ki, v.w, bs);
            had128(o[0], o[1], o[2], o[3], lane);
            const float4 s = ld_h4((const half*) tb.sv[p][e] + col);
            o[0] *= s.x; o[1] *= s.y; o[2] *= s.z; o[3] *= s.w;
        }
        const float4 d = ld_h4((const half*) tb.su[2][e] + col);
        float a[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = situ(g[k], u[k], gamma);
        a[0] *= d.x; a[1] *= d.y; a[2] *= d.z; a[3] *= d.w;
        had128(a[0], a[1], a[2], a[3], lane);
        const half2 o01 = __floats2half2_rn(a[0], a[1]), o23 = __floats2half2_rn(a[2], a[3]);
        uint2 ov; ov.x = *(const uint32_t*) &o01; ov.y = *(const uint32_t*) &o23;
        *(uint2*) (sc.ar + (size_t) i * I + col) = ov;
        const float2 f01 = __half22float2(o01), f23 = __half22float2(o23);
        const float s = warp_sum((f01.x + f01.y) + (f23.x + f23.y));
        if (lane == 0) sc.asp[(size_t) i * IC + cc] = s;
    }
}

template <int KMAX>
__global__ void __launch_bounds__(NT, 2) moe3i_b_kernel(const int* __restrict__ ids, const int* rows_dev, int rmax, int nb,
    Tabs tb, Scr sc)
{
    __shared__ Plan pl;
    extern __shared__ __align__(16) uint32_t ring_dyn[];
    constexpr int STAGES = ring_stages(KMAX);
    make_plan<2>(ids, nrows(rows_dev, rmax), nb, tb.kb, pl);
    const int64_t* trp[1] = {tb.tr[2]};
    const bool shared = tb.dsv != nullptr;
    stream3<1, H, I, STAGES, KMAX>(pl, trp,
        [&](int pick, int) { return sc.ar + (size_t) pick * I; },
        [&](int pick, int) { return sc.ds + (size_t) (shared ? pick / TOPK : pick) * H; },
        ring_dyn + (threadIdx.x >> 5) * STAGES * unit_words(KMAX));
}

// fin, per-expert down vectors: block job (row r, 128-chunk of H); warp k takes pick r * 8 + k:
// out[r] = sum_k [bf16] svh_k * H128(k_inv * ds + k_bias * sum(a)), summed over k = 0..7 in order. Clears ds.
// fin, shared down_svh: warp job (row r, 128-chunk): out[r] = svh * H128(k_inv * ds[r] + k_bias * sum over the row's picks
// of sum(a)), the experts' raw sums having met in ds[r]. Clears ds.
__global__ void __launch_bounds__(NT) moe3i_fin_kernel(const int* __restrict__ ids, const int* rows_dev, int rmax, int nb,
    Tabs tb, Scr sc, float* __restrict__ out, int round_bf16)
{
    __shared__ float4 part[WPB][32];
    pdl_wait();
    const int R = nrows(rows_dev, rmax), lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float ki = k_inv(), kbias = k_bias();
    if (tb.dsv)
    {
        for (int q = blockIdx.x * WPB + w; q < R * HC; q += gridDim.x * WPB)
        {
            const int r = q / HC, cc = q % HC, col = cc * 128 + lane * 4;
            float as = 0.f;
            bool any = false;
            #pragma unroll
            for (int k = 0; k < TOPK; ++k)
            {
                const int i = r * TOPK + k;
                if ((unsigned) ids[i] < (unsigned) nb) { any = true; if (lane < IC) as += __ldcg(sc.asp + (size_t) i * IC + lane); }
            }
            float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
            if (any)
            {
                const float4 v = acc_take4(sc.ds + (size_t) r * H + col);
                const float bs = kbias * warp_sum(as);
                float d0 = fmaf(ki, v.x, bs), d1 = fmaf(ki, v.y, bs), d2 = fmaf(ki, v.z, bs), d3 = fmaf(ki, v.w, bs);
                had128(d0, d1, d2, d3, lane);
                const float4 s = ld_h4(tb.dsv + col);
                o = make_float4(d0 * s.x, d1 * s.y, d2 * s.z, d3 * s.w);
            }
            *(float4*) (out + (size_t) r * H + col) = o;
        }
        return;
    }
    for (int job = blockIdx.x; job < R * HC; job += gridDim.x)
    {
        const int r = job / HC, cc = job % HC, col = cc * 128 + lane * 4, i = r * TOPK + w, e = ids[i];
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
        if ((unsigned) e < (unsigned) nb)
        {
            const float4 v = acc_take4(sc.ds + (size_t) i * H + col);
            const float* as = sc.asp + (size_t) i * IC;
            const float bs = kbias * warp_sum(lane < IC ? __ldcg(as + lane) : 0.f);
            float d0 = fmaf(ki, v.x, bs), d1 = fmaf(ki, v.y, bs), d2 = fmaf(ki, v.z, bs), d3 = fmaf(ki, v.w, bs);
            had128(d0, d1, d2, d3, lane);
            const float4 s = ld_h4((const half*) tb.sv[2][e] + col);
            o = make_float4(d0 * s.x, d1 * s.y, d2 * s.z, d3 * s.w);
            if (round_bf16) o = make_float4(bf16r(o.x), bf16r(o.y), bf16r(o.z), bf16r(o.w));
        }
        part[w][lane] = o;
        __syncthreads();
        if (w == 0)
        {
            float4 t = part[0][lane];
            #pragma unroll
            for (int k = 1; k < WPB; ++k) { const float4 u = part[k][lane]; t.x += u.x; t.y += u.y; t.z += u.z; t.w += u.w; }
            *(float4*) (out + (size_t) r * H + col) = t;
        }
        __syncthreads();
    }
}

template <typename Kern, typename... Args>
void launch_pdl(Kern k, dim3 grid, cudaStream_t st, int dsmem, bool pdl, Args... args)
{
    cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid; cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = dsmem; cfg.stream = st; cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}

template <int KMAX>
void launch_streams(dim3 grid, cudaStream_t st, bool a_not_b, const int* id, const int* rd, int R, int nb, Tabs tb, Scr sc)
{
    constexpr int smem = ring_stages(KMAX) * unit_words(KMAX) * 4 * WPB;
    static bool attr = [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe3i_a_kernel<KMAX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(moe3i_b_kernel<KMAX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        return true;
    }();
    (void) attr;
    if (a_not_b) launch_pdl(moe3i_a_kernel<KMAX>, grid, st, smem, true, id, rd, R, nb, tb, sc);
    else launch_pdl(moe3i_b_kernel<KMAX>, grid, st, smem, true, id, rd, R, nb, tb, sc);
}
}  // namespace

// scratch for up to MAXR rows (gu and ds must start at zero; the chain leaves them at zero)
#ifdef MOE3I_ATOMIC
#define ACC_DTYPE at::kFloat
#else
#define ACC_DTYPE at::kLong
#endif
std::vector<at::Tensor> moe3i_scratch(at::Device dev)
{
    auto f = at::TensorOptions().device(dev);
    return {at::zeros({MAXP, 2, H}, f.dtype(at::kHalf)), at::zeros({MAXP, 2, HC}, f.dtype(at::kFloat)),
            at::zeros({MAXP, 2, I}, f.dtype(ACC_DTYPE)), at::zeros({MAXP, I}, f.dtype(at::kHalf)),
            at::zeros({MAXP, IC}, f.dtype(at::kFloat)), at::zeros({MAXP, H}, f.dtype(ACC_DTYPE))};
}

// x [Rmax][H] bf16; ids [Rmax][8] int32 (local band ids, < 0 or >= nb = not in the band; distinct within a row);
// gam [Rmax][8] fp32 routing weights; kbits int8 [3][nb] (bit width of gate, up, down per local id, 3..5), kmax >= every
// entry; tables: int64 [nb] device pointers (trellis int16 [K/16][N/16][16 K], suh fp16 [K], svh fp16 [N]) for gate,
// up, down; gsu / dsv: the layer-shared gate / up input vector and down output vector (fp16 [H]) or empty tensors (then
// the per-expert gate / up suh and down svh tables are used); out [Rmax][H] fp32, rows [0, R) overwritten. rows: int32 [1]
// device word with the row count R <= Rmax (empty tensor: R = Rmax). round_bf16: round each expert's down output to bf16
// before the sum (per-expert down vectors only).
void moe3i_forward(at::Tensor x, at::Tensor ids, at::Tensor gam, int64_t nb, at::Tensor kbits, int64_t kmax,
                   at::Tensor gtr, at::Tensor gsu, at::Tensor gsv, at::Tensor utr, at::Tensor usu, at::Tensor usv,
                   at::Tensor dtr, at::Tensor dsu, at::Tensor dsv, at::Tensor sh_su, at::Tensor sh_sv,
                   at::Tensor xr, at::Tensor xsp, at::Tensor gu, at::Tensor ar, at::Tensor asp, at::Tensor ds, at::Tensor out,
                   at::Tensor rows, bool round_bf16)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    const int R = (int) x.size(0);
    TORCH_CHECK(R >= 1 && R <= MAXR && x.size(1) == H && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "moe3i: x");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.size(0) == R && ids.size(1) == TOPK && ids.is_contiguous(), "moe3i: ids");
    TORCH_CHECK(gam.scalar_type() == at::kFloat && gam.size(0) == R && gam.size(1) == TOPK && gam.is_contiguous(), "moe3i: gam");
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.size(0) >= R && out.size(1) == H && out.is_contiguous(), "moe3i: out");
    TORCH_CHECK(nb >= 1 && nb <= NBMAX && gtr.numel() >= nb, "moe3i: nb");
    TORCH_CHECK(kbits.scalar_type() == at::kChar && kbits.numel() == 3 * nb && kbits.is_contiguous(), "moe3i: kbits int8 [3][nb]");
    TORCH_CHECK(kmax >= KLO && kmax <= KHI, "moe3i: kmax 3..5");
    for (const at::Tensor& v : {sh_su, sh_sv})
        TORCH_CHECK(v.numel() == 0 || (v.numel() == H && v.scalar_type() == at::kHalf && v.is_contiguous()), "moe3i: shared vectors fp16 [4608]");
    TORCH_CHECK(!(round_bf16 && sh_sv.numel()), "moe3i: round_bf16 needs per-expert down vectors (a shared down_svh sums first)");
    const int* rd = rows.numel() ? (const int*) rows.data_ptr() : nullptr;
    Tabs tb{{(const int64_t*) gtr.data_ptr(), (const int64_t*) utr.data_ptr(), (const int64_t*) dtr.data_ptr()},
            {(const int64_t*) gsu.data_ptr(), (const int64_t*) usu.data_ptr(), (const int64_t*) dsu.data_ptr()},
            {(const int64_t*) gsv.data_ptr(), (const int64_t*) usv.data_ptr(), (const int64_t*) dsv.data_ptr()},
            (const int8_t*) kbits.data_ptr(),
            sh_su.numel() ? (const half*) sh_su.data_ptr() : nullptr, sh_sv.numel() ? (const half*) sh_sv.data_ptr() : nullptr};
    TORCH_CHECK(gu.scalar_type() == ACC_DTYPE && ds.scalar_type() == ACC_DTYPE, "moe3i: gu / ds scratch dtype (use scratch() of this build)");
    Scr sc{(half*) xr.data_ptr(), (float*) xsp.data_ptr(), (acc_t*) gu.data_ptr(), (half*) ar.data_ptr(), (float*) asp.data_ptr(),
           (acc_t*) ds.data_ptr()};
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int* id = (const int*) ids.data_ptr(); const float* g = (const float*) gam.data_ptr();
    const int nb_ = (int) nb;
    const int g_rot = tb.gsu ? std::min(sms, (R * HC + WPB - 1) / WPB) : std::min(sms, (R * TOPK * 2 * HC + WPB - 1) / WPB);
    const int g_act = std::min(sms, (R * TOPK * IC + WPB - 1) / WPB);
    const int g_fin = tb.dsv ? std::min(sms, (R * HC + WPB - 1) / WPB) : std::min(2 * sms, R * HC);
#ifndef MOE3I_X_STAGES          // experiment: launch only some kernels (bits: 1 rot, 2 A, 4 act, 8 B, 16 fin), for timing
#define MOE3I_X_STAGES 31
#endif
    constexpr int stages = MOE3I_X_STAGES;
    if (stages & 1)
        launch_pdl(moe3i_rot_kernel, dim3(g_rot), st, 0, false, (const __nv_bfloat16*) x.data_ptr(), id, rd, R, nb_, tb, sc);
    auto streams = [&](bool a) {
        if (kmax <= 3) launch_streams<3>(dim3(sms * 2), st, a, id, rd, R, nb_, tb, sc);
        else if (kmax == 4) launch_streams<4>(dim3(sms * 2), st, a, id, rd, R, nb_, tb, sc);
        else launch_streams<5>(dim3(sms * 2), st, a, id, rd, R, nb_, tb, sc);
    };
    if (stages & 2) streams(true);
    if (stages & 4) launch_pdl(moe3i_act_kernel, dim3(g_act), st, 0, true, id, g, rd, R, nb_, tb, sc);
    if (stages & 8) streams(false);
    if (stages & 16)
        launch_pdl(moe3i_fin_kernel, dim3(g_fin), st, 0, true, id, rd, R, nb_, tb, sc, (float*) out.data_ptr(), (int) round_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scratch", &moe3i_scratch);
    m.def("forward", &moe3i_forward);
    m.attr("H") = H; m.attr("I") = I; m.attr("MAXR") = MAXR; m.attr("NBMAX") = NBMAX; m.attr("KLO") = KLO; m.attr("KHI") = KHI;
}
