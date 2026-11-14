// Inkling 2 Small: Grace-core band for the coldest routed experts (track E).
//
// A pool of pinned threads spins on the ISSUE word of a slot in pinned host memory. The GB300 writes a job into the
// slot (layer, R <= 64 rows of the MoE input x in bf16, per row up to 8 local tail ids + fp32 routing weights) and
// releases ISSUE = seq (st.release.sys). The pool computes every picked tail expert in place from the published
// weights, in the layout vLLM's GPU kernels read (rows shuffled in blocks of 32, block scales swizzled R128c4: the
// band reads the engine's own host copies, nothing is duplicated), writes out[r][:] = the fp32 sum of row r's tail
// experts in slot (top-k) order, starting from 0, and releases DONE = seq.
//
// Formats (per layer: cb_add_layer for the published NVFP4 / MXFP8 weights, the lossless profile; cb_add_layer_i4 for
// Hecate's GPTQ int4, the fast profile) and arithmetic (products exact, fp32 accumulation; no activation quantization):
//   INT4 (tools/quant/gptq_int4.py's format, rows converted to row-major nibbles at load): code c = q + 8 -> bf16 c - 8
//     by table lookup (exact), BFMMLA as NVFP4 below without block scales, times the row's fp32 scale at the end
//   NVFP4 (layers 0-39; e2m1 codes 2 per byte, low nibble first; e4m3 block scale per 16; w = e2m1 * s * amax / 2688):
//     e2m1 -> bf16 by table lookup (exact), x and the activation as bf16 (exact: they are bf16 values), BFMMLA
//     (2 x-rows x 2 weight rows x 4 k per instruction; products exact, sums rounded to odd in fp32), each 16-block
//     sum times its e4m3 scale (fp32 FMA), times amax / 2688 at the end
//   MXFP8 (layers 40-47; e4m3 values; ue8m0 scale per 32): e4m3 * 2^-8 -> fp16 (exact, incl. subnormals), x * 64 and
//     activation * 64 in fp16 (exact: |x| <= sqrt(4608) * max|mlp_norm.weight| = 503, |act| <= 4 * 25 * 8 = 800),
//     FMLAL / FMLAL2 (fp16 products exact in fp32, IEEE fp32 FMA), each 32-block sum times 2^(E - 125)
// vLLM's GPU path quantizes x to NVFP4 / MXFP8 and uses tanh.approx; the band does neither, so it is closer to the
// dequantized model than the GPU path (both round the SiTU output to bf16 before the down projection).
//
// SiTU as vLLM ops/activation.py: (((softcap(g, 4) * sigmoid(g)) * softcap(u, 25)) * gamma), softcap(v, c) =
// c * tanh(v / c), in fp32 with accurate exp / tanh, rounded to bf16 (nearest even).
//
// Work split: phase X prepares the operands (NVFP4: each pick's x rows as BFMMLA pairs; MXFP8: x * 64 in fp16),
// phase A computes gate/up + SiTU (units of 8 stored rows = 4 gate/up pairs, split over threads across all picks),
// phase C the down projection (4-row groups split over threads; each group runs every pick, and each row sums its
// picks in slot order, so the result does not depend on the thread count). Two barriers and a gather per job.
#include <arm_neon.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <time.h>
#include <vector>

namespace {

constexpr int H = 4608, I = 2304, TOPK = 8, MAXR = 64, MAXE = MAXR * TOPK, NTMAX = 72;
constexpr int UNITS_A = 576;      // phase A units per pick: 36 tiles x 16 (m, m + 8) gate / up row pairs
constexpr int GROUPS_C = 1152;    // phase C 4-row groups: 36 tiles x 32 m
constexpr int NXT = 4;            // MXFP8: x vectors per weight pass
constexpr int NPT = 4;            // NVFP4: x pairs per weight pass
enum { NVFP4 = 0, MXFP8 = 1, INT4 = 2 };

// ---------------------------------------------------------------------------------------------------------------------
// slot (pinned host memory, mapped into the GPU): control words on their own lines, then the job, then the partial
struct alignas(256) Slot
{
    alignas(128) uint32_t issue;            // GPU -> CPU: sequence number of the posted job (release store)
    alignas(128) uint32_t done;             // CPU -> GPU: sequence number of the finished job (release store)
    alignas(128) uint32_t err;              // sticky error bits (CPU: 1 bad header, 2 tail id out of range; GPU: 0x100 wait timed out)
    alignas(128) int32_t layer;             // job header: layer, rows (one line)
    int32_t rows;
    int32_t pad[30];
    int32_t ids[MAXR][TOPK];                // local tail ids in [0, n), -1: not a tail expert
    float gam[MAXR][TOPK];                  // routing weights (fp32 gammas of the route kernel)
    alignas(256) uint16_t x[MAXR][H];       // bf16 MoE input rows
    alignas(256) float out[MAXR][H];        // fp32 partial: per row, its tail experts summed in slot order
};

struct Layer
{
    int fmt = -1, n = 0;
    // NVFP4 / MXFP8: vLLM's layout (w13 = interleaved gate / up rows, shuffled; scales swizzled)
    const uint8_t *w13 = nullptr, *s13 = nullptr, *w2 = nullptr, *s2 = nullptr;
    size_t w13_es = 0, s13_es = 0, w2_es = 0, s2_es = 0;     // byte strides between experts
    std::vector<float> f13, f2;                             // per expert output factors (NVFP4: amax / 2688, MXFP8: 1)
    // INT4 (Hecate's GPTQ format): gate [I][H/2], up [I][H/2], down [H][I/2] row-major codes (q + 8, low nibble = even
    // k), fp32 per-row scales gate [I], up [I], down [H]; w = (code - 8) * s[row]
    const uint8_t *wg = nullptr, *wu = nullptr, *wd = nullptr;
    const float *sg = nullptr, *su = nullptr, *sd = nullptr;
    size_t wg_es = 0, wu_es = 0, wd_es = 0, sg_es = 0, su_es = 0, sd_es = 0;
};

struct Job
{
    int L = -1, R = 0, P = 0, E = 0, NP = 0, ok = 0, bad = 0;
    int pk_e[MAXE], pk_n[MAXE], pk_0[MAXE], pk_pp[MAXE];   // pick: expert, entries, first entry, first pair
    int en_row[MAXE]; float en_gam[MAXE];       // entries in pick order, then row order
    int pp_e0[MAXE], pp_n[MAXE];                // NVFP4 pairs: first entry, entries (1 or 2)
    int rw_n[MAXR], rw_e[MAXR][TOPK];           // each row's entries in slot (top-k) order
    int nused, used[MAXR];                      // rows with at least one entry
    std::vector<int16_t> seen;
};

struct alignas(128) Flag { std::atomic<uint32_t> v{0}; };

struct Pool
{
    std::vector<Layer> layers;
    Slot* slot = nullptr;
    int nt = 0, core0 = 0, wait_mode = 0, prefetch = 0;     // prefetch: whole rows of the next unit / group into L2
    std::vector<int> cores;                                  // thread t -> cores[t] (CPUBAND_CORES), else core0 + t
    std::vector<std::thread> th;
    std::atomic<bool> stop{false};
    alignas(128) std::atomic<uint32_t> bar_sense{0};
    Flag bar_flag[NTMAX];
    Job job[NTMAX];                              // each thread parses the job itself (identical, no barrier)
    // NVFP4 operands: pairs of a pick's rows, per 16-block 4 x 8 bf16 [r0 k{0,2,4,6}, r1 ..], [.. {8,10,12,14}],
    // [.. {1,3,5,7}], [.. {9,11,13,15}] (the k sets of the decoded weight operands)
    alignas(256) bfloat16_t xpair[MAXE][2 * H];
    alignas(256) bfloat16_t apair[MAXE][2 * I];
    // MXFP8 operands: x * 64 and bf16(SiTU) * 64 in fp16, natural order
    alignas(256) float16_t xh[MAXR][H];
    alignas(256) float16_t act[MAXE][I];
    // stats (thread 0)
    uint64_t jobs = 0, picks = 0, entries = 0, ns_total = 0, ns_x = 0, ns_a = 0, ns_c = 0, bad = 0;
    uint64_t last[4] = {0, 0, 0, 0};
    double tick_ns = 1.0;
};
Pool* g_pool = nullptr;

inline uint64_t ticks() { uint64_t v; asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }
inline uint64_t tick_freq() { uint64_t v; asm volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

inline void split(int n, int t, int nt, int& a, int& b) { a = (int) ((int64_t) n * t / nt); b = (int) ((int64_t) n * (t + 1) / nt); }

// gather barrier (Hecate): each thread releases its own line, thread 0 acquires all of them, then releases the sense
void barrier(Pool& P, int t, uint32_t& gen)
{
    ++gen;
    if (t == 0)
    {
        for (int i = 1; i < P.nt; ++i) while (P.bar_flag[i].v.load(std::memory_order_acquire) != gen) asm volatile("yield");
        P.bar_sense.store(gen, std::memory_order_release);
    }
    else
    {
        P.bar_flag[t].v.store(gen, std::memory_order_release);
        while (P.bar_sense.load(std::memory_order_acquire) != gen) asm volatile("yield");
    }
}

// the last synchronization of a job: thread 0 waits for every thread's partial, the others go back to ISSUE
void gather(Pool& P, int t, uint32_t& gen)
{
    ++gen;
    if (t == 0) { for (int i = 1; i < P.nt; ++i) while (P.bar_flag[i].v.load(std::memory_order_acquire) != gen) asm volatile("yield"); }
    else P.bar_flag[t].v.store(gen, std::memory_order_release);
}

// ---------------------------------------------------------------------------------------------------------------------
// kernels: one 4-row group (stored rows 128 T + 32 q + m, q = 0..3; its scales are one 16-byte chunk per 4 blocks,
// bytes [q * 4 + block], chunks 512 bytes apart) against x operands. sums[j][q] for x vector j, row q.

// bf16 bytes of e2m1 codes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives
alignas(16) const uint8_t E2M1_BFHI[16] = {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
alignas(16) const uint8_t E2M1_BFLO[16] = {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
// bf16 bytes of int4 codes c = q + 8 (Hecate's GPTQ format): the values c - 8 = -8 .. 7
alignas(16) const uint8_t I4_BFHI[16] = {0xC1, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xBF, 0x00, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40};
alignas(16) const uint8_t I4_BFLO[16] = {0x00, 0xE0, 0xC0, 0xA0, 0x80, 0x40, 0x00, 0x80, 0x00, 0x80, 0x00, 0x40, 0x80, 0xA0, 0xC0, 0xE0};

// NVFP4, weight rows (2 PR, 2 PR + 1) over the whole K = 64 nc4 against NXP x pairs. Two weight rows' bytes are
// interleaved per 4 bytes (ZIP1/ZIP2 .4s), so the table decode of the low / high nibbles yields BFMMLA operands
// [row a k{0,2,4,6}, row b k{0,2,4,6}] etc. directly. acc[j] lanes: (x row i, weight row w) at 2 i + w.
template <int PR, int NXP, bool I4>
inline void nv_pass(const uint8_t* ra, const uint8_t* rb, const uint8_t* sc, int nc4, const bfloat16_t* const* xp, float32x4_t (&acc)[NXP])
{
    const uint8x16_t th = vld1q_u8(I4 ? I4_BFHI : E2M1_BFHI), tl = vld1q_u8(I4 ? I4_BFLO : E2M1_BFLO), m4 = vdupq_n_u8(15);
    const float32x4_t two120 = vdupq_n_f32(1.329227995784916e36f);   // 2^120
    for (int j = 0; j < NXP; ++j) acc[j] = vdupq_n_f32(0.f);
    for (int c = 0; c < nc4; ++c)
    {
        // NVFP4: the two rows' e4m3 scales -> fp32 (byte << 20 is e4m3 * 2^-120, denormal for subnormal e4m3; * 2^120
        // exact). INT4: no block scales (one scale per row, applied by the caller)
        float64x2_t Zl = vdupq_n_f64(0.0), Zh = Zl;
        if constexpr (!I4)
        {
            const uint16x8_t s16 = vmovl_u8(vld1_u8(sc + (size_t) c * 512 + 8 * PR));
            const float32x4_t Sa = vmulq_f32(vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(s16)), 20)), two120);
            const float32x4_t Sb = vmulq_f32(vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(s16), 20)), two120);
            Zl = vreinterpretq_f64_f32(vzip1q_f32(Sa, Sb)); Zh = vreinterpretq_f64_f32(vzip2q_f32(Sa, Sb));
        }
        const uint32x4_t A0 = vreinterpretq_u32_u8(vld1q_u8(ra + c * 32)), A1 = vreinterpretq_u32_u8(vld1q_u8(ra + c * 32 + 16));
        const uint32x4_t B0 = vreinterpretq_u32_u8(vld1q_u8(rb + c * 32)), B1 = vreinterpretq_u32_u8(vld1q_u8(rb + c * 32 + 16));
        #pragma GCC unroll 4
        for (int blk = 0; blk < 4; ++blk)
        {
            const uint32x4_t A = blk < 2 ? A0 : A1, B = blk < 2 ? B0 : B1;
            const uint8x16_t Pb = vreinterpretq_u8_u32((blk & 1) ? vzip2q_u32(A, B) : vzip1q_u32(A, B));
            const uint8x16_t lo = vandq_u8(Pb, m4), hi = vshrq_n_u8(Pb, 4);
            const uint8x16_t lh = vqtbl1q_u8(th, lo), ll = vqtbl1q_u8(tl, lo), hh = vqtbl1q_u8(th, hi), hl = vqtbl1q_u8(tl, hi);
            const bfloat16x8_t V0 = vreinterpretq_bf16_u8(vzip1q_u8(ll, lh)), V1 = vreinterpretq_bf16_u8(vzip2q_u8(ll, lh));
            const bfloat16x8_t V2 = vreinterpretq_bf16_u8(vzip1q_u8(hl, hh)), V3 = vreinterpretq_bf16_u8(vzip2q_u8(hl, hh));
            // block scale per lane: [s(a), s(b), s(a), s(b)]
            const float32x4_t Sv = vreinterpretq_f32_f64(blk == 0 ? vdupq_laneq_f64(Zl, 0) : blk == 1 ? vdupq_laneq_f64(Zl, 1)
                                                         : blk == 2 ? vdupq_laneq_f64(Zh, 0) : vdupq_laneq_f64(Zh, 1));
            const int kb = (c * 4 + blk) * 32;
            #pragma GCC unroll 4
            for (int j = 0; j < NXP; ++j)
            {
                const bfloat16_t* x = xp[j] + kb;
                float32x4_t t = vdupq_n_f32(0.f);
                t = vbfmmlaq_f32(t, vld1q_bf16(x), V0);
                t = vbfmmlaq_f32(t, vld1q_bf16(x + 8), V1);
                t = vbfmmlaq_f32(t, vld1q_bf16(x + 16), V2);
                t = vbfmmlaq_f32(t, vld1q_bf16(x + 24), V3);
                acc[j] = I4 ? vaddq_f32(acc[j], t) : vfmaq_f32(acc[j], t, Sv);
            }
        }
    }
}

// NVFP4 / INT4, all 4 rows of the group per pass (both pairs per c4 step: 4 code streams instead of 2); acc[pair][j]
template <int NXP, bool I4>
inline void nv_pass4(const uint8_t* const rp[4], const uint8_t* sc, int nc4, const bfloat16_t* const* xp, float32x4_t (&acc)[2][NXP])
{
    const uint8x16_t th = vld1q_u8(I4 ? I4_BFHI : E2M1_BFHI), tl = vld1q_u8(I4 ? I4_BFLO : E2M1_BFLO), m4 = vdupq_n_u8(15);
    const float32x4_t two120 = vdupq_n_f32(1.329227995784916e36f);
    for (int pr = 0; pr < 2; ++pr) for (int j = 0; j < NXP; ++j) acc[pr][j] = vdupq_n_f32(0.f);
    for (int c = 0; c < nc4; ++c)
    {
        uint8x16_t sb = vdupq_n_u8(0);
        if constexpr (!I4) sb = vld1q_u8(sc + (size_t) c * 512);
        #pragma GCC unroll 2
        for (int pr = 0; pr < 2; ++pr)
        {
            float64x2_t Zl = vdupq_n_f64(0.0), Zh = Zl;
            if constexpr (!I4)
            {
                const uint16x8_t s16 = pr ? vmovl_high_u8(sb) : vmovl_u8(vget_low_u8(sb));
                const float32x4_t Sa = vmulq_f32(vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(s16)), 20)), two120);
                const float32x4_t Sb = vmulq_f32(vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(s16), 20)), two120);
                Zl = vreinterpretq_f64_f32(vzip1q_f32(Sa, Sb)); Zh = vreinterpretq_f64_f32(vzip2q_f32(Sa, Sb));
            }
            const uint8_t* ra = rp[2 * pr] + c * 32; const uint8_t* rb = rp[2 * pr + 1] + c * 32;
            const uint32x4_t A0 = vreinterpretq_u32_u8(vld1q_u8(ra)), A1 = vreinterpretq_u32_u8(vld1q_u8(ra + 16));
            const uint32x4_t B0 = vreinterpretq_u32_u8(vld1q_u8(rb)), B1 = vreinterpretq_u32_u8(vld1q_u8(rb + 16));
            #pragma GCC unroll 4
            for (int blk = 0; blk < 4; ++blk)
            {
                const uint32x4_t A = blk < 2 ? A0 : A1, B = blk < 2 ? B0 : B1;
                const uint8x16_t Pb = vreinterpretq_u8_u32((blk & 1) ? vzip2q_u32(A, B) : vzip1q_u32(A, B));
                const uint8x16_t lo = vandq_u8(Pb, m4), hi = vshrq_n_u8(Pb, 4);
                const uint8x16_t lh = vqtbl1q_u8(th, lo), ll = vqtbl1q_u8(tl, lo), hh = vqtbl1q_u8(th, hi), hl = vqtbl1q_u8(tl, hi);
                const bfloat16x8_t V0 = vreinterpretq_bf16_u8(vzip1q_u8(ll, lh)), V1 = vreinterpretq_bf16_u8(vzip2q_u8(ll, lh));
                const bfloat16x8_t V2 = vreinterpretq_bf16_u8(vzip1q_u8(hl, hh)), V3 = vreinterpretq_bf16_u8(vzip2q_u8(hl, hh));
                const float32x4_t Sv = vreinterpretq_f32_f64(blk == 0 ? vdupq_laneq_f64(Zl, 0) : blk == 1 ? vdupq_laneq_f64(Zl, 1)
                                                             : blk == 2 ? vdupq_laneq_f64(Zh, 0) : vdupq_laneq_f64(Zh, 1));
                const int kb = (c * 4 + blk) * 32;
                #pragma GCC unroll 2
                for (int j = 0; j < NXP; ++j)
                {
                    const bfloat16_t* x = xp[j] + kb;
                    float32x4_t t = vdupq_n_f32(0.f);
                    t = vbfmmlaq_f32(t, vld1q_bf16(x), V0);
                    t = vbfmmlaq_f32(t, vld1q_bf16(x + 8), V1);
                    t = vbfmmlaq_f32(t, vld1q_bf16(x + 16), V2);
                    t = vbfmmlaq_f32(t, vld1q_bf16(x + 24), V3);
                    acc[pr][j] = I4 ? vaddq_f32(acc[pr][j], t) : vfmaq_f32(acc[pr][j], t, Sv);
                }
            }
        }
    }
}

bool g_nv4 = true;     // 4-row passes for one x pair (4 code streams in flight; CPUBAND_NV4=0: 2-row passes)

template <int NXP, bool I4>
__attribute__((noinline)) void nv_group(const uint8_t* const rp[4], const uint8_t* sc, int nc4, const bfloat16_t* const* xp, float (*sums)[4])
{
    if constexpr (NXP == 1)       // (with 2+ x pairs the 4-row pass spills; one row pair per pass is faster there)
    {
        if (g_nv4)
        {
            float32x4_t acc2[2][NXP];
            nv_pass4<NXP, I4>(rp, sc, nc4, xp, acc2);
            for (int pr = 0; pr < 2; ++pr) for (int j = 0; j < NXP; ++j)
            {
                sums[2 * j][2 * pr] = vgetq_lane_f32(acc2[pr][j], 0); sums[2 * j][2 * pr + 1] = vgetq_lane_f32(acc2[pr][j], 1);
                sums[2 * j + 1][2 * pr] = vgetq_lane_f32(acc2[pr][j], 2); sums[2 * j + 1][2 * pr + 1] = vgetq_lane_f32(acc2[pr][j], 3);
            }
            return;
        }
    }
    float32x4_t acc[NXP];
    nv_pass<0, NXP, I4>(rp[0], rp[1], sc, nc4, xp, acc);
    for (int j = 0; j < NXP; ++j)
    {
        sums[2 * j][0] = vgetq_lane_f32(acc[j], 0); sums[2 * j][1] = vgetq_lane_f32(acc[j], 1);
        sums[2 * j + 1][0] = vgetq_lane_f32(acc[j], 2); sums[2 * j + 1][1] = vgetq_lane_f32(acc[j], 3);
    }
    nv_pass<1, NXP, I4>(rp[2], rp[3], sc, nc4, xp, acc);
    for (int j = 0; j < NXP; ++j)
    {
        sums[2 * j][2] = vgetq_lane_f32(acc[j], 0); sums[2 * j][3] = vgetq_lane_f32(acc[j], 1);
        sums[2 * j + 1][2] = vgetq_lane_f32(acc[j], 2); sums[2 * j + 1][3] = vgetq_lane_f32(acc[j], 3);
    }
}

// sums[j][q] for the 2 np x rows of np pairs (the partner of an odd last row is zero), in tiles of NPT pairs;
// i4: Hecate's int4 rows (no block scales, sc unused)
inline void nv_dot(const uint8_t* const rp[4], const uint8_t* sc, int nc4, const bfloat16_t* const* xp, int np, float (*sums)[4], bool i4 = false)
{
    for (int j0 = 0; j0 < np; j0 += NPT)
    {
        const int n = np - j0 < NPT ? np - j0 : NPT;
        if (i4)
            switch (n)
            {
                case 1: nv_group<1, true>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
                case 2: nv_group<2, true>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
                case 3: nv_group<3, true>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
                default: nv_group<4, true>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
            }
        else
            switch (n)
            {
                case 1: nv_group<1, false>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
                case 2: nv_group<2, false>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
                case 3: nv_group<3, false>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
                default: nv_group<4, false>(rp, sc, nc4, xp + j0, sums + 2 * j0); break;
            }
    }
}

// MXFP8: 128 bytes of row Q at c4 = 4 blocks of 32; per block a fresh sum times 2^(E - 125) (the block scale, the
// 2^8 of the fp16 decode and the 1/64 of x), added into the row's accumulator
template <int Q, int Q0, int NQ, int NX>
inline void mx_row(const uint8_t* rp, const uint8_t* scb, const float16_t* const* xs, int k0, float32x4_t (&acc)[NQ][NX])
{
    const int16x8_t b14 = vdupq_n_s16(0x4000);
    #pragma GCC unroll 1
    for (int b = 0; b < 4; ++b)
    {
        const uint32_t sbits = (uint32_t) (scb[Q * 4 + b] + 2) << 23;
        float s; memcpy(&s, &sbits, 4);
        const int8x16_t B0 = vld1q_s8((const int8_t*) rp + b * 32), B1 = vld1q_s8((const int8_t*) rp + b * 32 + 16);
        // e4m3 -> fp16 bits: sign-extended byte << 7 has the sign in bit 15 and a stray bit 14 for negatives: clear it
        const float16x8_t w0 = vreinterpretq_f16_s16(vbicq_s16(vshll_n_s8(vget_low_s8(B0), 7), b14));
        const float16x8_t w1 = vreinterpretq_f16_s16(vbicq_s16(vshll_high_n_s8(B0, 7), b14));
        const float16x8_t w2 = vreinterpretq_f16_s16(vbicq_s16(vshll_n_s8(vget_low_s8(B1), 7), b14));
        const float16x8_t w3 = vreinterpretq_f16_s16(vbicq_s16(vshll_high_n_s8(B1, 7), b14));
        #pragma GCC unroll 4
        for (int j = 0; j < NX; ++j)
        {
            const float16_t* xp = xs[j] + k0 + b * 32;
            const float16x8_t x0 = vld1q_f16(xp), x1 = vld1q_f16(xp + 8), x2 = vld1q_f16(xp + 16), x3 = vld1q_f16(xp + 24);
            float32x4_t t0 = vdupq_n_f32(0.f), t1 = vdupq_n_f32(0.f);
            t0 = vfmlalq_low_f16(t0, w0, x0); t1 = vfmlalq_high_f16(t1, w0, x0);
            t0 = vfmlalq_low_f16(t0, w1, x1); t1 = vfmlalq_high_f16(t1, w1, x1);
            t0 = vfmlalq_low_f16(t0, w2, x2); t1 = vfmlalq_high_f16(t1, w2, x2);
            t0 = vfmlalq_low_f16(t0, w3, x3); t1 = vfmlalq_high_f16(t1, w3, x3);
            acc[Q - Q0][j] = vfmaq_n_f32(acc[Q - Q0][j], vaddq_f32(t0, t1), s);
        }
    }
}

template <int Q0, int NQ, int NX>
inline void mx_pass(const uint8_t* const rp[4], const uint8_t* sc, int nc4, const float16_t* const* xs, float (*sums)[4])
{
    float32x4_t acc[NQ][NX];
    for (int q = 0; q < NQ; ++q) for (int j = 0; j < NX; ++j) acc[q][j] = vdupq_n_f32(0.f);
    for (int c = 0; c < nc4; ++c)
    {
        // ue8m0 bytes [q * 4 + block] -> fp32 2^(E - 125): bits (E + 2) << 23 (E <= 252)
        const uint8_t* scb = sc + (size_t) c * 512;
        const int k0 = c * 128;
        if constexpr (Q0 == 0)
        {
            mx_row<0, Q0, NQ, NX>(rp[0] + c * 128, scb, xs, k0, acc);
            mx_row<1, Q0, NQ, NX>(rp[1] + c * 128, scb, xs, k0, acc);
        }
        if constexpr (Q0 + NQ == 4)
        {
            mx_row<2, Q0, NQ, NX>(rp[2] + c * 128, scb, xs, k0, acc);
            mx_row<3, Q0, NQ, NX>(rp[3] + c * 128, scb, xs, k0, acc);
        }
    }
    for (int q = 0; q < NQ; ++q) for (int j = 0; j < NX; ++j) sums[j][Q0 + q] = vaddvq_f32(acc[q][j]);
}

template <int NX>
__attribute__((noinline)) void mx_group(const uint8_t* const rp[4], const uint8_t* sc, int nc4, const float16_t* const* xs, float (*sums)[4])
{
    if constexpr (NX <= 2) mx_pass<0, 4, NX>(rp, sc, nc4, xs, sums);
    else { mx_pass<0, 2, NX>(rp, sc, nc4, xs, sums); mx_pass<2, 2, NX>(rp, sc, nc4, xs, sums); }
}

inline void mx_dot(const uint8_t* const rp[4], const uint8_t* sc, int nc4, const float16_t* const* xs, int nx, float (*sums)[4])
{
    for (int j0 = 0; j0 < nx; j0 += NXT)
    {
        switch (nx - j0)
        {
            case 1: mx_group<1>(rp, sc, nc4, xs + j0, sums + j0); break;
            case 2: mx_group<2>(rp, sc, nc4, xs + j0, sums + j0); break;
            case 3: mx_group<3>(rp, sc, nc4, xs + j0, sums + j0); break;
            default: mx_group<4>(rp, sc, nc4, xs + j0, sums + j0); break;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// SiTU in fp32 (accurate exp / tanh; vLLM: (((softcap(g, 4) * sigmoid(g)) * softcap(u, 25)) * gamma), rounded to bf16)

inline float32x4_t v_exp_neg(float32x4_t x)
{
    // e^x for x <= 0 (x >= -104, below that the result is 0 in fp32): x = n ln2 + r, |r| <= ln2 / 2, degree-7 Taylor
    // (rel. error < 2e-9 on the interval); 2^n applied in two steps so that denormal results round correctly
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(0.f)), vdupq_n_f32(-104.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));     // ln2 hi (exact product, |n| <= 151)
    r = vfmsq_f32(r, n, vdupq_n_f32(1.428606765330187e-06f));              // ln2 lo
    float32x4_t p = vdupq_n_f32(1.f / 5040.f);
    p = vfmaq_f32(vdupq_n_f32(1.f / 720.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 120.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 24.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 6.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    const int32x4_t ni = vcvtq_s32_f32(n), n1 = vshrq_n_s32(ni, 1), n2 = vsubq_s32(ni, n1);   // n1, n2 >= -76
    const float32x4_t s1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n1, vdupq_n_s32(127)), 23));
    const float32x4_t s2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n2, vdupq_n_s32(127)), 23));
    return vmulq_f32(vmulq_f32(p, s1), s2);
}

inline float32x4_t v_sigmoid(float32x4_t g)
{
    // E = e^-|g|: g >= 0: 1 / (1 + E), g < 0: E / (1 + E) (no overflow, denormal tails exact to rounding)
    const float32x4_t E = v_exp_neg(vnegq_f32(vabsq_f32(g)));
    const float32x4_t d = vaddq_f32(vdupq_n_f32(1.f), E);
    return vdivq_f32(vbslq_f32(vcgeq_f32(g, vdupq_n_f32(0.f)), vdupq_n_f32(1.f), E), d);
}

inline float32x4_t v_tanh(float32x4_t y)
{
    // |y| < 0.5: odd Taylor series to y^17 (rel. error < 1e-9); else sign(y) (1 - E) / (1 + E), E = e^(-2|y|)
    const float32x4_t a = vabsq_f32(y);
    const float32x4_t y2 = vmulq_f32(y, y);
    float32x4_t p = vdupq_n_f32(0.000590027440f);
    p = vfmaq_f32(vdupq_n_f32(-0.001455834387f), p, y2);
    p = vfmaq_f32(vdupq_n_f32(0.003592128037f), p, y2);
    p = vfmaq_f32(vdupq_n_f32(-0.008863235530f), p, y2);
    p = vfmaq_f32(vdupq_n_f32(0.021869488536f), p, y2);
    p = vfmaq_f32(vdupq_n_f32(-0.053968253968f), p, y2);
    p = vfmaq_f32(vdupq_n_f32(0.133333333333f), p, y2);
    p = vfmaq_f32(vdupq_n_f32(-0.333333333333f), p, y2);
    const float32x4_t small = vfmaq_f32(y, vmulq_f32(p, y2), y);
    const float32x4_t E = v_exp_neg(vmulq_n_f32(a, -2.f));
    const float32x4_t big = vdivq_f32(vsubq_f32(vdupq_n_f32(1.f), E), vaddq_f32(vdupq_n_f32(1.f), E));
    const float32x4_t bigs = vbslq_f32(vdupq_n_u32(0x80000000u), y, big);   // copy the sign of y
    return vbslq_f32(vcltq_f32(a, vdupq_n_f32(0.5f)), small, bigs);
}

// bf16(SiTU) of 4 (gate, up) pairs as fp32 values
inline float32x4_t v_situ_bf16(float32x4_t g, float32x4_t u, float gam)
{
    const float32x4_t sig = v_sigmoid(g);
    const float32x4_t gc = vmulq_n_f32(v_tanh(vmulq_n_f32(g, 0.25f)), 4.f);
    const float32x4_t uc = vmulq_n_f32(v_tanh(vdivq_f32(u, vdupq_n_f32(25.f))), 25.f);
    const float32x4_t v = vmulq_n_f32(vmulq_f32(vmulq_f32(gc, sig), uc), gam);
    // round to bf16 (nearest even; finite values)
    uint32x4_t bits = vreinterpretq_u32_f32(v);
    bits = vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7FFF), vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1))));
    return vreinterpretq_f32_u32(vandq_u32(bits, vdupq_n_u32(0xFFFF0000u)));
}

// element k of x row slot s (0, 1) in the NVFP4 pair layout
inline int pair_pos(int k, int s)
{
    const int w = k & 15, vec = (w & 1) ? ((w & 8) ? 3 : 2) : ((w & 8) ? 1 : 0);
    return (k >> 4) * 32 + vec * 8 + s * 4 + ((w & 7) >> 1);
}

// ---------------------------------------------------------------------------------------------------------------------
void parse(Pool& P, Job& J, uint32_t& errbits)
{
    const Slot* S = P.slot;
    J.L = __atomic_load_n(&S->layer, __ATOMIC_RELAXED); J.R = __atomic_load_n(&S->rows, __ATOMIC_RELAXED);
    J.P = J.E = J.NP = 0; J.bad = 0;
    J.ok = J.R >= 1 && J.R <= MAXR && J.L >= 0 && J.L < (int) P.layers.size() && P.layers[J.L].n > 0;
    if (!J.ok) { errbits |= 1; return; }
    const int n = P.layers[J.L].n;
    if ((int) J.seen.size() < n) J.seen.assign(n, -1);
    int fill[MAXE];
    for (int r = 0; r < J.R; ++r)
    {
        J.rw_n[r] = 0;
        for (int k = 0; k < TOPK; ++k)
        {
            const int id = S->ids[r][k];
            if (id < 0) continue;
            if (id >= n) { J.bad = 1; continue; }
            int p = J.seen[id];
            if (p < 0) { p = J.P++; J.seen[id] = (int16_t) p; J.pk_e[p] = id; J.pk_n[p] = 0; }
            ++J.pk_n[p]; ++J.E;
        }
    }
    int s = 0;
    for (int p = 0; p < J.P; ++p)
    {
        J.pk_0[p] = s; fill[p] = 0; J.pk_pp[p] = J.NP;
        for (int i = 0; i < J.pk_n[p]; i += 2) { J.pp_e0[J.NP] = s + i; J.pp_n[J.NP] = J.pk_n[p] - i >= 2 ? 2 : 1; ++J.NP; }
        s += J.pk_n[p];
    }
    for (int r = 0; r < J.R; ++r)
        for (int k = 0; k < TOPK; ++k)
        {
            const int id = S->ids[r][k];
            if (id < 0 || id >= n) continue;
            const int p = J.seen[id], e = J.pk_0[p] + fill[p]++;
            J.en_row[e] = r; J.en_gam[e] = S->gam[r][k];
            J.rw_e[r][J.rw_n[r]++] = e;
        }
    for (int p = 0; p < J.P; ++p) J.seen[J.pk_e[p]] = -1;
    J.nused = 0;
    for (int r = 0; r < J.R; ++r) if (J.rw_n[r]) J.used[J.nused++] = r;
    if (J.bad) errbits |= 2;
}

// rows without a pick: zero partial (64-byte chunks split over threads)
void zero_rows(Pool& P, const Job& J, int t)
{
    int nz = J.R - J.nused;
    if (nz <= 0) return;
    int zr[MAXR], k = 0;
    for (int r = 0; r < J.R; ++r) if (!J.rw_n[r]) zr[k++] = r;
    int a, b; split(nz * (H / 16), t, P.nt, a, b);
    const float32x4_t z = vdupq_n_f32(0.f);
    for (int w = a; w < b; ++w)
    {
        float* o = P.slot->out[zr[w / (H / 16)]] + (w % (H / 16)) * 16;
        vst1q_f32(o, z); vst1q_f32(o + 4, z); vst1q_f32(o + 8, z); vst1q_f32(o + 12, z);
    }
}

// phase X. NVFP4 / INT4: every pair of every pick into the pair layout (16-blocks split over threads);
// MXFP8: the rows with picks -> fp16 * 64 (16-blocks split over threads)
void prepare_x(Pool& P, const Job& J, int fmt, int t)
{
    if (fmt != MXFP8)            // NVFP4 and INT4: BFMMLA pairs
    {
        int a, b; split(J.NP * (H / 16), t, P.nt, a, b);
        const uint16x8_t zero = vdupq_n_u16(0);
        for (int w = a; w < b; ++w)
        {
            const int pp = w / (H / 16), k = (w % (H / 16)) * 16, e0 = J.pp_e0[pp];
            const uint16_t* x0 = P.slot->x[J.en_row[e0]] + k;
            const uint16x8_t a0 = vld1q_u16(x0), a1 = vld1q_u16(x0 + 8);
            uint16x8_t b0 = zero, b1 = zero;
            if (J.pp_n[pp] == 2) { const uint16_t* x1 = P.slot->x[J.en_row[e0 + 1]] + k; b0 = vld1q_u16(x1); b1 = vld1q_u16(x1 + 8); }
            const uint16x8_t ea = vuzp1q_u16(a0, a1), oa = vuzp2q_u16(a0, a1), eb = vuzp1q_u16(b0, b1), ob = vuzp2q_u16(b0, b1);
            uint16_t* d = (uint16_t*) P.xpair[pp] + k * 2;
            vst1q_u16(d, vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(ea), vreinterpretq_u64_u16(eb))));
            vst1q_u16(d + 8, vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(ea), vreinterpretq_u64_u16(eb))));
            vst1q_u16(d + 16, vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(oa), vreinterpretq_u64_u16(ob))));
            vst1q_u16(d + 24, vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(oa), vreinterpretq_u64_u16(ob))));
        }
        // the zero partner slot of an odd pair's activation (the SiTU stores only fill real entries)
        int c, d; split(J.NP, t, P.nt, c, d);
        for (int pp = c; pp < d; ++pp)
            if (J.pp_n[pp] == 1) for (int k = 0; k < 2 * I; k += 8) vst1_u16((uint16_t*) P.apair[pp] + k + 4, vget_low_u16(zero));
    }
    else
    {
        int a, b; split(J.nused * (H / 16), t, P.nt, a, b);
        for (int blk = a; blk < b; ++blk)
        {
            const int r = J.used[blk / (H / 16)], k = (blk % (H / 16)) * 16;
            const uint16x8_t v0 = vld1q_u16(P.slot->x[r] + k), v1 = vld1q_u16(P.slot->x[r] + k + 8);
            const float32x4_t f0 = vmulq_n_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v0), 16)), 64.f);
            const float32x4_t f1 = vmulq_n_f32(vreinterpretq_f32_u32(vshll_high_n_u16(v0, 16)), 64.f);
            const float32x4_t f2 = vmulq_n_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v1), 16)), 64.f);
            const float32x4_t f3 = vmulq_n_f32(vreinterpretq_f32_u32(vshll_high_n_u16(v1, 16)), 64.f);
            vst1q_f16(P.xh[r] + k, vcvt_high_f16_f32(vcvt_f16_f32(f0), f1));
            vst1q_f16(P.xh[r] + k + 8, vcvt_high_f16_f32(vcvt_f16_f32(f2), f3));
        }
    }
}

// software prefetch into L2 (PRFM PLDL2KEEP) of whole rows and their scale lines: the rows are short (NVFP4 w13 2304 B,
// w2 1152 B) and a pass streams only two of them, too few misses in flight for the hardware prefetcher alone
inline void prefetch_rows(const uint8_t* const rp[4], int rowb, const uint8_t* sc, int nc4)
{
    for (int q = 0; q < 4; ++q) for (int o = 0; o < rowb; o += 64) __builtin_prefetch(rp[q] + o, 0, 2);
    for (int c = 0; c < nc4; ++c) __builtin_prefetch(sc + (size_t) c * 512, 0, 2);
}

// the 8 rows (+ scales) of phase A unit w
void prefetch_unit_a(const Layer& Ly, const Job& J, int w)
{
    const int fmt = Ly.fmt, p = w / UNITS_A, un = w % UNITS_A, e = J.pk_e[p];
    if (fmt == INT4)
    {
        const uint8_t* rp[4]; const uint8_t* rq[4];
        for (int q = 0; q < 4; ++q)
        {
            rp[q] = Ly.wg + (size_t) e * Ly.wg_es + (size_t) (4 * un + q) * (H / 2);
            rq[q] = Ly.wu + (size_t) e * Ly.wu_es + (size_t) (4 * un + q) * (H / 2);
        }
        prefetch_rows(rp, H / 2, nullptr, 0); prefetch_rows(rq, H / 2, nullptr, 0);
        return;
    }
    const int rowb = fmt == NVFP4 ? H / 2 : H, nc4 = fmt == NVFP4 ? H / 64 : H / 128;
    const int T = un >> 4, mp = un & 15, m = mp + 8 * (mp >> 3);
    for (int hlf = 0; hlf < 2; ++hlf)
    {
        const int mm = m + 8 * hlf;
        const uint8_t* rp[4];
        for (int q = 0; q < 4; ++q) rp[q] = Ly.w13 + (size_t) e * Ly.w13_es + (size_t) (128 * T + 32 * q + mm) * rowb;
        prefetch_rows(rp, rowb, Ly.s13 + (size_t) e * Ly.s13_es + ((size_t) T * nc4 * 32 + mm) * 16, nc4);
    }
}

// the 4 rows (+ scales) of phase C group gi, every pick
void prefetch_group_c(const Layer& Ly, const Job& J, int gi)
{
    const int fmt = Ly.fmt, rowb = fmt == MXFP8 ? I : I / 2, nc4 = fmt == NVFP4 ? I / 64 : I / 128;
    const int T = gi >> 5, m = gi & 31;
    for (int p = 0; p < J.P; ++p)
    {
        const int e = J.pk_e[p];
        const uint8_t* rp[4];
        if (fmt == INT4)
        {
            for (int q = 0; q < 4; ++q) rp[q] = Ly.wd + (size_t) e * Ly.wd_es + (size_t) (4 * gi + q) * (I / 2);
            prefetch_rows(rp, I / 2, nullptr, 0);
            continue;
        }
        for (int q = 0; q < 4; ++q) rp[q] = Ly.w2 + (size_t) e * Ly.w2_es + (size_t) (128 * T + 32 * q + m) * rowb;
        prefetch_rows(rp, rowb, Ly.s2 + (size_t) e * Ly.s2_es + ((size_t) T * nc4 * 32 + m) * 16, nc4);
    }
}

// phase A: units [a, b) over all picks; unit = 4 gate / up pairs. NVFP4 / MXFP8: 8 stored rows (m, m + 8 of tile T,
// 4 q each) of vLLM's shuffled w13; INT4: gate and up rows 4 un .. 4 un + 3
void phase_a(Pool& P, const Job& J, int t)
{
    const Layer& Ly = P.layers[J.L];
    const int fmt = Ly.fmt, rowb = fmt == MXFP8 ? H : H / 2, nc4 = fmt == MXFP8 ? H / 128 : H / 64;
    const bool i4 = fmt == INT4;
    int a, b; split(J.P * UNITS_A, t, P.nt, a, b);
    const float16_t* xs[MAXE];
    const bfloat16_t* xp[MAXE];
    float g[MAXE][4], u[MAXE][4];
    int lastp = -1;
    for (int w = a; w < b; ++w)
    {
        const int p = w / UNITS_A, un = w % UNITS_A;
        const int e = J.pk_e[p], e0 = J.pk_0[p], ne = J.pk_n[p], pp0 = J.pk_pp[p], np = (ne + 1) / 2;
        if (p != lastp)
        {
            if (fmt != MXFP8) for (int j = 0; j < np; ++j) xp[j] = P.xpair[pp0 + j];
            else for (int j = 0; j < ne; ++j) xs[j] = P.xh[J.en_row[e0 + j]];
            lastp = p;
        }
        if (P.prefetch && w + 1 < b) prefetch_unit_a(Ly, J, w + 1);
        int cq[4];                          // channel of row q
        float32x4_t fg, fu;                 // per-row output factors of the gate / up sums
        if (i4)
        {
            const int c0 = 4 * un;
            const uint8_t* rg[4]; const uint8_t* ru[4];
            for (int q = 0; q < 4; ++q)
            {
                rg[q] = Ly.wg + (size_t) e * Ly.wg_es + (size_t) (c0 + q) * (H / 2);
                ru[q] = Ly.wu + (size_t) e * Ly.wu_es + (size_t) (c0 + q) * (H / 2);
                cq[q] = c0 + q;
            }
            nv_dot(rg, nullptr, nc4, xp, np, g, true);
            nv_dot(ru, nullptr, nc4, xp, np, u, true);
            fg = vld1q_f32((const float*) ((const uint8_t*) Ly.sg + (size_t) e * Ly.sg_es) + c0);
            fu = vld1q_f32((const float*) ((const uint8_t*) Ly.su + (size_t) e * Ly.su_es) + c0);
        }
        else
        {
            // vLLM shuffle_rows: stored row 8 i + j of a 32-block holds logical row 4 j + i, so the gate / up pair
            // (logical 2 c', 2 c' + 1) is stored at (m, m + 8), m in [0, 8) or [16, 24)
            const int T = un >> 4, mp = un & 15, m = mp + 8 * (mp >> 3);
            const uint8_t* W = Ly.w13 + (size_t) e * Ly.w13_es;
            const uint8_t* Sx = Ly.s13 + (size_t) e * Ly.s13_es;
            for (int hlf = 0; hlf < 2; ++hlf)
            {
                const int mm = m + 8 * hlf;
                const uint8_t* rp[4];
                for (int q = 0; q < 4; ++q) rp[q] = W + (size_t) (128 * T + 32 * q + mm) * rowb;
                const uint8_t* sc = Sx + ((size_t) T * nc4 * 32 + mm) * 16;
                if (fmt == NVFP4) nv_dot(rp, sc, nc4, xp, np, hlf ? u : g);
                else mx_dot(rp, sc, nc4, xs, ne, hlf ? u : g);
            }
            // stored m = 8 i + j (i in {0, 2}) holds logical 4 j + i: channel c = 64 T + 16 q + 2 j + i / 2
            const int c0 = 64 * T + 2 * (m & 7) + (m >> 4);
            for (int q = 0; q < 4; ++q) cq[q] = c0 + 16 * q;
            fg = fu = vdupq_n_f32(Ly.f13[e]);
        }
        for (int j = 0; j < ne; ++j)
        {
            const float32x4_t av = v_situ_bf16(vmulq_f32(vld1q_f32(g[j]), fg), vmulq_f32(vld1q_f32(u[j]), fu), J.en_gam[e0 + j]);
            if (fmt != MXFP8)
            {
                const uint16x4_t hb = vshrn_n_u32(vreinterpretq_u32_f32(av), 16);
                uint16_t* A = (uint16_t*) P.apair[pp0 + j / 2];
                for (int q = 0; q < 4; ++q) A[pair_pos(cq[q], j & 1)] = hb[q];
            }
            else
            {
                const float16x4_t hv = vcvt_f16_f32(vmulq_n_f32(av, 64.f));
                float16_t* A = P.act[e0 + j];
                for (int q = 0; q < 4; ++q) A[cq[q]] = hv[q];
            }
        }
    }
}

// phase C: 4-row groups [a, b) of the down projection (NVFP4 / MXFP8: stored rows 128 T + 32 q + m; INT4: rows
// 4 gi .. 4 gi + 3); every pick, then each row's sum in slot order
void phase_c(Pool& P, const Job& J, int t)
{
    const Layer& Ly = P.layers[J.L];
    const int fmt = Ly.fmt, rowb = fmt == MXFP8 ? I : I / 2, nc4 = fmt == MXFP8 ? I / 128 : I / 64;
    const bool i4 = fmt == INT4;
    int a, b; split(GROUPS_C, t, P.nt, a, b);
    const float16_t* xs[MAXE];
    const bfloat16_t* xp[MAXE];
    for (int e = 0; e < J.E; ++e) xs[e] = P.act[e];
    for (int pp = 0; pp < J.NP; ++pp) xp[pp] = P.apair[pp];
    float y[MAXE + 1][4];
    for (int gi = a; gi < b; ++gi)
    {
        const int T = gi >> 5, m = gi & 31;
        if (P.prefetch && gi + 1 < b) prefetch_group_c(Ly, J, gi + 1);
        for (int p = 0; p < J.P; ++p)
        {
            const int e = J.pk_e[p], e0 = J.pk_0[p], ne = J.pk_n[p];
            const uint8_t* rp[4];
            float32x4_t f;
            if (i4)
            {
                for (int q = 0; q < 4; ++q) rp[q] = Ly.wd + (size_t) e * Ly.wd_es + (size_t) (4 * gi + q) * (I / 2);
                nv_dot(rp, nullptr, nc4, xp + J.pk_pp[p], (ne + 1) / 2, y + e0, true);     // (may write y[e0 + ne])
                f = vld1q_f32((const float*) ((const uint8_t*) Ly.sd + (size_t) e * Ly.sd_es) + 4 * gi);
            }
            else
            {
                const uint8_t* W = Ly.w2 + (size_t) e * Ly.w2_es;
                for (int q = 0; q < 4; ++q) rp[q] = W + (size_t) (128 * T + 32 * q + m) * rowb;
                const uint8_t* sc = Ly.s2 + (size_t) e * Ly.s2_es + ((size_t) T * nc4 * 32 + m) * 16;
                if (fmt == NVFP4) nv_dot(rp, sc, nc4, xp + J.pk_pp[p], (ne + 1) / 2, y + e0);   // (may write y[e0 + ne])
                else mx_dot(rp, sc, nc4, xs + e0, ne, y + e0);
                f = vdupq_n_f32(Ly.f2[e]);
            }
            for (int j = 0; j < ne; ++j) vst1q_f32(y[e0 + j], vmulq_f32(vld1q_f32(y[e0 + j]), f));
        }
        // output rows of the group's q = 0..3
        int hq[4];
        if (i4) for (int q = 0; q < 4; ++q) hq[q] = 4 * gi + q;
        else { const int hb = 128 * T + 4 * (m & 7) + (m >> 3); for (int q = 0; q < 4; ++q) hq[q] = hb + 32 * q; }
        for (int i0 = 0; i0 < J.nused; ++i0)                  // (rows without a pick were zeroed in phase X)
        {
            const int r = J.used[i0];
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < J.rw_n[r]; ++i) { const int e = J.rw_e[r][i]; for (int q = 0; q < 4; ++q) v[q] += y[e][q]; }
            for (int q = 0; q < 4; ++q) P.slot->out[r][hq[q]] = v[q];
        }
    }
}

void worker(int t)
{
    Pool& P = *g_pool;
    if (P.core0 >= 0 || !P.cores.empty())
    {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P.cores.empty() ? P.core0 + t : P.cores[t], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    Job& J = P.job[t];
    uint32_t* issue = &P.slot->issue;
    uint32_t last = __atomic_load_n(issue, __ATOMIC_ACQUIRE), gen = 0;
    auto now_ns = [] { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec; };
    uint64_t last_job = now_ns(), spins = 0;
    while (!P.stop.load(std::memory_order_relaxed))
    {
        uint32_t s;
        if (P.wait_mode == 1)
        {
            asm volatile("ldaxr %w0, [%1]" : "=r"(s) : "r"(issue) : "memory");
            if (s == last) { asm volatile("wfe" ::: "memory"); continue; }
        }
        else
        {
            s = __atomic_load_n(issue, __ATOMIC_ACQUIRE);
            if (s == last)
            {
                // spin through decode; an idle server backs off by time since the last job: spin < 20 ms, then 20 us
                // sleeps, then 1 ms sleeps after 1 s (Hecate)
                if ((++spins & 1023) == 0)
                {
                    const uint64_t idle = now_ns() - last_job;
                    if (idle > 1000000000ull) { struct timespec ts = {0, 1000000}; nanosleep(&ts, nullptr); continue; }
                    if (idle > 20000000ull) { struct timespec ts = {0, 20000}; nanosleep(&ts, nullptr); continue; }
                }
                asm volatile("yield");
                continue;
            }
        }
        spins = 0; last = s; last_job = now_ns();
        const uint64_t t0 = t == 0 ? ticks() : 0;
        uint32_t errbits = 0;
        parse(P, J, errbits);
        if (!J.ok)
        {
            if (t == 0)
            {
                __atomic_fetch_or(&P.slot->err, errbits, __ATOMIC_RELAXED); ++P.bad;
                __atomic_store_n(&P.slot->done, s, __ATOMIC_RELEASE);
            }
            continue;
        }
        const int fmt = P.layers[J.L].fmt;
        if (P.prefetch && J.P > 0) { int a, b; split(J.P * UNITS_A, t, P.nt, a, b); if (a < b) prefetch_unit_a(P.layers[J.L], J, a); }
        zero_rows(P, J, t);
        if (J.P > 0) prepare_x(P, J, fmt, t);
        barrier(P, t, gen);
        const uint64_t t1 = t == 0 ? ticks() : 0;
        if (J.P > 0) phase_a(P, J, t);
        if (P.prefetch && J.P > 0) { int a, b; split(GROUPS_C, t, P.nt, a, b); if (a < b) prefetch_group_c(P.layers[J.L], J, a); }
        barrier(P, t, gen);
        const uint64_t t2 = t == 0 ? ticks() : 0;
        phase_c(P, J, t);
        gather(P, t, gen);
        if (t == 0)
        {
            const uint64_t t3 = ticks();
            if (errbits) { __atomic_fetch_or(&P.slot->err, errbits, __ATOMIC_RELAXED); ++P.bad; }
            ++P.jobs; P.picks += J.P; P.entries += J.E;
            P.last[0] = (uint64_t) ((t3 - t0) * P.tick_ns); P.last[1] = (uint64_t) ((t1 - t0) * P.tick_ns);
            P.last[2] = (uint64_t) ((t2 - t1) * P.tick_ns); P.last[3] = (uint64_t) ((t3 - t2) * P.tick_ns);
            P.ns_total += P.last[0]; P.ns_x += P.last[1]; P.ns_a += P.last[2]; P.ns_c += P.last[3];
            __atomic_store_n(&P.slot->done, s, __ATOMIC_RELEASE);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// C API (ctypes); pointers are passed as integers
extern "C" {

// slot byte offsets: issue, done, err, layer, rows, ids, gam, x, out, size
void cb_slot_layout(int64_t* o)
{
    o[0] = offsetof(Slot, issue); o[1] = offsetof(Slot, done); o[2] = offsetof(Slot, err); o[3] = offsetof(Slot, layer);
    o[4] = offsetof(Slot, rows); o[5] = offsetof(Slot, ids); o[6] = offsetof(Slot, gam); o[7] = offsetof(Slot, x);
    o[8] = offsetof(Slot, out); o[9] = sizeof(Slot);
}

// one layer's tail: n experts in vLLM's GPU layout (rows shuffled, scales swizzled R128c4), expert e of each tensor at
// base + e * stride bytes; amax13 / amax2: n floats (NVFP4) or null (MXFP8); denom: the outer scale denominator (2688)
int cb_add_layer(int layer, int fmt, int n, int64_t w13, int64_t w13_es, int64_t s13, int64_t s13_es, int64_t w2,
                 int64_t w2_es, int64_t s2, int64_t s2_es, int64_t amax13, int64_t amax2, double denom)
{
    if (!g_pool) g_pool = new Pool();
    if (!g_pool->th.empty() || layer < 0 || n <= 0 || n > 32767 || (fmt != NVFP4 && fmt != MXFP8)) return -1;
    if ((int) g_pool->layers.size() <= layer) g_pool->layers.resize(layer + 1);
    Layer& L = g_pool->layers[layer];
    L = Layer();
    L.fmt = fmt; L.n = n;
    L.w13 = (const uint8_t*) w13; L.w13_es = (size_t) w13_es; L.s13 = (const uint8_t*) s13; L.s13_es = (size_t) s13_es;
    L.w2 = (const uint8_t*) w2; L.w2_es = (size_t) w2_es; L.s2 = (const uint8_t*) s2; L.s2_es = (size_t) s2_es;
    L.f13.assign(n, 1.f); L.f2.assign(n, 1.f);
    if (fmt == NVFP4)
        for (int e = 0; e < n; ++e)
        {
            L.f13[e] = (float) ((double) ((const float*) amax13)[e] / denom);
            L.f2[e] = (float) ((double) ((const float*) amax2)[e] / denom);
        }
    return 0;
}

// one layer's tail in Hecate's GPTQ int4 format (tools/quant/gptq_int4.py: uniform int4 q in -8..7 stored as q + 8, one
// fp32 scale per output row, act-order undone before packing), converted to row-major nibbles (tier_split._rowmajor_i4):
// gate / up [n][I][H/2], down [n][H][I/2] (low nibble = even k), scales gate / up [n][I], down [n][H] fp32; expert e
// of each tensor at base + e * stride bytes
int cb_add_layer_i4(int layer, int n, int64_t wg, int64_t wg_es, int64_t wu, int64_t wu_es, int64_t wd, int64_t wd_es,
                    int64_t sg, int64_t sg_es, int64_t su, int64_t su_es, int64_t sd, int64_t sd_es)
{
    if (!g_pool) g_pool = new Pool();
    if (!g_pool->th.empty() || layer < 0 || n <= 0 || n > 32767) return -1;
    if ((int) g_pool->layers.size() <= layer) g_pool->layers.resize(layer + 1);
    Layer& L = g_pool->layers[layer];
    L = Layer();
    L.fmt = INT4; L.n = n;
    L.wg = (const uint8_t*) wg; L.wu = (const uint8_t*) wu; L.wd = (const uint8_t*) wd;
    L.sg = (const float*) sg; L.su = (const float*) su; L.sd = (const float*) sd;
    L.wg_es = (size_t) wg_es; L.wu_es = (size_t) wu_es; L.wd_es = (size_t) wd_es;
    L.sg_es = (size_t) sg_es; L.su_es = (size_t) su_es; L.sd_es = (size_t) sd_es;
    return 0;
}

// per-expert amax update (after an expert was swapped into a tail position; call between jobs)
int cb_set_amax(int layer, int e, double amax13, double amax2, double denom)
{
    if (!g_pool || layer >= (int) g_pool->layers.size()) return -1;
    Layer& L = g_pool->layers[layer];
    if (L.fmt != NVFP4 || e < 0 || e >= L.n) return -1;
    L.f13[e] = (float) (amax13 / denom); L.f2[e] = (float) (amax2 / denom);
    return 0;
}

// start the pool on the slot: threads pinned to cores core0 .. core0 + threads - 1 (core0 < 0: no pinning);
// wait_mode 0 = spin (yield, idle backoff), 1 = LDAXR + WFE (wakes on the slot's line being written or the event stream);
// env CPUBAND_CORES = core list ("8-62,64-71") instead of core0 + t; CPUBAND_PREFETCH = 0 / 1: software prefetch of
// the next unit's / group's whole rows and scale lines into L2 (and of each phase's first unit before its barrier)
int cb_start(int64_t slot, int threads, int core0, int wait_mode)
{
    if (!g_pool) g_pool = new Pool();
    Pool& P = *g_pool;
    if (!P.th.empty() || threads < 1 || threads > NTMAX || !slot) return -1;
    P.slot = (Slot*) slot; P.nt = threads; P.core0 = core0; P.wait_mode = wait_mode;
    { const char* e = getenv("CPUBAND_PREFETCH"); P.prefetch = e ? atoi(e) : 0; }
    { const char* e = getenv("CPUBAND_NV4"); g_nv4 = !e || atoi(e) != 0; }
    P.cores.clear();
    if (const char* e = getenv("CPUBAND_CORES"))             // e.g. "8-62,64-71": the first `threads` of these cores
    {
        for (const char* q = e; *q;)
        {
            char* end; long a = strtol(q, &end, 10), b = a;
            if (end == q) break;
            if (*end == '-') { q = end + 1; b = strtol(q, &end, 10); }
            for (long c = a; c <= b; ++c) P.cores.push_back((int) c);
            q = *end == ',' ? end + 1 : end;
        }
        if ((int) P.cores.size() < threads) return -1;
    }
    P.tick_ns = 1e9 / (double) tick_freq();
    P.bar_sense.store(0); for (int i = 0; i < NTMAX; ++i) P.bar_flag[i].v.store(0);
    for (int t = 0; t < P.nt; ++t) P.th.emplace_back(worker, t);
    return 0;
}

void cb_stop()
{
    if (!g_pool) return;
    g_pool->stop = true;
    for (auto& t : g_pool->th) t.join();
    g_pool->th.clear(); g_pool->stop = false;
}

// jobs, picks, entries, ns total / x / A / C (sums), bad jobs, last job ns total / x / A / C
void cb_stats(int64_t* o)
{
    Pool* P = g_pool;
    if (!P) { for (int i = 0; i < 12; ++i) o[i] = 0; return; }
    o[0] = P->jobs; o[1] = P->picks; o[2] = P->entries; o[3] = P->ns_total; o[4] = P->ns_x; o[5] = P->ns_a; o[6] = P->ns_c;
    o[7] = P->bad; o[8] = P->last[0]; o[9] = P->last[1]; o[10] = P->last[2]; o[11] = P->last[3];
}

// test / CPU-side poster: release ISSUE = seq; wait for DONE == seq (acquire), timeout in ns (returns 0 or -1)
void cb_post(int64_t slot, uint32_t seq) { __atomic_store_n(&((Slot*) slot)->issue, seq, __ATOMIC_RELEASE); }
int cb_wait(int64_t slot, uint32_t seq, int64_t timeout_ns)
{
    Slot* S = (Slot*) slot;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t i = 0;; ++i)
    {
        if (__atomic_load_n(&S->done, __ATOMIC_ACQUIRE) == seq) return 0;
        if ((i & 4095) == 0)
        {
            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
            if ((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec) > timeout_ns) return -1;
        }
        asm volatile("yield");
    }
}
// post + wait, returns the round trip in ns (or -1)
int64_t cb_run(int64_t slot, uint32_t seq, int64_t timeout_ns)
{
    const uint64_t a = ticks();
    cb_post(slot, seq);
    if (cb_wait(slot, seq, timeout_ns)) return -1;
    return (int64_t) ((ticks() - a) * (1e9 / (double) tick_freq()));
}

// test: bf16 bits of SiTU(g, u) * gam for n (multiple of 4) pairs, as the band computes it
void cb_test_situ(int64_t g, int64_t u, int64_t gam, int n, int64_t out)
{
    for (int i = 0; i < n; i += 4)
    {
        const float32x4_t v = v_situ_bf16(vld1q_f32((const float*) g + i), vld1q_f32((const float*) u + i), ((const float*) gam)[i / 4]);
        vst1_u16((uint16_t*) out + i, vshrn_n_u32(vreinterpretq_u32_f32(v), 16));
    }
}

// debug: the activation (bf16(SiTU)) of entry e of the last job of layer format fmt, as 2304 floats in channel order
void cb_last_act(int e, int fmt, int64_t dst)
{
    Pool* P = g_pool;
    if (!P || e < 0 || e >= MAXE) return;
    float* d = (float*) dst;
    if (fmt == MXFP8) { for (int c = 0; c < I; ++c) d[c] = (float) P->act[e][c] / 64.f; return; }
    // NVFP4: entry e is slot (e - first entry of its pick) & 1 of a pair; the last job's parse of thread 0 knows
    const Job& J = P->job[0];
    int pp = -1, s = 0;
    for (int i = 0; i < J.NP; ++i) if (e >= J.pp_e0[i] && e < J.pp_e0[i] + J.pp_n[i]) { pp = i; s = e - J.pp_e0[i]; }
    if (pp < 0) return;
    for (int c = 0; c < I; ++c) { const uint32_t b = (uint32_t) ((const uint16_t*) P->apair[pp])[pair_pos(c, s)] << 16; memcpy(d + c, &b, 4); }
}

}  // extern "C"
