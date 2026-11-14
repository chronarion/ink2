# ink2: Inkling 2 Small on a DGX Station GB300

> **Early release, likely buggy.** This is the third public version. It has been tested (see
> [Output quality](#output-quality)), but not under real traffic for long, and it runs a few patches on top of a
> vLLM preview image. Please report problems as issues.

A server for Thinking Machines' [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) that speaks
the OpenAI API and runs on one DGX Station GB300. It uses vLLM's own Inkling 2 code. The model is about 390 GB as
published and the GB300 has 252 GB of memory, so the experts the model uses most stay in GPU memory and the rest are
spread over the rest of the machine. There are two ways to run it:

- **fast** (recommended): the next most used experts run on the station's second GPU, the RTX PRO 6000, as 3-bit
  copies; the experts of layers 40-47 are recalibrated to 4 bits so more of them fit in GPU memory; the least used
  stay in the Grace CPU's memory, where the GPU reads them in place. It generates about 185 tokens per second for
  one request, about 715 in total for 8 requests and about 940 for 16, and reads prompts at about
  7,900 tokens per second. Its next-token probabilities stay close to the 8-bit release of the model (see
  [Output quality](#output-quality)). It needs both GPUs and 140 GB of extra weights from Hugging Face.
- **standard**: the official NVFP4 weights, unchanged, on the GB300 alone (the first release): about 120 tokens per
  second for one request, 365-390 for 8 and 490 for 16.

While serving, the set of experts kept in GPU memory follows the text being written.

## Requirements

- A DGX Station GB300 with Docker and the NVIDIA Container Toolkit (CDI). `ink2.sh` picks the GPUs by their CDI UUIDs:
  set `GB` (and for the fast profile `RTX`) to yours (`docker info | grep -A6 'Discovered Devices'` lists them).
- The image `vllm/vllm-openai:inkling-2-small-arm64`.
- The model `thinkingmachines/Inkling-2-Small` (the NVFP4 repository, 393.5 GB) in the Hugging Face cache under
  `/data/hf`, or any directory given as `MODEL`.
- About 220 GB of free CPU memory: the experts that do not fit in GPU memory are pinned there.
- For the fast profile: the `fast` folder of
  [chronarion/Inkling-2-Small-ink2](https://huggingface.co/chronarion/Inkling-2-Small-ink2) (140 GB):
  `HF_HOME=/data/hf hf download chronarion/Inkling-2-Small-ink2 --include "fast/*"`.

## Run

    PROFILE=fast ./ink2.sh up     # port 8111; the first start compiles kernels and takes ~10 minutes, later ~7
    ./ink2.sh up                  # the standard profile
    curl http://127.0.0.1:8111/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "inkling2", "messages": [{"role": "user", "content": "Hello"}]}'
    ./ink2.sh down

The fast profile finds the downloaded folder in the Hugging Face cache under `/data/hf`; elsewhere, set `FAST_DIR`.
The server answers to the names `inkling2`, `inkling-2-small` and `Inkling-2-Small`. Thinking and tool calls use
vLLM's Inkling parsers. `reasoning_effort` takes `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (or a number from 0
to 0.99); without it the model's default applies. Contexts up to 1M tokens are accepted.

## Speed

Local Inference Lab's [standard benchmark](https://github.com/local-inference-lab/llm-inference-bench) (`lil-bench`,
standard/1), generated tokens per second (8 and 16 requests: all requests together):

| context | profile | 1 request | 8 requests | 16 requests |
|---|---|---:|---:|---:|
| empty | fast | 186.4 | 716.9 | 941.3 |
| | standard | 122.5 | 386.2 | 489.1 |
| 64K tokens | fast | 161.9 | 631.5 | 829.0 |
| | standard | 122.0 | 392.9 | 489.9 |
| 128K tokens | fast | 160.5 | 564.1 | (skipped, see below) |
| | standard | 119.5 | 364.6 | (skipped) |

Prompt reading: 7,940 tokens per second for a 32K-token prompt (first token after 4.1 s) and 7,773 for 128K (16.6 s)
with the fast profile; 6,708 and 6,680 (4.8 s and 19.3 s) with the standard profile.

Unrelated conversations (`bench/mixed_bench.py`: 52 different prompts of wiki summaries, chat, maths, Chinese and
code, 1,024 tokens each, temperature 1), generated tokens per second:

| profile | 1 conversation | 8 conversations | 16 conversations |
|---|---:|---:|---:|
| fast | 218.2 | 651.8 | 883.9 |
| standard | 115.4 | 236.9 | 269.9 |

<details><summary>How it was measured</summary>

lil-bench ran inside the server container with `python3 -m lil_bench --no-upload --no-p2pmark --url
http://127.0.0.1:8111` (`bench/lil_full.sh`); the fast profile on 2026-11-14, the standard profile on 2026-11-13. The
16-request cell at 128K is skipped by the benchmark itself: it needs 2.13M tokens of KV cache and the server has
1,111,269 (fast) or 1,147,326 (standard). The mixed numbers are from one boot per profile, 90 s per cell after 20 s of
warm-up. Boots differ by 2-4%.

</details>

## How it works

- **Three places for expert weights (fast profile).** Each of the 48 MoE layers has 384 experts. The ones the model
  uses most (by routing counts collected on generated text) stay in GPU memory in their 4-bit form (NVFP4). The next
  most used, about 125-140 per layer, run on the RTX PRO 6000 as 3-bit copies. The rest stay in pinned Grace memory and the
  GB300 reads them in place over NVLink-C2C. For every decode step, the rows routed to the RTX's experts go to it
  through pinned host memory; the RTX computes them while the GB300 computes its own experts and the Grace-resident
  ones; one combine adds the three. One kernel routes each token once for all three places (the router, its top 8
  experts, the GPU parts' layouts and the RTX's hand-off). Prompts (long calls) do not use the RTX: they read the
  Grace-resident copies of those experts instead.
- **The 3-bit copies.** They follow brandonmusic's TR3v4 recipe for 3-bit trellis quantization (the EXL3 format of
  ExLlamaV3, based on QTIP): randomized Hadamard rotations, per-expert channel orderings and scales, error-feedback
  rounding against each expert's own captured inputs, with the down projection calibrated on the quantized gate and
  up outputs. They were made from the model's 8-bit (MXFP8) release, with the earlier layers' 3-bit copies active
  while later layers were captured.
- **Layers 40-47 at 4 bits.** The official checkpoint keeps the experts of its last 8 layers in 8 bits (MXFP8), as
  NVIDIA's NVFP4 training recipe does. The fast profile requantizes them to NVFP4 from the 8-bit release with GPTQ,
  calibrated on the model's own activations, choosing each 16-value block's scale with the error of the columns still
  to come in view. That halves their size and frees 18.7 GiB of GPU memory, which holds more experts and, by default,
  the draft model below.
- **Drafting (fast profile).** The checkpoint ships a small draft model (DSpark) that proposes the next tokens; the
  full model checks them all in one step. Here it proposes 3 tokens for one request, 1 for 2-16 requests and none
  from 17. The RTX computes the extra rows of a checking step at almost no extra cost, so drafting pays much more than
  in the first release: about +60% for one request and +17% for eight (in the first release +12% for one request and
  a loss for eight).
- **The expert set follows the text.** While serving, routing counts are kept for every expert (inside the combine
  kernel, at no extra cost). Every 8 steps an expert that the current text uses a lot can replace a little-used one in
  GPU memory, but only if it is likely to pay back the copy, judged with error margins on two time scales and with the
  general-purpose counts as a prior. The most generally useful experts are never replaced, and every swap is scored
  afterwards: swaps that do not pay make further swaps rarer. In the fast profile, the RTX's experts stay where they
  are; swaps happen between GPU memory and Grace memory.
- **One sum, one rounding.** Each token's 8 expert outputs are added in fp32 in a fixed order and rounded once,
  whichever place holds an expert, so moving experts between GPU and Grace memory never changes the output. The RTX's
  kernel splits each expert's work across many warps and adds their partial sums as 64-bit integers (fixed point),
  whose order does not matter, so the same call gives the same result bit for bit.
- **Faster prompt reading.** vLLM's preview image sends all prompt attention on this GPU through a slower fallback
  kernel; here it uses the faster kernel, which halves the time to read a 128K-token prompt.

## Output quality

The reference is the model's 8-bit release
([thinkingmachines/Inkling-2-Small-MXFP8](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8)). Mean KL
divergence of the next-token distribution from it, and how often both pick the same most likely token, over 36,848
positions in 18 windows of wiki text, code, chat, maths and Chinese (full vocabulary):

| | KL divergence | same top token |
|---|---:|---:|
| standard profile (the official NVFP4 weights) | 0.0188 | 96.02% |
| fast profile | 0.0207 | 95.97% |
| the RTX's 3-bit copies, official weights everywhere else | 0.0183 | 96.11% |

The 3-bit copies on the RTX compute with 16-bit activations, so they lose less than the official 4-bit weights with
4-bit activations; recalibrating layers 40-47 to 4 bits costs about 0.003. DSpark changes none of this: speculative
sampling keeps the model's own distribution.

Checks run on this release (fast profile):

- QA 6/6, and everyday prompts read by eye: an essay, an explanation, code, Chinese, a multi-turn chat, a list, a
  summary, maths (`bench/sample_outputs.py`);
- OpenAI API options 20/25 (sampling settings, stop strings, logprobs, n, JSON schema, penalties): the five that fail
  set `min_p` or `logit_bias`, which vLLM refuses while speculative decoding (DSpark) is on;
- tool calls 2/2, with thinking on and off;
- answers independent of request order: 92/96 with one stream (every prompt got the same answer in every order; one
  of the 24 always gets an answer the check does not accept) and 175/192 with four (the standard profile: 96/96 and
  191/192): see [Limitations](#limitations);
- 16 prompts of 30K tokens at once, all 16 answered correctly;
- GSM8K 195/200 at temperature 1 with 8 requests at once;
- a needle found at three depths of 129K- and 513K-token prompts (`bench/needle.py --seed 7 --max-tokens 256`: at
  513K the model, with the official weights too, often first says a sentence about the question before the
  passphrase).

The standard profile passed the same checks in the first release. Moving experts between GPU and Grace memory is
exact: prompt log-probabilities are identical bit for bit with the expert set fixed, following the text, or swapped at
random before every step. The faster prompt-attention kernel gives slightly different numbers than the fallback (mean
|Δ log-prob| 0.11 per token on an 8K-token text, the same as between any two attention kernels here), with the same
mean log-likelihood.

## Limitations

- **Early release.** Expect rough edges; issues are welcome.
- The fast profile needs both GPUs of the station and its weights from Hugging Face.
- In the fast profile a request gives the same answer every time it runs alone, but the numbers can change slightly
  with what else runs at the same time: the RTX's kernel divides each expert's work according to the size of the whole
  batch. A greedy answer at a near-tie can then change (the request-order check with four streams: 177/192, the
  standard profile 191/192). The 3-bit copies also shift a few near-ties for good: one of the check's 24 short
  prompts (a countdown from 30 after a short preamble) always gets an answer the check does not accept.
- The fast profile drafts tokens with DSpark (see How it works). While it is on, vLLM refuses requests that set
  `min_p` or `logit_bias`; `SPEC= PROFILE=fast ./ink2.sh up` turns it off and keeps about 580 more experts in GPU
  memory instead (about 123 / 616 / 976 tokens per second for 1 / 8 / 16 requests then).
- Computing the least used experts on the Grace cores (as 4-bit copies) is built but off: on this machine it was
  slower than letting the GB300 read them in place (7% at 8 requests, 16% at 16), because the cores take longer per
  expert than the GPU's read over NVLink-C2C. It stays in the code as an experiment (`INK2_GRACE`).
- Prompt reading is limited by streaming the Grace-resident experts once per 4,096-token chunk.
- One server per station; the patches assume vLLM's preview image named above.

## Credits

- **Inkling 2 Small** by Thinking Machines (Apache 2.0), and the vLLM project's Inkling 2 support (model code,
  kernels, parsers) in the preview image.
- **The 3-bit copies**: brandonmusic's TR3v4 recipe
  ([GLM-5.2-EXL3-TR3v4 model card](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78), MIT),
  implemented here on ExLlamaV3's ([turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT)
  trellis and LDLQ primitives; QTIP (Tseng, Sun, Hou & De Sa, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235)) for
  trellis-coded quantization, QuIP# (Tseng et al., [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) for Hadamard
  incoherence processing, and QuIP (Chee et al., [arXiv:2307.13304](https://arxiv.org/abs/2307.13304)) for LDLQ
  rounding. HOBBIT (Tang et al., [arXiv:2411.01433](https://arxiv.org/abs/2411.01433)) for keeping lower-precision
  copies of less used experts.
- **Layers 40-47 at 4 bits**: GPTQ (Frantar, Ashkboos, Hoefler & Alistarh,
  [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)); NVIDIA, [Pretraining Large Language Models with
  NVFP4](https://arxiv.org/abs/2509.25149) (the format and the recipe's higher-precision last layers); Four Over Six
  (Cook et al., [arXiv:2512.02010](https://arxiv.org/abs/2512.02010)), the activation scale rule vLLM uses for NVFP4,
  which the calibration models.
- **Keeping the most used experts on the GPU, the rest in CPU memory**: Fiddler (Kamahori et al.,
  [arXiv:2402.07033](https://arxiv.org/abs/2402.07033)); MoE-Infinity (Xue et al.,
  [arXiv:2401.14361](https://arxiv.org/abs/2401.14361)) for activation-aware expert caching; HybriMoE (Zhong et al.,
  [arXiv:2504.05897](https://arxiv.org/abs/2504.05897)) for splitting a layer's experts between processors.
- **An expert cache that follows the text**: Eliseev & Mazur, [Fast Inference of Mixture-of-Experts Language Models
  with Offloading](https://arxiv.org/abs/2312.17238) (per-layer expert cache); Lin, He & Chen, [In-depth Analysis on
  Caching and Pre-fetching in MoE Offloading](https://arxiv.org/abs/2511.05814) (frequency-based replacement);
  FineMoE (Yu et al., [arXiv:2502.05370](https://arxiv.org/abs/2502.05370)) for score-based caching under changing
  traffic.
- **Reading CPU memory in place over NVLink-C2C**: DAK (Lin, Guo & Lin,
  [arXiv:2604.26074](https://arxiv.org/abs/2604.26074)), and NVIDIA's DGX Station development guide (coherency, ATS).
- **Speculative decoding** (the DSpark drafter): Leviathan, Kalman & Matias, [Fast Inference from
  Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192); Chen et al., [Accelerating Large Language
  Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318); [DSpark: Confidence-Scheduled
  Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147); draft length per batch
  size as in TurboSpec (Liu et al., [arXiv:2406.14066](https://arxiv.org/abs/2406.14066)).
- **Local Inference Lab**'s llm-inference-bench (lil-bench) for the benchmark protocol.

## Layout

`ink2.sh` starts and stops the server; `ink2/` holds the patches that `sitecustomize.py` installs into vLLM's Inkling 2
model; `rtx/` holds the RTX's 3-bit expert kernel; `placement/` holds the standard profile's expert placements and the
routing counts they were built from; `tools/mkplace.py` builds a placement from routing counts; `bench/` holds the
checks and benchmarks. The fast profile's weights and placement are on Hugging Face
([chronarion/Inkling-2-Small-ink2](https://huggingface.co/chronarion/Inkling-2-Small-ink2)).

<details><summary>Settings (environment variables)</summary>

| variable | default | what it does |
|---|---|---|
| `PROFILE` | standard | `fast`: the RTX band, layers 40-47 at 4 bits and the larger GPU set, from `FAST_DIR` |
| `FAST_DIR` | the `fast` folder of chronarion/Inkling-2-Small-ink2 under `/data/hf` | the fast profile's files (`band/`, `gptq40/`, `placement.json`) |
| `PORT` | 8111 | server port |
| `GB`, `RTX` | the author's UUIDs | CDI UUIDs of the GB300 and the RTX PRO 6000 |
| `MODEL` | the Inkling-2-Small snapshot under `/data/hf` | model directory |
| `MAXLEN` | 1048576 | maximum context length |
| `MAXSEQS` | 32 | maximum requests decoded together |
| `MAXBT` | 4096 | prompt tokens per step (more needs more KV room) |
| `GPU_MEM_UTIL` | 0.94 | share of GPU memory vLLM may use |
| `SPEC` | fast: DSpark (3 / 1 / 0 drafts for 1 / 2-16 / 17+ requests); standard: off | vLLM `--speculative-config` JSON; empty turns drafting off |
| `EXTRA_ARGS` | | more `vllm serve` arguments |
| `INK2_PLACEMENT` | `placement/gen-count167-s64.json` (fast: `placement.json` in `FAST_DIR`) | which experts stay in GPU memory and on the RTX (JSON: per layer the experts most used first, how many stay in GPU memory and, for the fast profile, how many go to the RTX) |
| `INK2_BAND` | (fast: `band/` in `FAST_DIR`) | the RTX's 3-bit expert files |
| `INK2_BAND_MAX`, `INK2_BAND_MIN_T` | 64, 1 | calls of this many tokens use the RTX (longer ones read the Grace-resident copies) |
| `INK2_EXPERT_OVERRIDE` | (fast: `gptq40/` in `FAST_DIR`) | per-layer expert weights that replace the checkpoint's (layers 40-47) |
| `INK2_FOLLOW` | 1 | the expert set follows the text (0: fixed) |
| `INK2_FOLLOW_PRIOR` | `placement/counts-gen.json` | routing counts behind the placement, the prior of the rule |
| `INK2_FOLLOW_PERIOD`, `_SWAPS`, `_TAU_F`, `_TAU_M`, `_Z`, `_KAPPA`, `_TH`, `_TM`, `_MARGIN`, `_PMAX`, `_GRACE`, `_GRACE_OUT`, `_COST_NV`, `_COST_MX` | 8, 4, 128, 1024, 2.0, 128, 512, 4096, 2, 128, 128, 256, 5.8, 4.1 | the rule's parameters (see `ink2/follow.py`) |
| `INK2_PARALLEL` | 1 | Grace-resident experts on a second stream (calls up to `INK2_PAR_MAX` = 512 tokens) |
| `INK2_FA4` | all | prompt attention kernel: `all` (fast kernel everywhere), `window` (sliding-window layers only), `none` (vLLM's choice) |
| `INK2_COUNT` | (off) | path: write routing counts of generated tokens there every 30 s (to build a placement) |
| `INK2_GRACE`, `INK2_GRACE_INT4`, `INK2_GRACE_MIN_T`, `INK2_GRACE_THREADS`, `INK2_GRACE_CORE0`, `INK2_GRACE_TIMEOUT_MS` | (off) | experimental: the least used experts computed on the Grace cores (see `ink2/graceband.py`) |
| `INK2_ROUTE3` | 1 | the one routing kernel for calls of up to 16 tokens (0: a route per place, as before) |
| `INK2_DEBUG` | 0 | 1: checks after every MoE stage outside CUDA graphs (names the stage of a GPU fault) |
| `INK2_PROF_STEPS` | (off) | `start:stop`: cudaProfilerStart / Stop around those decode steps (development) |

</details>

<details><summary>Files</summary>

| file | what it is |
|---|---|
| `ink2.sh` | start / stop / status / logs of the server container |
| `sitecustomize.py` | installs the patches when vLLM imports its Inkling 2 model |
| `ink2/split.py` | the MoE layer: experts split between GPU memory, the RTX and Grace memory, the second stream, routing counts |
| `ink2/band.py` | the RTX's part: its experts, the pinned hand-off slot, the send and stage kernels |
| `ink2/route3.py` | one routing kernel for a decode call: the GPU parts' layouts and the RTX's hand-off from one top 8 |
| `ink2/combine2.py` | one combine kernel over all parts (and the counts) |
| `ink2/follow.py` | the expert set that follows the text: rates, the swap rule, the ledger |
| `ink2/attn.py` | the prompt-attention kernel choice |
| `ink2/graceband.py`, `cpuband/` | experimental: the least used experts on the Grace cores |
| `ink2/debug.py` | `INK2_DEBUG` checks |
| `rtx/moe3i.py`, `rtx/moe3i.cu` | the RTX's 3-bit trellis expert kernel (built at the first start) |
| `inc/` | links to CUDA headers inside the image that its default include path lacks (for the kernel builds) |
| `tools/mkplace.py` | a placement from routing counts and a GPU memory budget |
| `placement/gen-count167-s64.json` | the standard profile's placement (167 GiB of experts in GPU memory) |
| `placement/gen-count159-s64.json` | a smaller one, for DSpark |
| `placement/counts-gen.json` | routing counts of 267K generated tokens (a general corpus) |
| `bench/qa.py`, `api_check.py`, `tools_check.py`, `order_check.py`, `burst_check.py`, `gsm8k_acc.py`, `needle.py`, `longcheck.py`, `logprob_check.py` | correctness checks |
| `bench/mixed_bench.py`, `quick.sh`, `lil_full.sh`, `route_corpus.py` | benchmarks and the routing-count corpus driver |
| `bench/greedy_compare.py` | greedy decode of fixed prompts with logprobs, to compare two builds token for token |
| `tools/trace_summary.py` | kernel time per device and stream from a vLLM torch-profiler trace |

`bench/*` scripts that read text take a corpus JSON (`INK2_CORPUS`: `{"wikitext_test": [...], "no_robots_test":
[...], "gsm8k_test": [{"q", "a"}], "zhwiki": [{"text"}]}` built from WikiText-2, HuggingFaceH4/no_robots, GSM8K and
Chinese Wikipedia) and a directory of Python source files (`INK2_CODE_DIR`).

</details>

## License

MIT (this code). Inkling 2 Small is released by Thinking Machines under Apache 2.0; the fast profile's weights are
derived from it and carry the same license.
