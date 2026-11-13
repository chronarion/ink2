# ink2: Inkling 2 Small on a DGX Station GB300

> **Initial release, likely buggy.** This is the first public version. It has been tested (see
> [Output quality](#output-quality)), but not under real traffic for long, and it runs a few patches on top of a
> vLLM preview image. Please report problems as issues.

A server for Thinking Machines' [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) that speaks
the OpenAI API and runs on one DGX Station GB300. It uses vLLM's own Inkling 2 code and the official NVFP4 checkpoint,
unchanged. The model is about 390 GB as published and the GB300 has 252 GB of memory, so the experts the model uses
most stay in GPU memory and the rest stay in the Grace CPU's memory, where the GPU reads them in place. While serving,
the set of experts kept in GPU memory follows the text being written.

On Local Inference Lab's standard benchmark it generates about 120 tokens per second for one request, about
365-390 in total for 8 requests and about 490 for 16, and reads prompts at about 6,700 tokens per second.
On 8 or 16 unrelated conversations at once it generates about 240 and 270 tokens per second.

## Requirements

- A DGX Station GB300 with Docker and the NVIDIA Container Toolkit (CDI). Only the GB300 is used; the RTX PRO 6000
  stays free. `ink2.sh` picks the GB300 by its CDI UUID: set `GB` to yours
  (`docker info | grep -A6 'Discovered Devices'` lists them).
- The image `vllm/vllm-openai:inkling-2-small-arm64`.
- The model `thinkingmachines/Inkling-2-Small` (the NVFP4 repository, 393.5 GB) in the Hugging Face cache under
  `/data/hf`, or any directory given as `MODEL`.
- About 220 GB of free CPU memory: the experts that do not fit in GPU memory are pinned there.

## Run

    ./ink2.sh up          # port 8111; the first start compiles kernels and takes ~10 minutes, later ~6
    curl http://127.0.0.1:8111/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "inkling2", "messages": [{"role": "user", "content": "Hello"}]}'
    ./ink2.sh down

The server answers to the names `inkling2`, `inkling-2-small` and `Inkling-2-Small`. Thinking and tool calls use
vLLM's Inkling parsers. `reasoning_effort` takes `none`, `minimal`, `low`, `medium`, `high`, `xhigh` (or a number from 0
to 0.99); without it the model's default applies. Contexts up to 1M tokens are accepted.

## Speed

Local Inference Lab's [standard benchmark](https://github.com/local-inference-lab/llm-inference-bench) (`lil-bench`,
standard/1), generated tokens per second (8 and 16 requests: all requests together):

| context | 1 request | 8 requests | 16 requests |
|---|---:|---:|---:|
| empty | 122.5 | 386.2 | 489.1 |
| 64K tokens | 122.0 | 392.9 | 489.9 |
| 128K tokens | 119.5 | 364.6 | (skipped, see below) |

Prompt reading: 6,708 tokens per second for a 32K-token prompt (first token after 4.8 s), 6,680 for 128K (19.3 s).

Unrelated conversations (`bench/mixed_bench.py`: 52 different prompts of wiki summaries, chat, maths, Chinese and
code, 1,024 tokens each, temperature 1), generated tokens per second:

| expert set | 1 conversation | 8 conversations | 16 conversations |
|---|---:|---:|---:|
| follows the text (default) | 115.4 | 236.9 | 269.9 |
| fixed (`INK2_FOLLOW=0`) | 105.1 | 211.9 | 235.5 |

The benchmark's requests in one cell all share one prompt, and the expert set that follows the text gains the most
there; on unrelated conversations the gain is smaller (the second table).

<details><summary>How it was measured</summary>

lil-bench ran inside the server container with `python3 -m lil_bench --no-upload --no-p2pmark --url
http://127.0.0.1:8111` (`bench/lil_full.sh`), on 2026-11-13. The 16-request cell at 128K is skipped by the benchmark
itself: it needs 2.13M tokens of KV cache and the server has 1,147,326 (the 8 other cells fit). The mixed numbers are
from one boot per configuration, 90 s per cell after 20 s of warm-up. Boots differ by 2-4%.

</details>

## How it works

- **Two places for expert weights.** Each of the 48 MoE layers has 384 experts. The ones the model uses most (by routing
  counts collected on generated text) stay in GPU memory; the rest stay in pinned Grace memory and the GPU reads them in
  place over NVLink-C2C (about 350 GB/s, against about 6,300 GB/s for its own memory). Everything else (attention, the
  router, embeddings) is in GPU memory. vLLM's own kernels compute both groups; the Grace group runs on a second
  stream at the same time as the GPU group.
- **The expert set follows the text.** While serving, routing counts are kept for every expert (inside the combine
  kernel, at no extra cost). Every 8 steps an expert that the current text uses a lot can replace a little-used one in
  GPU memory, but only if it is likely to pay back the copy, judged with error margins on two time scales and with the
  general-purpose counts as a prior. The most generally useful experts are never replaced, and every swap is scored
  afterwards: swaps that do not pay make further swaps rarer. A log line every 256 ticks compares what was read from
  Grace memory with what the starting placement would have read on the same tokens.
- **One sum, one rounding.** Each token's 8 expert outputs are added in fp32 in a fixed order and rounded once,
  whichever group holds an expert, so moving experts never changes the output.
- **Faster prompt reading.** vLLM's preview image sends all prompt attention on this GPU through a slower fallback
  kernel; here it uses the faster kernel, which halves the time to read a 128K-token prompt.

## Output quality

The weights are the official ones, unchanged. Two things differ from vLLM's stock behaviour:

- Moving experts between GPU and Grace memory is exact: prompt log-probabilities are identical bit for bit with the
  expert set fixed, following the text, or swapped at random before every step.
- The faster prompt-attention kernel gives slightly different numbers than the fallback (mean |Δ log-prob| 0.11 per
  token on an 8K-token text, the same as between any two attention kernels here), with the same mean log-likelihood.

Checks run on this release:

- QA 6/6, and everyday prompts read by eye: an essay, an explanation, code, Chinese, a multi-turn chat, a list, a
  summary, maths (`bench/sample_outputs.py`);
- OpenAI API options 25/25 (sampling settings, stop strings, logprobs, n, JSON schema, penalties, logit bias);
- tool calls 2/2, with thinking on and off;
- answers independent of request order: 96/96 with one stream, 191/192 with four;
- 16 prompts of 30K tokens at once, all served (a few greedy answers start thinking despite `reasoning_effort: none`
  when batched);
- GSM8K 195/200 at temperature 1 with 8 requests at once;
- a needle found in 129K- and 513K-token prompts.

## Limitations

- **Initial release.** Expect rough edges; issues are welcome.
- The RTX PRO 6000 is not used yet. The next release moves less-used experts onto it (3-bit) and computes the coldest
  ones on the Grace cores.
- The DSpark draft model that ships with the checkpoint is off by default. With 2 drafts at batch 1 it gives about
  +12% for one request on the benchmark, but it costs the KV room the 64K-context 16-request cell needs. To turn it on:
  `SPEC='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","num_speculative_tokens_per_batch_size":[[1,1,2],[2,4,1],[5,32,0]]}' INK2_PLACEMENT=/ink2engine/placement/gen-count159-s64.json ./ink2.sh up`.
  With it, vLLM refuses requests that set `min_p` or `logit_bias` (a limitation of speculative decoding in this vLLM).
- Prompt reading is limited by streaming the Grace-resident experts once per 4,096-token chunk.
- One GPU only (no tensor parallelism); the patches assume vLLM's preview image named above.

## Credits

- **Inkling 2 Small** by Thinking Machines (Apache 2.0), and the vLLM project's Inkling 2 support (model code,
  kernels, parsers) in the preview image.
- **Keeping the most used experts on the GPU, the rest in CPU memory**: Fiddler (Kamahori et al.,
  [arXiv:2402.07033](https://arxiv.org/abs/2402.07033)); MoE-Infinity (Xue et al.,
  [arXiv:2401.14361](https://arxiv.org/abs/2401.14361)) for activation-aware expert caching.
- **An expert cache that follows the text**: Eliseev & Mazur, [Fast Inference of Mixture-of-Experts Language Models
  with Offloading](https://arxiv.org/abs/2312.17238) (per-layer expert cache); Lin, He & Chen, [In-depth Analysis on
  Caching and Pre-fetching in MoE Offloading](https://arxiv.org/abs/2511.05814) (frequency-based replacement);
  HybriMoE (Zhong et al., [arXiv:2504.05897](https://arxiv.org/abs/2504.05897)) and FineMoE (Yu et al.,
  [arXiv:2502.05370](https://arxiv.org/abs/2502.05370)) for score-based caching under changing traffic.
- **Reading CPU memory in place over NVLink-C2C**: DAK (Lin, Guo & Lin,
  [arXiv:2604.26074](https://arxiv.org/abs/2604.26074)), and NVIDIA's DGX Station development guide (coherency, ATS).
- **Speculative decoding** (the optional DSpark drafter): Leviathan, Kalman & Matias, [Fast Inference from
  Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192); Chen et al., [Accelerating Large Language
  Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318); [DSpark: Confidence-Scheduled
  Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147); draft length per batch
  size as in TurboSpec (Liu et al., [arXiv:2406.14066](https://arxiv.org/abs/2406.14066)).
- **Local Inference Lab**'s llm-inference-bench (lil-bench) for the benchmark protocol.

## Layout

`ink2.sh` starts and stops the server; `ink2/` holds the patches that `sitecustomize.py` installs into vLLM's Inkling 2
model; `placement/` holds the expert placements and the routing counts they were built from; `tools/mkplace.py` builds
a placement from routing counts; `bench/` holds the checks and benchmarks. The placement files are also on Hugging Face
([chronarion/Inkling-2-Small-ink2](https://huggingface.co/chronarion/Inkling-2-Small-ink2)).

<details><summary>Settings (environment variables)</summary>

| variable | default | what it does |
|---|---|---|
| `PORT` | 8111 | server port |
| `GB` | the author's GB300 UUID | CDI UUID of the GB300 |
| `MODEL` | the Inkling-2-Small snapshot under `/data/hf` | model directory |
| `MAXLEN` | 1048576 | maximum context length |
| `MAXSEQS` | 32 | maximum requests decoded together |
| `MAXBT` | 4096 | prompt tokens per step (more needs more KV room) |
| `GPU_MEM_UTIL` | 0.94 | share of GPU memory vLLM may use |
| `SPEC` | (off) | vLLM `--speculative-config` JSON |
| `EXTRA_ARGS` | | more `vllm serve` arguments |
| `INK2_PLACEMENT` | `placement/gen-count167-s64.json` | which experts stay in GPU memory (JSON: per layer the experts most used first, and how many stay) |
| `INK2_FOLLOW` | 1 | the expert set follows the text (0: fixed) |
| `INK2_FOLLOW_PRIOR` | `placement/counts-gen.json` | routing counts behind the placement, the prior of the rule |
| `INK2_FOLLOW_PERIOD`, `_SWAPS`, `_TAU_F`, `_TAU_M`, `_Z`, `_KAPPA`, `_TH`, `_TM`, `_MARGIN`, `_PMAX`, `_GRACE`, `_GRACE_OUT`, `_COST_NV`, `_COST_MX` | 8, 4, 128, 1024, 2.0, 128, 512, 4096, 2, 128, 128, 256, 5.8, 4.1 | the rule's parameters (see `ink2/follow.py`) |
| `INK2_PARALLEL` | 1 | Grace-resident experts on a second stream (calls up to `INK2_PAR_MAX` = 512 tokens) |
| `INK2_FA4` | all | prompt attention kernel: `all` (fast kernel everywhere), `window` (sliding-window layers only), `none` (vLLM's choice) |
| `INK2_COUNT` | (off) | path: write routing counts of generated tokens there every 30 s (to build a placement) |

</details>

<details><summary>Files</summary>

| file | what it is |
|---|---|
| `ink2.sh` | start / stop / status / logs of the server container |
| `sitecustomize.py` | installs the patches when vLLM imports its Inkling 2 model |
| `ink2/split.py` | the MoE layer: experts split between GPU and Grace memory, the second stream, routing counts |
| `ink2/combine2.py` | one combine kernel over both groups (and the counts) |
| `ink2/follow.py` | the expert set that follows the text: rates, the swap rule, the ledger |
| `ink2/attn.py` | the prompt-attention kernel choice |
| `tools/mkplace.py` | a placement from routing counts and a GPU memory budget |
| `placement/gen-count167-s64.json` | the default placement (167 GiB of experts in GPU memory) |
| `placement/gen-count159-s64.json` | a smaller one, for DSpark |
| `placement/counts-gen.json` | routing counts of 267K generated tokens (a general corpus) |
| `bench/qa.py`, `api_check.py`, `tools_check.py`, `order_check.py`, `burst_check.py`, `gsm8k_acc.py`, `needle.py`, `longcheck.py`, `logprob_check.py` | correctness checks |
| `bench/mixed_bench.py`, `quick.sh`, `lil_full.sh`, `route_corpus.py` | benchmarks and the routing-count corpus driver |

`bench/*` scripts that read text take a corpus JSON (`INK2_CORPUS`: `{"wikitext_test": [...], "no_robots_test":
[...], "gsm8k_test": [{"q", "a"}], "zhwiki": [{"text"}]}` built from WikiText-2, HuggingFaceH4/no_robots, GSM8K and
Chinese Wikipedia) and a directory of Python source files (`INK2_CODE_DIR`).

</details>

## License

MIT (this code). Inkling 2 Small is released by Thinking Machines under Apache 2.0.
