# OLMoE inference: diagnosis and optimization

A reproducible workflow for diagnosing inference latency on a chat application
backed by **OLMoE-1B-7B-Instruct** under **vLLM**, deployed on EKS with the
model in a GPU node pool and the web/API tier in a CPU node pool.

The application was missing its latency target under load. This repo measures
where the time goes, evaluates the levers that move it, and recommends a
ranked plan — without writing a custom kernel.

## The diagnosis flow

```
1. Baseline   — TTFT and TPOT under vLLM defaults, seed pinned
2. Profile    — torch.profiler kernel breakdown for prefill and decode
3. Levers     — FP8, prefix caching, batching, tensor parallelism
4. Verdict    — ranked recommendations with measured tradeoffs
```

No lever is evaluated until the profile justifies looking at it. No kernel
work is undertaken unless the profile shows a kernel-level gap that vLLM's
production kernels aren't already covering.

## What's measured

| Lever | Result | Source |
|---|---|---|
| Baseline TTFT / TPOT | see `results/baseline_*.json` | `scripts/benchmark_latency.py` |
| Kernel-level profile | `fused_moe_kernel` dominates prefill; router is noise | `scripts/profile_prefill.py` |
| FP8 quantization | end-to-end speedup, accuracy preserved on 3 of 4 evals | `scripts/benchmark_latency.py --fp8` |
| FP8 accuracy | MMLU / HellaSwag / ARC within noise; GSM8K trades a bit | `scripts/eval_suite.py` |
| Prefix caching | TTFT collapses on shared system-prompt traffic | `scripts/bench_prefix_caching.py` |
| Batching | MoE-tax knee, then linear throughput scaling above it | `scripts/bench_batching*.py` |
| Tensor parallelism (TP=2) | latency vs throughput tradeoff across two GPUs | `scripts/benchmark_latency.py --tp 2` |
| Custom kernel | not applicable — profile shows no headroom | (verified, not measured) |

## Hardware

The current production target and the reproduction target for this repo:

| Component | Value |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB each) |
| Driver | 580.126.16 |
| CUDA runtime | 13 (pulled in via `uv sync`) |
| Python | 3.12 |

The single-L4 measurements that originally seeded this workflow are
preserved under `results/archive/` as a historical baseline. Numbers cited
in the deck and in `tasks/design.md` come from the Blackwell runs unless
explicitly labeled otherwise.

For lighter-weight reproductions: any NVIDIA GPU with compute capability
≥ 8.9 (L4, L40S, RTX 4090, H100, RTX PRO 6000 Blackwell) and driver 580+
will work. Older cards (T4, A100) likely fail or perform poorly because the
FP8 kernels used here are tagged for sm_89+.

## Setup

This repo uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone <repo-url>
cd moe-inference-optimize
uv sync
```

`uv sync` creates `.venv/` and installs the pinned dependencies. The first
`vllm` invocation also pulls the OLMoE weights (~14 GB) into the
HuggingFace cache.

## Reproducing the results

`main.py` prints the canonical pipeline. The short version:

```bash
# 0. Sanity check — confirm vLLM + OLMoE + GPU work end-to-end
uv run python scripts/smoke_test.py

# 1. Baseline TTFT/TPOT sweep (bf16, single GPU)
uv run python scripts/benchmark_latency.py

# 2. FP8 sweep (same grid, --quantization fp8)
uv run python scripts/benchmark_latency.py --fp8

# 3. Kernel-level profiles
uv run python scripts/profile_prefill.py        # bf16 prefill
uv run python scripts/profile_prefill.py --fp8  # FP8 prefill
uv run python scripts/profile_decode.py         # decode (bf16 + FP8)

# 4. Accuracy evals (lm-evaluation-harness; ~2 hrs)
uv run python scripts/eval_suite.py

# 5. Engine-agnostic HTTP harness (prefix caching + batching)
uv run python -m scripts.bench_compat_check        # harness reconciliation
uv run python -m scripts.bench_prefix_caching      # shared-prefix workload
uv run python -m scripts.bench_batching            # ShareGPT concurrency
uv run python -m scripts.bench_batching_decode     # pure-decode concurrency

# 6. Tensor parallelism across both GPUs
uv run python scripts/benchmark_latency.py --tp 2
```

Every step writes timestamped JSON under `results/`, so reruns don't
clobber prior data.

For the ShareGPT batching workload, fetch the dataset first (gitignored
because of license ambiguity and size):

```bash
mkdir -p data
curl -L -o data/sharegpt.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Regenerating the figures and the deck

All plots under `talk/figures/` are reproducible from the committed JSON:

```bash
uv run python scripts/plots/ttft_vs_length.py
uv run python scripts/plots/tpot_vs_length.py
uv run python scripts/plots/prefix_caching.py
uv run python scripts/plots/batching.py
uv run python scripts/plots/batching_decode_compare.py
```

Then rebuild the deck:

```bash
uv run python scripts/build_deck.py
# → talk/deck.pptx
```

## Repo layout

```
main.py                      # prints the canonical reproduction pipeline
pyproject.toml               # uv-managed dependencies
scripts/
  smoke_test.py              # end-to-end sanity check
  benchmark_latency.py       # TTFT/TPOT grid sweep (bf16, --fp8, --tp)
  profile_prefill.py         # torch.profiler trace, prefill-dominated
  profile_decode.py          # torch.profiler trace, decode-dominated
  eval_suite.py              # lm-evaluation-harness, bf16 vs FP8
  bench_compat_check.py      # harness vs `vllm bench latency` reconciliation
  bench_prefix_caching.py    # prefix caching experiment
  bench_batching.py          # ShareGPT concurrent workload
  bench_batching_decode.py   # pure-decode concurrent workload
  build_deck.py              # talk/deck.pptx generator
  harness/                   # vLLM-server launcher + streaming HTTP client
  plots/                     # matplotlib figure generators (incl. diagrams.py)
results/
  baseline_*.json            # bf16 latency cells
  fp8_*.json                 # FP8 latency cells
  baseline_tp2_*.json        # bf16 TP=2 latency cells
  eval/                      # lm-evaluation-harness outputs
  harness/                   # HTTP-harness outputs (prefix, batching)
  profile/                   # torch.profiler traces + kernel summaries
  archive/                   # historical L4 runs (kept for traceability)
talk/
  deck.pptx                  # the presentation
  numbers.py                 # reads results/ → live numbers for the deck
  figures/                   # embedded plots and diagrams
tasks/
  requirements.md            # user stories
  design.md                  # methodology + lever board
  todo.md                    # progress checklist
```

## Method, in one paragraph

Measure before you optimize. Build a back-of-envelope model from the
hypothesis. Profile to verify. Enumerate the levers. Measure each one
against the same baseline with the same seed. Report tradeoffs, not
decrees. The kernel — if you ever write one — is the easy part; the hard
part is knowing whether you should. For this workload on this engine, the
profile said *don't*.

## License

MIT for the code in this repo. Model weights and datasets referenced have
their own licenses — check the upstream sources before redistributing.
