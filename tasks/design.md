# Design: profiling and optimizing OLMoE inference for a chat application

## The question we're answering

The chat application — an EKS-deployed product with the model in a GPU node
pool and the web/API tier in a CPU node pool — is missing its inference
latency target under load. The model is `OLMoE-1B-7B-Instruct` served by vLLM.
Why is the latency where it is, and which interventions actually move it?

This document is the measurement pipeline, the analysis framework, and the
set of optimization levers we evaluate to answer that question with data.

## Methodology

```
1. Baseline:  measure prefill (TTFT) and decode (TPOT) under vLLM defaults
2. Profile:   torch.profiler on the target shape to see kernel breakdown
3. Levers:    apply one optimization at a time, re-measure, attribute the delta
4. Verdict:   compare measured deltas to expected upside; rank the levers
```

No lever is evaluated until the profile justifies looking at it. No kernel
work happens unless the profile shows a kernel-level opportunity that vLLM's
production kernels aren't already covering.

## Components

### 1. Benchmark harness (`scripts/benchmark_latency.py`)
Drives `vllm bench latency` across a (input_len, output_len) grid, with
optional `--fp8` and `--tp N` flags.

- **Input lengths:** 128, 512, 1024, 1500 — chat-app-skewed, from the
  typical system-prompt floor through p90 user input.
- **Output lengths:** 1 (isolates TTFT) and 200 (assistant-turn typical decode).
- **30 timed iterations per cell**, 3 warmup iterations discarded.
- **Seed pinned** so the synthetic token sequences are identical run to run.
- **Output:** per-run JSON under `results/` with percentile latencies.

Decode per-token (TPOT) is derived from the two output-length runs per
input length:
`TPOT ≈ (mean_latency(L, O=200) − mean_latency(L, O=1)) / 199`

### 2. Profilers (`scripts/profile_prefill.py`, `scripts/profile_decode.py`)
Drive `vllm bench latency` with `--profile --profiler-config.profiler=torch`,
producing a Chrome-trace file plus a per-kernel summary by CUDA time.

- **Prefill profile:** `input-len=1024, output-len=1, batch-size=1, seed=42`
  — the cell where prefill dominates without prompt-length being so long
  that attention's N² term takes over.
- **Decode profile:** `input-len=1, output-len=128, batch-size=1, seed=42`
  — the cell where decode dominates.

### 3. Optimization levers

Each lever gets a dedicated benchmark run with otherwise-identical
configuration. The name and status are recorded as they are attempted.

| Lever | Expected magnitude | Measured delta | Status |
|---|---|---|---|
| vLLM defaults (baseline) | — | — | measured |
| FP8 quantization | 20–40% on bandwidth-bound paths | — | measured (Blackwell) |
| Prefix caching | 0–90% (workload-dependent) | — | measured |
| Batching | 5–20× throughput at the cost of per-request TPOT | — | measured |
| Tensor parallelism (TP=2) | latency/throughput tradeoff | — | measured (Blackwell-only lever) |
| Custom kernel | n/a — profile shows no headroom | — | not pursued |

Each lever's measurement goes next to the baseline in `results/` with a
descriptive filename prefix (`baseline_*`, `fp8_*`, `baseline_tp2_*`,
`harness/prefix_caching/*`, `harness/batching*`).

The L4 single-GPU runs that originally seeded this study live under
`results/archive/` for traceability.

## What we are *not* doing, and why

We are not writing a custom kernel for MoE routing. The L4 profile at
`input-len=1024` showed `fused_moe_kernel` at ~80% of TTFT and the router
at ~0.09% (77 µs). vLLM's production CUDA kernels (grouped GEMM on tensor
cores via CUTLASS-style implementations) already occupy the hot path. A
solo custom kernel cannot close the tensor-core gap in the time budget of
this project and would deliver no measurable TTFT improvement even if it
matched vLLM's performance on its targeted op. The Blackwell re-baseline
keeps this conclusion under review — if `fused_moe_kernel` ceases to
dominate on Blackwell, the verdict is revisited.

The technical conclusion is that for this workload on this engine, the
levers that move performance are configuration (prefix caching, batching),
quantization (FP8), and hardware sharding (TP=2) — not custom kernels.

## Data flow (MoE layer, for reference)

```
hidden_states (num_tokens, 2048)
  → router linear (2048 → 64) → logits (num_tokens, 64)
  → softmax → top-K (K=8) → expert assignments
  → dispatch (permute tokens to experts)
  → grouped expert GEMM (where ~80% of FFN time lives)
  → combine (weighted sum back to original positions)
```

This is kept in the design doc because the FLOP analysis in the talk
references these shapes. It is not the target of optimization.
