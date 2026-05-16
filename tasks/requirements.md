# Requirements

## Context

A chat application backed by OLMoE-1B-7B-Instruct on vLLM is missing its
latency target under load. The model runs in a GPU node pool on EKS; the web /
API tier runs in a CPU node pool. Inference latency is the technical blocker.

## User Stories

### US-001: Reproducible baseline
**As a** platform engineer, **I want** a TTFT/TPOT baseline I can rerun on a
clean instance and get the same numbers, **so that** every optimization claim
is measured against a fixed reference.

#### Acceptance Criteria
- [x] vLLM + OLMoE on a single L4, batch=1, seed pinned.
- [x] Sweeps input length ∈ {32, 128, 512, 1024} × output length ∈ {1, 128}.
- [x] Reports mean and p90 with variance tight enough that any speedup ≥2 ms is signal.

### US-002: Kernel-level profile
**As a** platform engineer, **I want** to see which CUDA kernels actually
consume TTFT and TPOT, **so that** I optimize the bottleneck instead of guessing.

#### Acceptance Criteria
- [x] Top kernels by CUDA time for prefill (L=1024) and decode.
- [x] Routing overhead expressed as % of TTFT.
- [x] Trace files openable in Perfetto for ad-hoc analysis.

### US-003: Ranked optimization levers
**As a** platform engineer, **I want** each candidate lever (FP8, prefix
caching, batching, alternative engine, larger GPU) measured against the same
baseline, **so that** the recommendation is data-driven and the tradeoffs
(latency vs throughput, accuracy vs speed) are explicit.

#### Acceptance Criteria
- [x] Each lever has a dedicated bench script and JSON output under `results/`.
- [x] FP8 accuracy verified on MMLU, HellaSwag, ARC-Challenge, GSM8K before
      recommending it for production.
- [x] A single lever board summarizes status, measured delta, and tradeoff.

## Non-Functional Requirements

- All Python work uses a virtual environment managed by `uv`.
- Measurements are reproducible (seed pinned, dependencies pinned in `uv.lock`).
- Runs on NVIDIA GPUs with compute capability ≥ 8.9 (L4, L40S, RTX 4090, H100).
