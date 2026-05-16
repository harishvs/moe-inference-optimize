# TODO

## Phase 1 — Diagnosis

- [x] Environment setup (uv venv, vLLM + deps, verify GPU).
- [x] Smoke test (load OLMoE via vLLM, generate one token end-to-end).
- [x] Baseline TTFT/TPOT sweep (`scripts/benchmark_latency.py`, bf16).
- [x] Profile prefill (`scripts/profile_prefill.py`) and decode (`scripts/profile_decode.py`).
- [x] Document top kernels and routing overhead %.

## Phase 2 — Levers

- [x] FP8 quantization sweep (same grid, `--fp8`).
- [x] FP8 accuracy eval on MMLU, HellaSwag, ARC-Challenge, GSM8K.
- [x] Prefix caching benefit at P ∈ {128, 512, 2048}.
- [x] Batching sweep (ShareGPT and pure-decode workloads).
- [x] TRT-LLM applicability check (verified: OLMoE not in 1.2 registry).

## Phase 3 — Decision

- [x] Lever board (`tasks/design.md`).
- [x] Ranked recommendations + tradeoffs.
- [x] Verdict on custom kernels: profile shows no headroom on the hot path.

## Review

The original plan included a Phase 2 to write a fused softmax + top-K Numba
kernel for the MoE router. The profile killed that plan: the router is 0.09%
of TTFT (77 µs), so even a perfect 10× kernel saves 0.08% of the budget.
That negative result is the most important finding in this work — it
redirected effort to FP8 (-32% end-to-end), prefix caching (-67% TTFT on
shared prefixes), and batch sizing (knee at B=32 for MoE on L4), which are
the actual levers.
