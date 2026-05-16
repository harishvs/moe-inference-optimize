"""Top-level entry point: list the diagnosis pipeline and how to run each step.

Run individual scripts directly with `uv run python scripts/<name>.py`. This
file just prints the canonical order for someone reproducing the work end to
end. Each step writes timestamped JSON under `results/`, so reruns don't
clobber prior data.
"""
from __future__ import annotations

PIPELINE = [
    ("0. Smoke test",
     "scripts/smoke_test.py",
     "Confirm vLLM + OLMoE + GPU are wired up. ~60s on first run."),
    ("1. Baseline TTFT/TPOT (bf16)",
     "scripts/benchmark_latency.py",
     "Sweep input ∈ {32, 128, 512, 1024} × output ∈ {1, 128}. ~20 min."),
    ("2. FP8 sweep",
     "scripts/benchmark_latency.py --fp8",
     "Same grid with --quantization fp8. ~20 min."),
    ("3. Profile prefill (bf16 + FP8)",
     "scripts/profile_prefill.py [--fp8]",
     "torch.profiler trace at L=1024. Open in Perfetto."),
    ("4. Profile decode",
     "scripts/profile_decode.py",
     "torch.profiler trace at L=1, O=128 (bf16 and FP8 in one run)."),
    ("5. FP8 accuracy",
     "scripts/eval_suite.py",
     "lm-eval-harness on MMLU, GSM8K, HellaSwag, ARC-Challenge. ~2 hours."),
    ("6. Harness sanity check",
     "python -m scripts.bench_compat_check",
     "Confirm HTTP harness numbers match `vllm bench latency`."),
    ("7. Prefix caching",
     "python -m scripts.bench_prefix_caching",
     "TTFT with vs without prefix caching at P ∈ {128, 512, 2048}."),
    ("8. Batching (ShareGPT)",
     "python -m scripts.bench_batching",
     "Concurrency ∈ {1, 4, 16, 32} on real prompts. Needs data/sharegpt.json."),
    ("9. Batching (pure-decode)",
     "python -m scripts.bench_batching_decode",
     "Concurrency ∈ {1, 4, 16, 32, 64, 128} on 1-token prompts."),
    ("10. Plots",
     "scripts/plots/*.py",
     "Regenerate every figure in talk/figures/ from results/."),
    ("11. Build deck",
     "scripts/build_deck.py",
     "Generate talk/deck.pptx from a clean modern layout + live JSON results."),
]


def main() -> None:
    print("OLMoE on L4 — diagnosis & optimization pipeline")
    print("=" * 64)
    for title, cmd, desc in PIPELINE:
        print(f"\n{title}")
        print(f"  $ uv run {cmd}")
        print(f"  {desc}")
    print()


if __name__ == "__main__":
    main()
