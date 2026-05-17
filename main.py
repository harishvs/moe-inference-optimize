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
     "Sweep input ∈ {128, 512, 1024, 1500} × output ∈ {1, 200}. ~20 min."),
    ("2. FP8 sweep",
     "scripts/benchmark_latency.py --fp8",
     "Same grid with --quantization fp8. ~20 min."),
    ("3. TP=2 sweep",
     "scripts/benchmark_latency.py --tp 2",
     "Same grid across both GPUs. ~20 min."),
    ("4. Profile prefill (bf16 + FP8)",
     "scripts/profile_prefill.py [--fp8]",
     "torch.profiler trace at L=1024. Open in Perfetto."),
    ("5. Profile decode",
     "scripts/profile_decode.py",
     "torch.profiler trace at L=1, O=128 (bf16 and FP8 in one run)."),
    ("6. FP8 accuracy",
     "scripts/eval_suite.py",
     "lm-eval-harness on MMLU, GSM8K, HellaSwag, ARC-Challenge. ~2 hours."),
    ("7. Harness sanity check",
     "python -m scripts.bench_compat_check",
     "Confirm HTTP harness numbers match `vllm bench latency`."),
    ("8. Prefix caching (4 combos)",
     "python -m scripts.bench_prefix_caching [--fp8] [--tp 2]",
     "Chat-app shape (L ∈ {128, 512, 1024, 1500}, suffix=64, output=200). "
     "Run all four: bf16, --fp8, --tp 2, and --fp8 --tp 2."),
    ("9. Batching (ShareGPT, concurrent)",
     "python -m scripts.bench_batching",
     "Concurrency ∈ {1, 4, 16, 32} on real prompts. Needs data/sharegpt.json."),
    ("10. Batching (pure-decode)",
     "python -m scripts.bench_batching_decode [--fp8]",
     "Concurrency ∈ {1, 4, 16, 32, 64, 128} on 1-token prompts."),
    ("11. Figures",
     "scripts/plots/diagrams.py && scripts/plots/appendix.py",
     "Regenerate every figure in talk/figures/ from results/."),
    ("12. Build deck",
     "scripts/build_deck.py",
     "Generate talk/deck.pptx from a clean modern layout + live JSON results."),
]


def main() -> None:
    print("OLMoE-1B-7B on Blackwell — diagnosis & optimization pipeline")
    print("=" * 64)
    for title, cmd, desc in PIPELINE:
        print(f"\n{title}")
        print(f"  $ uv run {cmd}")
        print(f"  {desc}")
    print()


if __name__ == "__main__":
    main()
