"""Measure the effect of prefix caching on TTFT and TPOT.

For each total input length L in {128, 512, 1024, 1500} (chat-app shape),
build a shared (L − 64)-token prefix and 30 requests each consisting of
(prefix || unique 64-token suffix). Measure with max_tokens=200 to capture
both TTFT and TPOT. Run twice: cache off, cache on. The delta per L is the
caching benefit for that input length.

Workload assumption: chat app with a long shared system prompt; the 64-token
suffix represents the user's turn. As the system prompt grows, caching pays
off more.

Usage:
    uv run python -m scripts.bench_prefix_caching                    # bf16
    uv run python -m scripts.bench_prefix_caching --fp8              # FP8
    uv run python -m scripts.bench_prefix_caching --fp8 --tp 2       # FP8 + TP=2
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from scripts.harness.client import Request
from scripts.harness.run import make_synthetic_prompt, run_experiment

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
SUFFIX_LEN = 64
OUTPUT_TOKENS = 200
N_REQUESTS = 30
WARMUP = 3
INPUT_LENGTHS = [128, 512, 1024, 1500]   # match slide 5 baseline grid
DEFAULT_RESULTS_DIR = Path("results/harness/prefix_caching")
FP8_RESULTS_DIR = Path("results/harness/prefix_caching_fp8")


def build_requests(input_len: int, suffix_len: int, n: int) -> list[Request]:
    """Build n requests of total length `input_len` sharing a common prefix.

    The shared prefix is (input_len - suffix_len) tokens. Each request
    appends a unique suffix. This matches a chat workload where the system
    prompt is shared and the user turn varies.
    """
    prefix_len = max(1, input_len - suffix_len)
    prefix = make_synthetic_prompt(prefix_len, model=MODEL, seed=0)

    reqs: list[Request] = []
    for i in range(n):
        suffix = make_synthetic_prompt(suffix_len, model=MODEL, seed=1000 + i)
        full = prefix + " " + suffix
        reqs.append(Request(prompt=full, max_tokens=OUTPUT_TOKENS,
                            label=f"L{input_len}_iter{i}"))
    return reqs


def run_one(*, cache_on: bool, input_len: int, timestamp: str,
            fp8: bool, tp: int, results_dir: Path):
    label = (
        f"prefix_{'on' if cache_on else 'off'}_L{input_len}_S{SUFFIX_LEN}_"
        f"O{OUTPUT_TOKENS}_{timestamp}"
    )
    engine_args = [
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
    ]
    if fp8:
        engine_args += ["--quantization", "fp8"]
    if tp > 1:
        engine_args += ["--tensor-parallel-size", str(tp)]
    if cache_on:
        engine_args += ["--enable-prefix-caching"]
    else:
        engine_args += ["--no-enable-prefix-caching"]

    reqs = build_requests(input_len, SUFFIX_LEN, N_REQUESTS)
    return run_experiment(
        engine="vllm",
        model=MODEL,
        engine_args=engine_args,
        requests=reqs,
        warmup=WARMUP,
        seed=42,
        label=label,
        results_dir=results_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fp8", action="store_true",
                        help="Combine prefix caching with FP8 quantization.")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor-parallel size (default: 1).")
    args = parser.parse_args()

    suffix_parts = []
    if args.fp8: suffix_parts.append("fp8")
    if args.tp > 1: suffix_parts.append(f"tp{args.tp}")
    suffix = "_".join(suffix_parts) if suffix_parts else None
    results_dir = (
        DEFAULT_RESULTS_DIR.with_name(f"prefix_caching_{suffix}")
        if suffix else DEFAULT_RESULTS_DIR
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary: dict = {}
    for L in INPUT_LENGTHS:
        print(f"\n=== L={L}, cache OFF ===")
        off = run_one(cache_on=False, input_len=L, timestamp=timestamp,
                      fp8=args.fp8, tp=args.tp, results_dir=results_dir)
        print(f"\n=== L={L}, cache ON ===")
        on = run_one(cache_on=True, input_len=L, timestamp=timestamp,
                     fp8=args.fp8, tp=args.tp, results_dir=results_dir)
        summary[L] = {
            "off_ttft_ms": off.ttft_ms,
            "on_ttft_ms": on.ttft_ms,
            "off_tpot_ms": off.tpot_ms,
            "on_tpot_ms": on.tpot_ms,
            "off_total_ms": off.total_ms,
            "on_total_ms": on.total_ms,
            "ttft_delta_pct": 100.0 * (off.ttft_ms["mean"] - on.ttft_ms["mean"]) / off.ttft_ms["mean"],
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}\n")

    print("=" * 88)
    print(f"{'L':>5}  {'TTFT off':>10}  {'TTFT on':>10}  "
          f"{'TPOT off':>10}  {'TPOT on':>10}  {'TTFT saved':>12}")
    print("=" * 88)
    for L, s in summary.items():
        off_t = s["off_ttft_ms"]["mean"]
        on_t = s["on_ttft_ms"]["mean"]
        off_p = s["off_tpot_ms"]["mean"]
        on_p = s["on_tpot_ms"]["mean"]
        print(
            f"{L:>5}  {off_t:>7.1f} ms  {on_t:>7.1f} ms  "
            f"{off_p:>7.2f} ms  {on_p:>7.2f} ms  "
            f"{s['ttft_delta_pct']:>10.1f}%"
        )


if __name__ == "__main__":
    main()
