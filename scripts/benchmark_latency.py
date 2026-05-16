"""Sweep TTFT/TPOT for OLMoE on vLLM across a (input_len, output_len) grid.

Wraps `vllm bench latency`. One JSON per cell under `results/`, timestamped
so reruns don't clobber prior data.

Usage:
    uv run python scripts/benchmark_latency.py            # bf16 baseline (TP=1)
    uv run python scripts/benchmark_latency.py --fp8      # FP8 quantization
    uv run python scripts/benchmark_latency.py --tp 2     # tensor-parallel across 2 GPUs

TPOT is derived from the two output-length cells per input length:
    TPOT ≈ (mean_latency(L, O_long) − mean_latency(L, O=1)) / (O_long − 1)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
# Chat-app skewed grid: typical-system-prompt floor through p90 input.
# Output 1 isolates TTFT; output 200 is the assistant-turn-typical decode.
INPUT_LENS = [128, 512, 1024, 1500]
OUTPUT_LENS = [1, 200]
SEED = 42
RESULTS_DIR = Path("results")

VLLM = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fp8", action="store_true",
                        help="Run with --quantization fp8 (default: bf16).")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor-parallel size (default: 1).")
    parser.add_argument("--prefix", default=None,
                        help="Filename prefix (default reflects --fp8 / --tp).")
    args = parser.parse_args()

    default_prefix = "fp8" if args.fp8 else "baseline"
    if args.tp > 1:
        default_prefix = f"{default_prefix}_tp{args.tp}"
    prefix = args.prefix or default_prefix
    quant_flags = ["--quantization", "fp8"] if args.fp8 else []
    tp_flags = ["--tensor-parallel-size", str(args.tp)] if args.tp > 1 else []

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for input_len in INPUT_LENS:
        for output_len in OUTPUT_LENS:
            output_json = (
                RESULTS_DIR
                / f"{prefix}_{timestamp}_L{input_len}_O{output_len}_seed{SEED}.json"
            )
            subprocess.run(
                [
                    VLLM, "bench", "latency",
                    "--model", MODEL,
                    *quant_flags,
                    *tp_flags,
                    "--input-len", str(input_len),
                    "--output-len", str(output_len),
                    "--batch-size", "1",
                    "--num-iters-warmup", "3",
                    "--num-iters", "30",
                    "--seed", str(SEED),
                    "--output-json", str(output_json),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
