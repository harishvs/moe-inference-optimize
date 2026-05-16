"""torch.profiler trace of OLMoE prefill at L=1024.

Writes a Chrome-trace JSON plus a kernel summary under `results/profile/`.
Open the trace at https://perfetto.dev to inspect per-kernel timing.

Usage:
    uv run python scripts/profile_prefill.py            # bf16
    uv run python scripts/profile_prefill.py --fp8      # FP8 quantization

Use `scripts/profile_decode.py` for a decode-dominated trace (1-token prompt,
128-token output).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
INPUT_LEN = 1024
SEED = 42
RESULTS_DIR = Path("results")
PROFILE_DIR = RESULTS_DIR / "profile"

VLLM = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fp8", action="store_true",
                        help="Profile with --quantization fp8 (default: bf16).")
    parser.add_argument("--prefix", default=None,
                        help="Run-dir prefix (default: 'fp8' if --fp8 else 'baseline').")
    args = parser.parse_args()

    prefix = args.prefix or ("fp8" if args.fp8 else "baseline")
    quant_flags = ["--quantization", "fp8"] if args.fp8 else []

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = PROFILE_DIR / f"{prefix}_{timestamp}_L{INPUT_LEN}_seed{SEED}"
    run_dir.mkdir()

    subprocess.run(
        [
            VLLM, "bench", "latency",
            "--model", MODEL,
            *quant_flags,
            "--input-len", str(INPUT_LEN),
            "--output-len", "1",
            "--batch-size", "1",
            "--num-iters-warmup", "3",
            "--seed", str(SEED),
            "--profile",
            "--profiler-config.profiler", "torch",
            "--profiler-config.torch_profiler_dir", str(run_dir),
            "--profiler-config.torch_profiler_with_stack", "false",
            "--profiler-config.torch_profiler_record_shapes", "true",
        ],
        check=True,
    )

    print(f"\nProfile trace(s) written to: {run_dir}")


if __name__ == "__main__":
    main()
