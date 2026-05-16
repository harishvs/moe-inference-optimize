#!/bin/bash
# Chain the rest of the diagnosis pipeline.
# bf16 baseline is assumed already done.
# Stops on first error.
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=/tmp/moe_pipeline_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"

step() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] === $name ==="
    echo "  cmd: $*"
    echo "  log: $log"
    "$@" >"$log" 2>&1
    echo "[$(date +%H:%M:%S)] === $name DONE ==="
}

step fp8_baseline      uv run python scripts/benchmark_latency.py --fp8
step profile_prefill   uv run python scripts/profile_prefill.py
step profile_prefill_f uv run python scripts/profile_prefill.py --fp8
step profile_decode    uv run python scripts/profile_decode.py
step prefix_caching    uv run python -m scripts.bench_prefix_caching
step batching_decode   uv run python -m scripts.bench_batching_decode
step batching_sharegpt uv run python -m scripts.bench_batching
step tp2_baseline      uv run python scripts/benchmark_latency.py --tp 2
step eval_suite        uv run python scripts/eval_suite.py

echo "[$(date +%H:%M:%S)] PIPELINE COMPLETE"
