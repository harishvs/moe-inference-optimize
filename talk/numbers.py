"""Pull the latest measured numbers out of `results/` for the deck.

Each function returns a dict keyed by the cell shape. The deck builder
calls these so a fresh run automatically updates the slides; no manual
number-copying.

Conventions:
- `baseline_*` files = bf16, TP=1
- `fp8_*` files     = FP8, TP=1
- `*_tp2_*` files   = bf16, TP=2 (and `fp8_tp2_*` = FP8, TP=2)
- All lat values are converted to milliseconds.

For each (input_len, output_len) cell, the latest timestamp wins. Older
runs land under `results/archive/` and are not picked up.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

CELL_RE = re.compile(
    r"^(?P<prefix>[a-z0-9_]+?)_(?P<ts>\d{8}_\d{6})_L(?P<L>\d+)_O(?P<O>\d+)_seed\d+\.json$"
)


def _scan(prefix: str) -> dict[tuple[int, int], dict]:
    """Return {(L, O): payload} for the latest timestamp per cell with given prefix."""
    by_cell: dict[tuple[int, int], tuple[str, dict]] = {}
    for path in RESULTS.glob(f"{prefix}_*.json"):
        m = CELL_RE.match(path.name)
        if not m or m.group("prefix") != prefix:
            continue
        ts = m.group("ts")
        L, O = int(m.group("L")), int(m.group("O"))
        prev = by_cell.get((L, O))
        if prev is None or ts > prev[0]:
            by_cell[(L, O)] = (ts, json.loads(path.read_text()))
    return {k: v[1] for k, v in by_cell.items()}


def cell_summary(payload: dict) -> dict:
    """Convert a `vllm bench latency` payload to a {mean,p50,p90,p99} ms dict."""
    return {
        "mean": payload["avg_latency"] * 1000,
        "p50": payload["percentiles"]["50"] * 1000,
        "p90": payload["percentiles"]["90"] * 1000,
        "p99": payload["percentiles"]["99"] * 1000,
    }


def latency_grid(prefix: str = "baseline") -> dict[tuple[int, int], dict]:
    """{(L, O): {mean,p50,p90,p99} in ms} for the named prefix family."""
    raw = _scan(prefix)
    return {k: cell_summary(v) for k, v in raw.items()}


def derive_tpot(grid: dict[tuple[int, int], dict]) -> dict[int, dict]:
    """Per input length, derive TTFT and TPOT from O=1 and the longest O cell."""
    out: dict[int, dict] = {}
    by_L: dict[int, dict[int, dict]] = {}
    for (L, O), s in grid.items():
        by_L.setdefault(L, {})[O] = s
    for L, by_O in by_L.items():
        if 1 not in by_O:
            continue
        long_Os = [o for o in by_O if o > 1]
        if not long_Os:
            continue
        long_O = max(long_Os)
        ttft = by_O[1]["mean"]
        end_to_end = by_O[long_O]["mean"]
        tpot = (end_to_end - ttft) / (long_O - 1)
        out[L] = {
            "ttft_mean_ms": ttft,
            "ttft_p90_ms": by_O[1]["p90"],
            "end_to_end_mean_ms": end_to_end,
            "end_to_end_O": long_O,
            "tpot_mean_ms": tpot,
        }
    return out


def _delta_grid(left: dict[int, dict], right: dict[int, dict]) -> dict[int, dict]:
    """For each L common to both: % delta from `left` to `right` per metric."""
    out: dict[int, dict] = {}
    for L in sorted(set(left) & set(right)):
        out[L] = {
            "lhs": left[L],
            "rhs": right[L],
            "ttft_pct": 100 * (right[L]["ttft_mean_ms"] - left[L]["ttft_mean_ms"]) / left[L]["ttft_mean_ms"],
            "tpot_pct": 100 * (right[L]["tpot_mean_ms"] - left[L]["tpot_mean_ms"]) / left[L]["tpot_mean_ms"],
            "e2e_pct": 100 * (right[L]["end_to_end_mean_ms"] - left[L]["end_to_end_mean_ms"]) / left[L]["end_to_end_mean_ms"],
        }
    return out


def fp8_vs_bf16() -> dict[int, dict]:
    """Per input length: bf16 vs FP8 deltas (Δ% TTFT, Δ% TPOT, Δ% end-to-end)."""
    return _delta_grid(
        derive_tpot(latency_grid("baseline")),
        derive_tpot(latency_grid("fp8")),
    )


def tp2_vs_baseline() -> dict[int, dict]:
    """Per input length: TP=1 vs TP=2 baseline (bf16). Negative pct = TP=2 faster."""
    return _delta_grid(
        derive_tpot(latency_grid("baseline")),
        derive_tpot(latency_grid("baseline_tp2")),
    )


def eval_accuracy() -> Optional[dict]:
    """{task: {bf16, fp8, delta_pp}} for the most recent paired eval run.

    Picks the latest timestamp shared by both the `baseline_<ts>_<task>` and
    `fp8_<ts>_<task>` directories, so partial / mismatched runs are skipped.
    """
    eval_dir = RESULTS / "eval"
    if not eval_dir.exists():
        return None

    # Collect (timestamp, kind, task) tuples
    runs: dict[str, dict[str, dict[str, Path]]] = {}
    for p in eval_dir.iterdir():
        if not p.is_dir():
            continue
        # Pattern: <kind>_<YYYYMMDD_HHMMSS>_<task_name>
        parts = p.name.split("_", 2)
        if len(parts) < 3:
            continue
        kind, ts_date, rest = parts
        if kind not in ("baseline", "fp8"):
            continue
        ts_time, _, task = rest.partition("_")
        if not ts_time:
            continue
        ts = f"{ts_date}_{ts_time}"
        runs.setdefault(ts, {}).setdefault(kind, {})[task] = p

    # Find the latest timestamp where both kinds cover the same task set
    candidates = sorted(
        ts for ts, by_kind in runs.items()
        if "baseline" in by_kind and "fp8" in by_kind
        and set(by_kind["baseline"]) & set(by_kind["fp8"])
    )
    if not candidates:
        return None
    ts = candidates[-1]

    # Headline metric per task
    metric_keys = {
        "mmlu": ("mmlu", "acc,none"),
        "hellaswag": ("hellaswag", "acc,none"),
        "arc_challenge": ("arc_challenge", "acc,none"),
        "gsm8k": ("gsm8k", "exact_match,strict-match"),
    }

    out: dict[str, dict] = {}
    for task, (lookup_key, metric) in metric_keys.items():
        bf_dir = runs[ts]["baseline"].get(task)
        fp_dir = runs[ts]["fp8"].get(task)
        if not bf_dir or not fp_dir:
            continue
        bf_files = list(bf_dir.rglob("results_*.json"))
        fp_files = list(fp_dir.rglob("results_*.json"))
        if not bf_files or not fp_files:
            continue
        bf_data = json.loads(bf_files[0].read_text())["results"].get(lookup_key)
        fp_data = json.loads(fp_files[0].read_text())["results"].get(lookup_key)
        if bf_data is None or fp_data is None:
            continue
        bv = bf_data.get(metric)
        fv = fp_data.get(metric)
        if bv is None or fv is None:
            continue
        out[task] = {
            "bf16_pct": bv * 100,
            "fp8_pct": fv * 100,
            "delta_pp": (fv - bv) * 100,
        }
    return out or None


def latest_summary(subdir: str) -> Optional[dict]:
    """Load the most recent summary_<ts>.json under results/harness/<subdir>."""
    candidates = sorted((RESULTS / "harness" / subdir).glob("summary_*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def prefix_caching_summary() -> Optional[dict]:
    """{ P: {off_mean, on_mean, delta_pct} } for the latest prefix-caching run."""
    raw = latest_summary("prefix_caching")
    if not raw:
        return None
    return {
        int(P): {
            "off_mean_ms": v["off_ttft_ms"]["mean"],
            "on_mean_ms": v["on_ttft_ms"]["mean"],
            "savings_pct": v["delta_pct"],  # source field is signed: positive = caching is faster
        }
        for P, v in raw.items()
    }


def batching_decode_summary() -> Optional[dict]:
    """{ B: {tok_per_s, ttft_p90, tpot_mean} } for the latest pure-decode batching run."""
    raw = latest_summary("batching_decode")
    if not raw:
        return None
    return {
        int(B): {
            "tok_per_s": v["output_tokens_per_s"],
            "req_per_s": v["requests_per_s"],
            "ttft_mean_ms": v["ttft_ms"]["mean"],
            "ttft_p90_ms": v["ttft_ms"]["p90"],
            "tpot_mean_ms": v["tpot_ms"]["mean"],
        }
        for B, v in raw.items()
    }


def batching_sharegpt_summary() -> Optional[dict]:
    """Same shape as batching_decode_summary but for the ShareGPT run."""
    raw = latest_summary("batching")
    if not raw:
        return None
    return {
        int(B): {
            "tok_per_s": v["output_tokens_per_s"],
            "req_per_s": v["requests_per_s"],
            "ttft_mean_ms": v["ttft_ms"]["mean"],
            "ttft_p90_ms": v["ttft_ms"]["p90"],
            "tpot_mean_ms": v["tpot_ms"]["mean"],
        }
        for B, v in raw.items()
    }


def main() -> None:
    """Quick text dump — useful while building the deck."""
    print("=== bf16 baseline (Blackwell, TP=1) ===")
    for L, s in sorted(derive_tpot(latency_grid("baseline")).items()):
        print(f"  L={L:5d}  TTFT={s['ttft_mean_ms']:6.2f} ms  "
              f"TPOT={s['tpot_mean_ms']:5.3f} ms  "
              f"end-to-end @ O={s['end_to_end_O']}={s['end_to_end_mean_ms']:7.2f} ms")

    fp = fp8_vs_bf16()
    if fp:
        print("\n=== FP8 vs bf16 ===")
        for L, d in sorted(fp.items()):
            print(f"  L={L:5d}  TTFT Δ={d['ttft_pct']:+5.1f}%  "
                  f"TPOT Δ={d['tpot_pct']:+5.1f}%  e2e Δ={d['e2e_pct']:+5.1f}%")
    else:
        print("\n=== FP8 grid not yet measured ===")

    tp = tp2_vs_baseline()
    if tp:
        print("\n=== TP=2 vs TP=1 (bf16 baseline) ===")
        for L, d in sorted(tp.items()):
            print(f"  L={L:5d}  TTFT Δ={d['ttft_pct']:+5.1f}%  "
                  f"TPOT Δ={d['tpot_pct']:+5.1f}%  e2e Δ={d['e2e_pct']:+5.1f}%")

    pc = prefix_caching_summary()
    if pc:
        print("\n=== Prefix caching ===")
        for P, v in sorted(pc.items()):
            print(f"  P={P:5d}  off={v['off_mean_ms']:6.1f} ms  "
                  f"on={v['on_mean_ms']:6.1f} ms  Δ={v['savings_pct']:+5.1f}%")

    bd = batching_decode_summary()
    if bd:
        print("\n=== Batching (pure decode) ===")
        for B, v in sorted(bd.items()):
            print(f"  B={B:4d}  {v['tok_per_s']:7.0f} tok/s  "
                  f"TPOT={v['tpot_mean_ms']:5.2f} ms  "
                  f"TTFT p90={v['ttft_p90_ms']:7.1f} ms")


if __name__ == "__main__":
    main()
