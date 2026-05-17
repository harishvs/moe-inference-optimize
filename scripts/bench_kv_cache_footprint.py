"""Measure how much KV cache is consumed when prefix caching is on.

Launches a vLLM server with --enable-prefix-caching, polls
`vllm:kv_cache_usage_perc` from the /metrics endpoint at three points:

  1. Before any requests       (cache is empty)
  2. After warming the prefix   (one request that populates the cache)
  3. After repeated cache hits  (subsequent requests reusing the prefix)

For each prefix length P in {128, 512, 1024, 2048}, it reports the
incremental KV cache footprint. The result is a JSON file under
`results/harness/kv_cache_footprint/` that downstream slides can read.

Usage:
    uv run python -m scripts.bench_kv_cache_footprint
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from scripts.harness.client import Request, _send_one
from scripts.harness.run import make_synthetic_prompt
from scripts.harness.server import ServerConfig, serve


def fire_one_request(*, url: str, model: str, request: Request) -> None:
    """Single-shot wrapper around the harness's _send_one for simple use."""
    _send_one(url=url, model=model, req=request, temperature=0.0, seed=42)

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
# Match the input-length grid used elsewhere in the deck (slide 5 baseline,
# verdict tables). Each L is treated as the total prompt: prefix is L − suffix.
INPUT_LENGTHS = [128, 512, 1024, 1500]
SUFFIX_LEN = 64
WARM_REUSES = 5
RESULTS_DIR = Path("results/harness/kv_cache_footprint")


_METRIC_RES = {
    "kv_cache_usage_perc": re.compile(r'^vllm:kv_cache_usage_perc(?:\{[^}]*\})?\s+([0-9eE.+-]+)$', re.MULTILINE),
    "prefix_cache_queries": re.compile(r'^vllm:prefix_cache_queries_total(?:\{[^}]*\})?\s+([0-9eE.+-]+)$', re.MULTILINE),
    "prefix_cache_hits": re.compile(r'^vllm:prefix_cache_hits_total(?:\{[^}]*\})?\s+([0-9eE.+-]+)$', re.MULTILINE),
    "prompt_tokens_cached": re.compile(r'^vllm:prompt_tokens_cached_total(?:\{[^}]*\})?\s+([0-9eE.+-]+)$', re.MULTILINE),
}


def fetch_metrics(metrics_url: str) -> dict[str, float]:
    """Return current values for the metrics we care about.

    Counters are summed across engines; the kv_cache gauge takes the max.
    """
    r = httpx.get(metrics_url, timeout=5.0)
    r.raise_for_status()
    text = r.text
    out = {}
    for name, pat in _METRIC_RES.items():
        matches = [float(m) for m in pat.findall(text)]
        if not matches:
            out[name] = 0.0
        elif name == "kv_cache_usage_perc":
            out[name] = max(matches)
        else:
            out[name] = sum(matches)
    return out


def _settle(url: str) -> None:
    """Wait briefly for /metrics to update after a request lands."""
    time.sleep(0.5)


def measure_prefix(*, url: str, metrics_url: str, input_len: int) -> dict:
    prefix_len = max(1, input_len - SUFFIX_LEN)
    prefix = make_synthetic_prompt(prefix_len, model=MODEL, seed=0)
    suffix = make_synthetic_prompt(SUFFIX_LEN, model=MODEL, seed=1234)
    full_prompt = prefix + " " + suffix

    # 1. Snapshot before any requests
    _settle(url)
    before = fetch_metrics(metrics_url)

    # 2. Fire one request to populate the prefix in the cache
    fire_one_request(
        url=url,
        model=MODEL,
        request=Request(prompt=full_prompt, max_tokens=1, label=f"warm_P{prefix_len}"),
    )
    _settle(url)
    after_warm = fetch_metrics(metrics_url)

    # 3. Fire several reuse requests; cache should hit on the prefix
    for i in range(WARM_REUSES):
        suffix_i = make_synthetic_prompt(SUFFIX_LEN, model=MODEL, seed=2000 + i)
        prompt_i = prefix + " " + suffix_i
        fire_one_request(
            url=url,
            model=MODEL,
            request=Request(prompt=prompt_i, max_tokens=1, label=f"reuse_P{prefix_len}_iter{i}"),
        )
    _settle(url)
    after_reuse = fetch_metrics(metrics_url)

    queries_during_reuse = after_reuse["prefix_cache_queries"] - after_warm["prefix_cache_queries"]
    hits_during_reuse = after_reuse["prefix_cache_hits"] - after_warm["prefix_cache_hits"]
    cached_tokens_added = after_warm["prompt_tokens_cached"] - before["prompt_tokens_cached"]

    return {
        "input_len": input_len,
        "prefix_len": prefix_len,
        "suffix_len": SUFFIX_LEN,
        "n_reuse": WARM_REUSES,
        "cold": before,
        "after_warm": after_warm,
        "after_reuse": after_reuse,
        "kv_cache_pct_after_warm":   after_warm["kv_cache_usage_perc"],
        "kv_cache_pct_after_reuse":  after_reuse["kv_cache_usage_perc"],
        "queries_on_reuse":          queries_during_reuse,
        "hits_on_reuse":             hits_during_reuse,
        "hit_rate_on_reuse":         hits_during_reuse / max(queries_during_reuse, 1),
        "cached_tokens_added_on_warm": cached_tokens_added,
    }


def main() -> None:
    cfg = ServerConfig(
        engine="vllm",
        model=MODEL,
        extra_args=[
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.85",
            "--enable-prefix-caching",
            "--no-enable-chunked-prefill",
        ],
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with serve(cfg) as url:
        metrics_url = f"{url}/metrics"
        # Capture the total KV cache budget from the server log if reported
        # via the /v1/models endpoint or fall back to the gauge denominator.
        # The gauge is normalized to total KV cache, so we don't need an
        # absolute byte count — but we'll record the raw config for context.

        for L in INPUT_LENGTHS:
            print(f"\n=== Measuring KV cache for L={L} (prefix={L-SUFFIX_LEN}, suffix={SUFFIX_LEN}) ===")
            r = measure_prefix(url=url, metrics_url=metrics_url, input_len=L)
            print(f"  kv_cache_usage gauge: warm {r['kv_cache_pct_after_warm']*100:.3f}%  "
                  f"reuse {r['kv_cache_pct_after_reuse']*100:.3f}%")
            print(f"  prompt tokens cached on warm: +{r['cached_tokens_added_on_warm']:.0f}")
            print(f"  prefix-cache hit rate on reuse: "
                  f"{r['hits_on_reuse']:.0f}/{r['queries_on_reuse']:.0f} = "
                  f"{r['hit_rate_on_reuse']*100:.1f}%")
            results.append(r)

    out_path = RESULTS_DIR / f"summary_{timestamp}.json"
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"\n[summary] wrote {out_path}")

    # Compute byte-level estimates for context (first-principles math)
    print("\n=== KV cache footprint summary (first-principles) ===")
    bytes_per_token = 16 * 2 * 16 * 128 * 2  # OLMoE: 16 layers × KV × 16 heads × 128 d_head × 2 B = 131072 = 128 KiB
    print(f"  bytes/token = {bytes_per_token} ({bytes_per_token/1024:.0f} KiB)")
    print(f"  total KV cache budget on g7e.12xlarge: ~545,184 tokens (~73 GiB)\n")
    print(f"{'L':>5}  {'prefix':>7}  {'MiB (math)':>12}  {'% of budget':>13}  {'gauge measured':>18}  {'hit rate':>10}")
    for r in results:
        mib = r["prefix_len"] * bytes_per_token / (1024 * 1024)
        budget_pct = r["prefix_len"] / 545184 * 100
        gauge_pct = r["kv_cache_pct_after_warm"] * 100
        hit_rate = r["hit_rate_on_reuse"] * 100
        print(f"  L={r['input_len']:>3}  {r['prefix_len']:>7d}  {mib:>9.1f} MiB  "
              f"{budget_pct:>10.3f}%  {gauge_pct:>15.3f}%  {hit_rate:>8.1f}%")


if __name__ == "__main__":
    main()
