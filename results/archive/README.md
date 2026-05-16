# Archived results

Older runs kept for traceability but not cited by the deck or plots.
The currently-deck-referenced runs all live one directory up.

## L4 baseline runs (top level)

- `baseline_20260502_232641_*.json` — first pass without a fixed seed.
- `baseline_20260502_234554_*.json` — second pass, seed 42 but missing
  the `_O{output_len}` filename suffix from a now-fixed bug in the bench
  script.
- `baseline_20260503_051642_*.json` — L4 single-GPU baseline (Blackwell
  re-baseline at `../baseline_20260515_*` superseded these).
- `fp8_20260503_055736_*.json` — L4 FP8 baseline.

## Older prefix-caching runs (`harness/`)

Earlier prefix-caching harness runs used a different grid shape: keyed
by **prefix length P** with a 64-token suffix and `max_tokens=1` (TTFT
only). The current verdict slides use the **input-length L** grid with
`max_tokens=200` so TPOT is also measured.

- `harness/prefix_caching/` — L4 + Blackwell P-keyed runs
  (`summary_20260503_190501.json`, `summary_20260516_000730.json`).
- `harness/prefix_caching_fp8/` — Blackwell FP8 P-keyed
  (`summary_20260516_171450.json`).
- `harness/prefix_caching_fp8_tp2/` — Blackwell FP8 + TP=2 P-keyed
  (`summary_20260516_173131.json`).
- `harness/prefix_caching_tp2/` — Blackwell TP=2 P-keyed only; no
  L-grid rerun was needed because TP=2 + caching alone isn't a
  recommended config and isn't cited in the verdict slides.

The canonical caching runs (referenced by `talk.numbers` and the
verdict slides) are at `../harness/prefix_caching*/summary_20260516_17*.json`.
