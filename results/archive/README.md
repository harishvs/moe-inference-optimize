# Archived results

Exploratory runs kept for traceability but not cited by the deck or plots.

- `baseline_20260502_232641_*.json` — first pass without a fixed seed.
- `baseline_20260502_234554_*.json` — second pass, seed 42 but missing the
  `_O{output_len}` filename suffix from a now-fixed bug in the bench
  script.

The canonical runs are the seed-42 grid at `baseline_20260503_051642_*.json`
(bf16) and `fp8_20260503_055736_*.json` (FP8) one directory up. Those are
what the deck and `talk/substack.md` reference.
