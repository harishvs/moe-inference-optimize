"""Graph-based appendix figures for the deck.

Each function writes a PNG into `talk/figures/`. Re-run any time the
underlying JSON data changes; build_deck.py picks the latest off disk.

Style: clean, minimal, white background, restrained palette, no gridlines
unless they carry information. Mirrors `scripts/plots/diagrams.py` so the
deck has one visual voice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "talk" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

from talk import numbers  # noqa: E402

INK = "#1F2A37"
SUBTLE = "#555F6E"
ACCENT = "#2E66C2"
ACCENT_LIGHT = "#5A8DDF"
WARN = "#C04A2A"
RULE = "#E2E5EA"
GREEN = "#3F8F5C"


def _style(ax, *, ylabel=None, xlabel=None, title=None):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(RULE)
    ax.tick_params(colors=SUBTLE, length=4)
    ax.yaxis.label.set_color(SUBTLE)
    ax.xaxis.label.set_color(SUBTLE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=SUBTLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color=SUBTLE)
    if title:
        ax.set_title(title, loc="left", fontsize=14, color=INK,
                     fontweight="bold", pad=14)
    ax.grid(axis="y", color=RULE, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


# ------------------------------------------------------------
# Appendix 1: FP8 accuracy
# ------------------------------------------------------------

def fp8_accuracy() -> Path:
    # Pull from the latest paired eval run on disk; fall back to L4 numbers
    # if no eval results are present yet.
    acc = numbers.eval_accuracy()
    label_map = {
        "mmlu": "MMLU",
        "hellaswag": "HellaSwag",
        "arc_challenge": "ARC-C",
        "gsm8k": "GSM8K",
    }
    if acc:
        order = ["mmlu", "hellaswag", "arc_challenge", "gsm8k"]
        tasks = [label_map[t] for t in order if t in acc]
        bf16 = [acc[t]["bf16_pct"] for t in order if t in acc]
        fp8  = [acc[t]["fp8_pct"]  for t in order if t in acc]
    else:
        tasks = ["MMLU", "HellaSwag", "ARC-C", "GSM8K"]
        bf16 = [52.34, 60.62, 49.49, 35.03]
        fp8  = [52.19, 60.25, 49.57, 33.13]

    x = np.arange(len(tasks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - w / 2, bf16, w, label="bf16", color=ACCENT)
    b2 = ax.bar(x + w / 2, fp8,  w, label="FP8",  color=ACCENT_LIGHT)

    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.6,
                    f"{rect.get_height():.1f}%",
                    ha="center", va="bottom",
                    fontsize=10, color=INK)

    # Δ markers above each pair
    for xi, (b, f) in enumerate(zip(bf16, fp8)):
        delta = f - b
        color = WARN if delta < -1.0 else SUBTLE
        ax.text(xi, max(b, f) + 4.5,
                f"Δ {delta:+.2f} pp",
                ha="center", va="bottom",
                fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12, color=INK)
    ax.set_ylim(0, max(max(bf16), max(fp8)) * 1.30)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    # Title reports the largest measured drop dynamically
    biggest_drop = max(b - f for b, f in zip(bf16, fp8))
    biggest_label = tasks[max(range(len(tasks)), key=lambda i: bf16[i] - fp8[i])]
    _style(ax, ylabel="accuracy (%)",
           title=f"FP8 quality: three tasks unchanged within noise; "
                 f"{biggest_label} drops {biggest_drop:.1f} points")

    fig.tight_layout()
    out = OUT / "appendix_fp8_accuracy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Appendix 2: TTFT vs input length, three levers
# ------------------------------------------------------------

def ttft_vs_length() -> Path:
    bf = numbers.derive_tpot(numbers.latency_grid("baseline"))
    fp = numbers.derive_tpot(numbers.latency_grid("fp8"))
    tp = numbers.derive_tpot(numbers.latency_grid("baseline_tp2"))

    Ls = sorted(bf.keys())

    def ttft(grid, L):
        return grid.get(L, {}).get("ttft_mean_ms")

    bf_y = [ttft(bf, L) for L in Ls]
    fp_y = [ttft(fp, L) for L in Ls]
    tp_y = [ttft(tp, L) for L in Ls]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor("white")

    ax.plot(Ls, bf_y, "-o", color=ACCENT,       linewidth=2.4, markersize=8,
            label="default settings (one GPU)")
    if any(v is not None for v in fp_y):
        ax.plot(Ls, fp_y, "-s", color=GREEN,    linewidth=2.4, markersize=8,
                label="lower-precision math (one GPU)")
    if any(v is not None for v in tp_y):
        ax.plot(Ls, tp_y, "-^", color=WARN,     linewidth=2.4, markersize=8,
                label="both GPUs together")

    # Annotate baseline endpoints
    for L, y in zip(Ls, bf_y):
        ax.text(L, y + 1.0, f"{y:.1f}", ha="center", va="bottom",
                fontsize=10, color=ACCENT)

    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls], fontsize=11, color=INK)
    ax.set_ylim(0, max(bf_y) * 1.25)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    _style(ax,
           ylabel="TTFT mean (ms)",
           xlabel="input length (tokens)",
           title="TTFT by approach — single user")

    fig.tight_layout()
    out = OUT / "appendix_ttft_vs_length.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def tpot_vs_length() -> Path:
    bf = numbers.derive_tpot(numbers.latency_grid("baseline"))
    fp = numbers.derive_tpot(numbers.latency_grid("fp8"))
    tp = numbers.derive_tpot(numbers.latency_grid("baseline_tp2"))

    Ls = sorted(bf.keys())

    def tpot(grid, L):
        return grid.get(L, {}).get("tpot_mean_ms")

    bf_y = [tpot(bf, L) for L in Ls]
    fp_y = [tpot(fp, L) for L in Ls]
    tp_y = [tpot(tp, L) for L in Ls]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor("white")

    ax.plot(Ls, bf_y, "-o", color=ACCENT, linewidth=2.4, markersize=8,
            label="default settings (one GPU)")
    if any(v is not None for v in fp_y):
        ax.plot(Ls, fp_y, "-s", color=GREEN, linewidth=2.4, markersize=8,
                label="FP8 (one GPU)")
    if any(v is not None for v in tp_y):
        ax.plot(Ls, tp_y, "-^", color=WARN, linewidth=2.4, markersize=8,
                label="both GPUs together (TP=2)")

    for L, y in zip(Ls, bf_y):
        ax.text(L, y + 0.10, f"{y:.2f}", ha="center", va="bottom",
                fontsize=10, color=ACCENT)

    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls], fontsize=11, color=INK)
    ax.set_ylim(0, max(bf_y) * 1.30)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    _style(ax,
           ylabel="TPOT mean (ms per token)",
           xlabel="input length (tokens)",
           title="TPOT by approach — single user (batch=1)")

    fig.tight_layout()
    out = OUT / "appendix_tpot_vs_length.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Appendix 3: Batching Pareto
# ------------------------------------------------------------

def batching_pareto() -> Path:
    bd = numbers.batching_decode_summary()
    if not bd:
        # Fall back to placeholder, but build_deck handles missing fig file
        return OUT / "appendix_batching_pareto.png"

    Bs = sorted(bd.keys())
    tok = [bd[B]["tok_per_s"] for B in Bs]
    tpot = [bd[B]["tpot_mean_ms"] for B in Bs]

    fig, ax1 = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor("white")

    # Throughput bars on left axis
    bars = ax1.bar(range(len(Bs)), tok, width=0.55,
                   color=ACCENT, label="tokens served per second")
    for rect, v in zip(bars, tok):
        ax1.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() + max(tok) * 0.025,
                 f"{v:,.0f}",
                 ha="center", va="bottom",
                 fontsize=10, color=INK, fontweight="bold")
    ax1.set_xticks(range(len(Bs)))
    ax1.set_xticklabels([str(B) for B in Bs], fontsize=11, color=INK)
    _style(ax1,
           ylabel="tokens served per second",
           xlabel="batch size  (in-flight requests in one vLLM pod)",
           title="Continuous batching: throughput rises, per-request TPOT degrades")
    ax1.set_ylim(0, max(tok) * 1.18)

    # TPOT line on right axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(Bs)), tpot, "-o",
             color=WARN, linewidth=2.4, markersize=8,
             label="TPOT (ms per token)")
    for i, v in enumerate(tpot):
        ax2.text(i, v + max(tpot) * 0.04, f"{v:.1f}",
                 ha="center", va="bottom",
                 fontsize=9, color=WARN)
    for spine in ("top", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(RULE)
    ax2.spines["bottom"].set_color(RULE)
    ax2.tick_params(colors=WARN, length=4)
    ax2.set_ylabel("TPOT (ms per token)", color=WARN, fontsize=12)
    ax2.set_ylim(0, max(tpot) * 1.25)

    # Combined legend
    lines = [bars, ax2.lines[0]]
    labels = ["throughput (tok/s)", "TPOT (ms/tok)"]
    ax1.legend(lines, labels, loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    out = OUT / "appendix_batching_pareto.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def sla_status() -> Path:
    """SLA pass/fail under concurrent load: bf16 vs FP8, with thresholds."""
    import json

    def latest_summary(subdir):
        files = sorted((REPO / "results" / "harness" / subdir).glob("summary_*.json"))
        return json.loads(files[-1].read_text()) if files else {}

    bf = latest_summary("batching_decode")
    fp = latest_summary("batching_decode_fp8")
    if not bf or not fp:
        return OUT / "appendix_sla_status.png"

    Bs = sorted(set(bf) & set(fp), key=int)
    bf_ttft = [bf[b]["ttft_ms"]["p90"] for b in Bs]
    fp_ttft = [fp[b]["ttft_ms"]["p90"] for b in Bs]
    SLA = 500

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor("white")

    xs = list(range(len(Bs)))
    # Shaded fail region
    ax.axhspan(SLA, max(max(bf_ttft), max(fp_ttft)) * 1.2,
               color="#FBE9E1", alpha=0.7, zorder=0)

    # SLA threshold line
    ax.axhline(SLA, color=WARN, linewidth=1.6, linestyle="--", zorder=1)
    ax.text(len(Bs) - 0.5, SLA + 30, f"  SLA: TTFT p90 < {SLA} ms",
            ha="right", va="bottom",
            fontsize=11, color=WARN, fontweight="bold")

    # Lines
    ax.plot(xs, bf_ttft, "-o", color=ACCENT, linewidth=2.4, markersize=10,
            label="bf16 baseline")
    ax.plot(xs, fp_ttft, "-s", color=GREEN, linewidth=2.4, markersize=10,
            label="FP8")

    # Pass/fail markers
    for x, b in zip(xs, Bs):
        for y, color in ((bf_ttft[x], ACCENT), (fp_ttft[x], GREEN)):
            ax.text(x, y - SLA * 0.06,
                    f"{y:.0f}",
                    ha="center", va="top",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in Bs], fontsize=11, color=INK)
    ax.set_ylim(0, max(max(bf_ttft), max(fp_ttft)) * 1.15)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    _style(ax,
           ylabel="TTFT p90 (ms)",
           xlabel="batch size  (in-flight requests in one vLLM pod)",
           title="SLA-passing ceiling: both lines fail past batch=32")

    fig.tight_layout()
    out = OUT / "appendix_sla_status.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Appendix 4: Prefix caching savings
# ------------------------------------------------------------

def prefix_caching() -> Path:
    pc = numbers.prefix_caching_summary()
    if not pc:
        return OUT / "appendix_prefix_caching.png"

    Ps = sorted(pc.keys())
    off = [pc[P]["off_mean_ms"] for P in Ps]
    on  = [pc[P]["on_mean_ms"]  for P in Ps]

    x = np.arange(len(Ps))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - w / 2, off, w, label="cache OFF", color=SUBTLE)
    b2 = ax.bar(x + w / 2, on,  w, label="cache ON",  color=GREEN)

    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(off) * 0.02,
                    f"{rect.get_height():.1f}",
                    ha="center", va="bottom",
                    fontsize=10, color=INK)

    # Savings annotation per P
    for i, P in enumerate(Ps):
        savings = pc[P]["savings_pct"]
        ax.text(i, max(off[i], on[i]) + max(off) * 0.12,
                f"−{savings:.0f}%",
                ha="center", va="bottom",
                fontsize=11, color=GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"P={P}" for P in Ps], fontsize=12, color=INK)
    ax.set_ylim(0, max(off) * 1.45)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    _style(ax,
           ylabel="TTFT mean (ms)",
           title="Prefix caching: payoff scales with shared-prefix length")

    fig.tight_layout()
    out = OUT / "appendix_prefix_caching.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Appendix 5: FLOP math — attention vs FFN at L=1024
# ------------------------------------------------------------

def flop_math() -> Path:
    # Per-layer GFLOPs at L=1024 (numbers from substack write-up / OLMoE shapes).
    labels = ["Attention\nN² term", "MoE FFN\n(8 active experts)"]
    gflops = [8.6, 69.0]
    colors = [SUBTLE, ACCENT]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor("white")

    bars = ax.bar(labels, gflops, color=colors, width=0.55)
    for rect, v in zip(bars, gflops):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 1.5,
                f"{v:.1f} GFLOPs",
                ha="center", va="bottom",
                fontsize=12, color=INK, fontweight="bold")

    # Ratio annotation in clear whitespace (mid-left)
    ratio = gflops[1] / gflops[0]
    ax.annotate(f"{ratio:.0f}× more linear FFN work\nthan quadratic attention work",
                xy=(1.0, gflops[1] * 0.80),
                xytext=(0.18, gflops[1] * 0.55),
                ha="left", va="center",
                fontsize=12, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))

    ax.tick_params(axis="x", labelsize=12, colors=INK, length=0)
    ax.set_ylim(0, max(gflops) * 1.25)
    _style(ax, ylabel="GFLOPs per layer",
           title="Why prefill isn't attention-bound — per-layer FLOPs at L=1024")

    fig.tight_layout()
    out = OUT / "appendix_flop_math.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    paths = [
        fp8_accuracy(),
        ttft_vs_length(),
        tpot_vs_length(),
        batching_pareto(),
        sla_status(),
        prefix_caching(),
        flop_math(),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
