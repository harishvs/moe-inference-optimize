"""Diagrams for the deck (matplotlib, no measured data — pure illustration).

Each function writes a PNG into `talk/figures/`. Re-run any time visual
tweaks are needed; build_deck.py picks the latest file off disk.

Style: clean, minimal, white background, single accent color, no gridlines
unless they carry information.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "talk" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette mirrors the deck
INK = "#1F2A37"
SUBTLE = "#555F6E"
ACCENT = "#2E66C2"
WARN = "#C04A2A"
RULE = "#E2E5EA"
TINT = "#EAF1FB"
TINT2 = "#F2F4F7"


def _box(ax, x, y, w, h, label, *, face=TINT, edge=ACCENT, lw=1.5,
         text_color=INK, fontsize=11, bold=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x1, y1, x2, y2, *, color=ACCENT, lw=2.0, label=None):
    ar = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(ar)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label,
                ha="center", va="bottom",
                fontsize=10, color=SUBTLE, style="italic")


# ------------------------------------------------------------
# Slide 2: EKS topology
# ------------------------------------------------------------

def eks_topology() -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # User
    _box(ax, 0.2, 2.1, 1.2, 1.0, "user", face=TINT2, edge=SUBTLE,
         text_color=SUBTLE, fontsize=11)

    # Load balancer
    _box(ax, 1.7, 2.1, 1.6, 1.0, "load\nbalancer", face=TINT2, edge=SUBTLE,
         text_color=SUBTLE, fontsize=10)

    # CPU node pool envelope (taller / wider so labels breathe)
    cpu_pool = FancyBboxPatch((3.7, 0.7), 2.8, 4.0,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.2, edgecolor=SUBTLE,
                              facecolor="white", linestyle="--")
    ax.add_patch(cpu_pool)
    ax.text(5.1, 4.45, "CPU node pool",
            ha="center", va="bottom", fontsize=11, color=SUBTLE,
            fontweight="bold")
    _box(ax, 4.0, 2.85, 2.2, 1.1, "web pod", face=TINT, edge=ACCENT, fontsize=12)
    _box(ax, 4.0, 1.35, 2.2, 1.1, "API pod", face=TINT, edge=ACCENT, fontsize=12)

    # GPU node pool envelope (wider so the 3-line label fits with margin)
    gpu_pool = FancyBboxPatch((6.9, 0.7), 5.9, 4.0,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.6, edgecolor=WARN,
                              facecolor="white")
    ax.add_patch(gpu_pool)
    ax.text(9.85, 4.45, "GPU node pool — latency target missed here",
            ha="center", va="bottom", fontsize=11, color=WARN,
            fontweight="bold")
    _box(ax, 7.3, 1.9, 5.1, 1.7,
         "vLLM\nOLMoE-1B-7B\n2× Blackwell GPUs",
         face="#FBE9E1", edge=WARN, text_color=INK, fontsize=12)

    # Arrows
    _arrow(ax, 1.4, 2.6, 1.7, 2.6)
    _arrow(ax, 3.3, 2.6, 4.0, 3.0, label="HTTP")
    _arrow(ax, 6.2, 1.9, 7.3, 2.3, label="inference call")
    _arrow(ax, 7.3, 3.0, 6.2, 3.2)  # response back

    # Caption
    ax.text(6.5, 0.2,
            "The latency target is missed inside the GPU pool. "
            "CPU tier is fine; the model serving path is the bottleneck.",
            ha="center", va="bottom", fontsize=11, color=SUBTLE, style="italic")

    out = OUT / "diagram_eks_topology.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Slide 3: TTFT vs TPOT timeline
# ------------------------------------------------------------

def ttft_tpot_timeline() -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Prefill block — widened, label shortened so text breathes
    prefill_w = 2.8
    _box(ax, 0.6, 1.4, prefill_w, 1.2, "PREFILL\ninput prompt",
         face=TINT, edge=ACCENT, fontsize=12)

    # Decode tokens (small boxes streaming right)
    n_tok = 8
    tok_w = 0.62
    gap = 0.06
    start = 0.6 + prefill_w + 0.4
    for i in range(n_tok):
        x = start + i * (tok_w + gap)
        _box(ax, x, 1.55, tok_w, 0.9, f"tok{i+1}",
             face=TINT2, edge=SUBTLE, text_color=SUBTLE, fontsize=9)

    # Ellipsis + final
    last_x = start + n_tok * (tok_w + gap)
    ax.text(last_x + 0.15, 2.0, "…", ha="left", va="center",
            fontsize=18, color=SUBTLE)

    # Time axis arrow
    ax.annotate("", xy=(10.7, 0.9), xytext=(0.6, 0.9),
                arrowprops=dict(arrowstyle="-|>", color=SUBTLE, lw=1.4))
    ax.text(10.7, 0.7, "time", ha="right", va="top",
            fontsize=10, color=SUBTLE, style="italic")

    # TTFT bracket (start → end of prefill)
    ttft_x1, ttft_x2 = 0.6, 0.6 + prefill_w + 0.4 + tok_w
    ax.annotate("", xy=(ttft_x2, 3.05), xytext=(ttft_x1, 3.05),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6))
    ax.text((ttft_x1 + ttft_x2) / 2, 3.2, "TTFT — time to first token",
            ha="center", va="bottom", fontsize=12, color=ACCENT,
            fontweight="bold")

    # TPOT bracket (between two adjacent tokens)
    tp_x1 = start + 2 * (tok_w + gap)
    tp_x2 = start + 3 * (tok_w + gap) + tok_w
    ax.annotate("", xy=(tp_x2, 1.0), xytext=(tp_x1, 1.0),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.6))
    ax.text((tp_x1 + tp_x2) / 2, 0.6, "TPOT", ha="center", va="top",
            fontsize=11, color=WARN, fontweight="bold")

    # Caption
    ax.text(5.5, 0.18,
            "TTFT = how long until the user sees anything.   "
            "TPOT = how fast tokens stream after that.",
            ha="center", va="bottom", fontsize=11, color=SUBTLE, style="italic")

    out = OUT / "diagram_ttft_tpot_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Slide 6: prefill kernel breakdown (stacked bar)
# ------------------------------------------------------------

def prefill_kernel_breakdown() -> Path:
    # Numbers from L4 profile (Blackwell shows the same shape).
    # Blackwell torch.profiler at L=1024, bf16, single-user (parsed from
    # results/profile/baseline_20260516_000526_L1024_seed42/profiler_out_0.txt
    # and renormalized to 100% of accounted leaf-kernel GPU time).
    kernels = [
        ("fused_moe_kernel",       68.0,  "#2E66C2"),
        ("attention projections",  18.0,  "#5A8DDF"),
        ("MoE dispatch + combine",  5.0,  "#90B4ED"),
        ("flash_attn",              5.0,  "#B7CFF3"),
        ("other",                   4.0,  "#E2E5EA"),
    ]

    fig, ax = plt.subplots(figsize=(11, 3.4))
    fig.patch.set_facecolor("white")

    INLINE_THRESHOLD = 15.0  # only inline-label segments wider than this

    left = 0.0
    callouts = []  # narrow segments → render labels above with leader lines
    for name, pct, color in kernels:
        ax.barh(0, pct, left=left, color=color, edgecolor="white", linewidth=1.2)
        center = left + pct / 2
        if pct >= INLINE_THRESHOLD:
            ax.text(center, 0,
                    f"{name}\n{pct:.0f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if name == "fused_moe_kernel" else INK)
        else:
            callouts.append((center, name, pct, color))
        left += pct

    # Render narrow-segment labels above the bar with short leader lines.
    # Stagger horizontally to avoid overlap.
    for i, (x_center, name, pct, color) in enumerate(callouts):
        # alternate between two staggered y positions so labels don't collide
        y_label = 0.85 + (0.30 if i % 2 else 0.0)
        ax.annotate(
            f"{name}  {pct:.0f}%",
            xy=(x_center, 0.30),
            xytext=(x_center, y_label),
            ha="center", va="bottom",
            fontsize=10, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=SUBTLE, lw=0.8,
                            shrinkA=0, shrinkB=2),
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 1.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=SUBTLE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="x", colors=SUBTLE, length=0)
    ax.set_title("Where TTFT actually goes — prefill kernel breakdown",
                 loc="left", fontsize=14, color=INK, fontweight="bold",
                 pad=16)

    fig.tight_layout()
    out = OUT / "diagram_prefill_kernels.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def repo_qr(url: str = "https://github.com/harishvs/moe-inference-optimize") -> Path:
    """Generate a QR code pointing at the repo, for the Q&A slide."""
    import qrcode
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=14,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=INK, back_color="white")
    out = OUT / "repo_qr.png"
    img.save(out)
    return out


def main() -> None:
    paths = [
        eks_topology(),
        ttft_tpot_timeline(),
        prefill_kernel_breakdown(),
        repo_qr(),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
