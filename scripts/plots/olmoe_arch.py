"""Draw a side-by-side dense-vs-MoE layer diagram for the deck.

Our own take on the architecture so we don't have to attribute an external figure.
Left column: dense transformer layer (like OLMo).
Right column: OLMoE-style MoE layer with router + 64 experts (8 active shown).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "talk" / "figures" / "olmoe_arch_custom.png"

# Colors (matplotlib tab palette, matching the rest of the deck)
COLOR_BG = "white"
COLOR_BLOCK = "#cfe2f3"       # pale blue — structural blocks
COLOR_ATTN = "#d9e1f2"        # slightly different blue for attention
COLOR_FFN = "#fff2cc"         # pale yellow — dense FFN
COLOR_EXPERT_ACTIVE = "#ffe699"   # yellow — active experts
COLOR_EXPERT_INACTIVE = "#e7e6e6"  # grey — inactive experts
COLOR_ROUTER = "#f4cccc"      # pale red — router decision
COLOR_ARROW = "#555555"
COLOR_TEXT = "#222222"


def draw_block(ax, x, y, w, h, label, facecolor, fontsize=11, fontweight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="#333", facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=COLOR_TEXT)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.2, color=COLOR_ARROW):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0,
    ))


def draw_dense(ax, cx):
    """Draw the dense layer centered at x=cx."""
    w = 2.8
    hw = w / 2

    # Input arrow
    arrow(ax, cx, 9.6, cx, 9.1)

    # Attention block
    draw_block(ax, cx - hw, 8.2, w, 0.9, "Attention", COLOR_ATTN, fontsize=12, fontweight="bold")

    # Residual skip + down
    arrow(ax, cx, 8.2, cx, 7.7)

    # Norm (small) — keeping the diagram simple, so skip

    # FFN block
    draw_block(ax, cx - hw, 6.8, w, 0.9, "Feed-Forward (one FFN)",
               COLOR_FFN, fontsize=12, fontweight="bold")

    # Output down
    arrow(ax, cx, 6.8, cx, 6.3)

    # Layer label
    ax.text(cx, 5.9, "× 16 layers", ha="center", va="center",
            fontsize=11, style="italic", color=COLOR_TEXT)

    # Column title
    ax.text(cx, 10.1, "Dense transformer layer\n(e.g. OLMo)",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
            color=COLOR_TEXT)


def draw_moe(ax, cx):
    """Draw the MoE layer centered at x=cx."""
    w = 4.0
    hw = w / 2

    # Input arrow
    arrow(ax, cx, 9.6, cx, 9.1)

    # Attention block
    draw_block(ax, cx - hw, 8.2, w, 0.9, "Attention", COLOR_ATTN,
               fontsize=12, fontweight="bold")

    # Down to router
    arrow(ax, cx, 8.2, cx, 7.9)

    # Router block
    draw_block(ax, cx - 1.3, 7.0, 2.6, 0.8, "Router\n(picks top-8 of 64)",
               COLOR_ROUTER, fontsize=10, fontweight="bold")

    # Expert row (64 small boxes, 8 colored active)
    # Active experts: positions chosen to look plausible (spread-out)
    active = {2, 9, 18, 27, 35, 44, 52, 60}
    n_experts = 64
    row_y = 6.0
    row_h = 0.45
    expert_area_left = cx - hw
    expert_area_w = w
    per_exp_w = expert_area_w / n_experts

    for i in range(n_experts):
        ex_x = expert_area_left + i * per_exp_w
        fc = COLOR_EXPERT_ACTIVE if i in active else COLOR_EXPERT_INACTIVE
        ec = "#333" if i in active else "#aaa"
        box = FancyBboxPatch(
            (ex_x + 0.005, row_y), per_exp_w - 0.01, row_h,
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=0.6, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(box)

    # Label under expert row
    ax.text(cx, row_y - 0.22, "64 experts — 8 active per token (yellow)",
            ha="center", va="top", fontsize=10, color=COLOR_TEXT)

    # Router → experts indicator arrows (one symbolic broad arrow)
    arrow(ax, cx, 7.0, cx, 6.5, lw=1.2)

    # Experts → combine
    arrow(ax, cx, row_y, cx, 5.3, lw=1.2)

    # Combine block
    draw_block(ax, cx - 1.3, 4.9, 2.6, 0.5,
               "Combine (weighted sum of 8 expert outputs)",
               COLOR_BLOCK, fontsize=9, fontweight="bold")

    # Down out
    arrow(ax, cx, 4.9, cx, 4.5)

    # Layer label
    ax.text(cx, 4.2, "× 16 layers", ha="center", va="center",
            fontsize=11, style="italic", color=COLOR_TEXT)

    # Column title
    ax.text(cx, 10.1, "MoE layer (OLMoE)\nsame attention, different FFN",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
            color=COLOR_TEXT)


def main():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Axis bounds (used like a coordinate system for drawing)
    ax.set_xlim(0, 12)
    ax.set_ylim(3.5, 11.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw two columns
    draw_dense(ax, cx=3.0)
    draw_moe(ax, cx=8.2)

    # Bottom tag line
    fig.text(0.5, 0.02,
             "7B total parameters, ~1B active per token — MoE's capacity-vs-compute trick.",
             ha="center", fontsize=11, style="italic", color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=COLOR_BG)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
