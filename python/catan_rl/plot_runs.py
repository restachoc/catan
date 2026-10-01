"""Plot evaluation curves of training runs side by side.

    python -m catan_rl.plot_runs diag selfplay [--labels "Mixed + VP" "Self-play"] [--slots 1 3] [--out plots/x.png]

Output defaults to plots/<run>_vs_<run>.png, one file per comparison.

Two panels sharing the x-axis (training steps): win rate vs 3 heuristic bots, and average final VP.
Uses runs/<run>/eval_curve.csv when it exists (every snapshot re-evaluated on the same games, see
eval_curve.py), else the in-training evaluations in metrics.csv (on fixed games since 2026-10-01).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (fixed order, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(run: str) -> tuple[list[float], list[float], list[float]]:
    steps, wr, vp = [], [], []
    curve = Path("runs") / run / "eval_curve.csv"
    if curve.exists():
        with curve.open() as f:
            for row in csv.DictReader(f):
                steps.append(int(row["steps"]) / 1e6)
                wr.append(float(row["win_rate"]) * 100)
                vp.append(float(row["avg_vp"]))
        return steps, wr, vp
    with (Path("runs") / run / "metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if row["eval_wr_heuristic"]:
                steps.append(int(row["steps"]) / 1e6)
                wr.append(float(row["eval_wr_heuristic"]) * 100)
                vp.append(float(row["eval_vp"]))
    return steps, wr, vp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--slots", nargs="+", type=int, help="palette slot (1-6) per run, so a run keeps its colour across charts")
    ap.add_argument("--out", help="default: plots/<run>_vs_<run>.png")
    args = ap.parse_args()
    labels = args.labels or args.runs
    colors = [SERIES[k - 1] for k in args.slots] if args.slots else SERIES

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor=SURFACE)
    panels = [
        (axes[0], 1, "Win rate vs 3 heuristic bots", "%", 25, "chance (25%)"),
        (axes[1], 2, "Average final VP vs 3 heuristic bots", "VP", 10, "win threshold (10 VP)"),
    ]
    data = [load(r) for r in args.runs]
    for ax, col, title, unit, ref, ref_label in panels:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.axhline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0, ref, f" {ref_label}", color=INK2, fontsize=9, va="bottom")
        ends = []
        for i, (d, label) in enumerate(zip(data, labels)):
            x, y = d[0], d[col]
            ax.plot(x, y, color=colors[i], linewidth=2, solid_joinstyle="round", solid_capstyle="round", label=label)
            ax.plot(x[-1], y[-1], "o", color=colors[i], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            ends.append((y[-1], x[-1]))
        # End-value labels, spread vertically so converging lines don't collide; leader lines keep the link.
        top = 60 if unit == "%" else 10.5
        gap = 0.06 * top
        placed: list[float] = []
        x_lab = max(x for _, x in ends) + 0.03 * max(max(d[0]) for d in data)
        for yv, xv in sorted(ends):
            yl = max(yv, placed[-1] + gap) if placed else yv
            placed.append(yl)
            if abs(yl - yv) > 1e-9 or x_lab - xv > 0.08 * x_lab:
                ax.plot([xv, x_lab], [yv, yl], color=INK2, linewidth=0.8)
            ax.text(x_lab, yl, f" {yv:.1f}{'%' if unit == '%' else ''}", va="center", color=INK, fontsize=10,
                    fontweight="bold")
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Training steps (millions)")
        ax.set_ylim(0, 60 if unit == "%" else 10.5)
        ax.set_xlim(0, max(max(d[0]) for d in data) * 1.12)
    # Legend below both panels, so it never covers a curve.
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=min(3, len(names)),
               frameon=False, labelcolor=INK)
    note = ("Each point: the same 400 games (boards, dev decks, dice) for every snapshot, policy in one seat, "
            "3 heuristic bots in the others.")
    fig.text(0.01, 0.01, note,
             color=INK2, fontsize=9)
    rows = -(-len(names) // 3)
    fig.tight_layout(rect=(0, 0.06 + 0.045 * rows, 1, 1))
    out = Path(args.out or f"plots/{'_vs_'.join(args.runs)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
