"""Does a run's policy generalise beyond the board it trained on?

    python -m catan_rl.generalization diag selfplay ... [--labels ...] [--games 2000]

Evaluates each run's best.pt vs 3 heuristic bots on the fixed beginner board and on random boards
(raw policy, no search). Results are cached per run in runs/<run>/generalization.json (invalidated
when best.pt changes). Chart: plots/generalization.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_rl.evaluate import evaluate
from catan_rl.model import load

SEED = 555


def results(run: str, games: int) -> dict:
    ck = Path("runs") / run / "best.pt"
    cache = Path("runs") / run / "generalization.json"
    stamp = ck.stat().st_mtime
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("mtime") == stamp and c.get("games") == games:
            return c
    net = load(ck)
    out = {"mtime": stamp, "games": games}
    for key, rb in (("fixed", False), ("random", True)):
        r = evaluate(net, "heuristic", games=games, random_board=rb, seed=SEED)
        out[key] = {"win_rate": r["win_rate"], "avg_vp": r["avg_vp"]}
    cache.write_text(json.dumps(out, indent=2))
    return out


def plot(labels: list[str], res: list[dict], games: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from catan_rl.plot_runs import GRID, INK, INK2, SERIES, SURFACE

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    y = np.arange(len(labels))
    h = 0.36
    panels = [(axes[0], "win_rate", "Win rate vs 3 heuristic bots", 100, 25, "chance (25%)", "%"),
              (axes[1], "avg_vp", "Average final VP vs 3 heuristic bots", 1, 10, "win (10 VP)", "")]
    for ax, key, title, scale, ref, ref_label, unit in panels:
        ax.set_facecolor(SURFACE)
        for k, (board, name) in enumerate((("fixed", "Beginner board"), ("random", "Random boards"))):
            vals = np.array([r[board][key] * scale for r in res])
            ax.barh(y + (k - 0.5) * h, vals, height=h * 0.92, color=SERIES[k], label=name)
            for yi, v in zip(y + (k - 0.5) * h, vals):
                ax.text(v, yi, f" {v:.1f}{unit}", va="center", color=INK, fontsize=9.5)
        ax.axvline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(ref, len(labels) - 0.45, f" {ref_label}", color=INK2, fontsize=9, va="bottom")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=18)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(length=0)
        ax.set_xlim(0, (55 if key == "win_rate" else 11))
    axes[1].set_yticklabels([])
    axes[0].legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.01, f"Each bar: {games} games, the run's best checkpoint (raw policy, no search) in one seat, "
             "3 heuristic bots in the others. The heuristic bot plays any board equally well.",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path("plots") / "generalization.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--games", type=int, default=2000)
    args = ap.parse_args()
    res = [results(r, args.games) for r in args.runs]
    for r, name in zip(res, args.runs):
        print(f"{name:<22} fixed {r['fixed']['win_rate']:6.1%} {r['fixed']['avg_vp']:5.2f} VP | "
              f"random {r['random']['win_rate']:6.1%} {r['random']['avg_vp']:5.2f} VP")
    print(f"saved {plot(args.labels or args.runs, res, args.games)}")


if __name__ == "__main__":
    main()
