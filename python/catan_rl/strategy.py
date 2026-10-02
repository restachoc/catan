"""Which strategies do the bots adopt in self-play, and how does that change over training?

    python -m catan_rl.strategy <run> [--games 300] [--device cuda]

For every snapshot in runs/<run>/pool (plus latest.pt), plays `games` pure self-play games (all seats
the same snapshot, on the run's board setting), records each player's end-of-game statistics, and classifies the strategy that
player followed (see `classify`). Results are cached in runs/<run>/strategy.csv; only new snapshots are
evaluated on a re-run. The chart goes to plots/<run>_strategy.png with two panels: winners only and all
players.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from catan_rl import N_ACTIONS, OBS_SIZE, VecEnv
from catan_rl._engine import STAT_NAMES
from catan_rl.model import PolicyNet, load

S = {n: i for i, n in enumerate(STAT_NAMES)}

# Strategy labels in plot order (bottom to top).
STRATEGIES = ["Road builder", "Balanced", "OWS: dev cards", "OWS: cities", "Undeveloped"]
DESCRIPTIONS = {
    "Road builder": ">= 60% of spent cards went to roads + extra settlements",
    "Balanced": "40-60% of spent cards on expansion",
    "OWS: dev cards": "<= 40% on expansion; more cards into dev cards than cities",
    "OWS: cities": "<= 40% on expansion; more cards into cities than dev cards",
    "Undeveloped": "fewer than 12 cards spent on anything",
}


def spending(st: np.ndarray) -> dict[str, np.ndarray]:
    """Resource cards each player spent per purchase type (setup pieces are free and excluded)."""
    buildings = st[:, S["settlements"]] + st[:, S["cities"]]
    return {
        "roads": 2 * (st[:, S["roads"]] - 2).clip(min=0),
        "settlements": 4 * (buildings - 2).clip(min=0),
        "cities": 5 * st[:, S["cities"]],
        "dev": 3 * st[:, S["dev_bought"]],
    }


def classify(st: np.ndarray) -> np.ndarray:
    """Label index per row of end-of-game stats [M, len(STAT_NAMES)], from where cards were spent."""
    sp = spending(st)
    total = sum(sp.values())
    expansion = (sp["roads"] + sp["settlements"]) / np.maximum(total, 1)
    label = np.full(len(st), STRATEGIES.index("Balanced"))
    label[expansion >= 0.6] = STRATEGIES.index("Road builder")
    ows = expansion <= 0.4
    label[ows & (sp["dev"] >= sp["cities"])] = STRATEGIES.index("OWS: dev cards")
    label[ows & (sp["dev"] < sp["cities"])] = STRATEGIES.index("OWS: cities")
    label[total < 12] = STRATEGIES.index("Undeveloped")
    return label


@torch.inference_mode()
def selfplay_stats(net: PolicyNet, games: int, seed: int = 4242, random_board: bool = False,
                   n_players: int = 4) -> np.ndarray:
    """Play `games` self-play games; returns stats [games * n_players, len(STAT_NAMES)]."""
    N = min(games, 256)
    env = VecEnv(N, seed=seed, n_players=n_players, random_board=random_board)
    obs = np.zeros((N, OBS_SIZE), np.float32)
    mask = np.zeros((N, N_ACTIONS), np.bool_)
    actor = np.zeros(N, np.int64)
    done = np.zeros(N, np.bool_)
    winner = np.zeros(N, np.int64)
    length = np.zeros(N, np.int64)
    final_vp = np.zeros((N, 4), np.int64)
    env.reset(obs, mask, actor)
    net.eval()
    rows, finished = [], np.zeros(N, np.int64)
    per_env = -(-games // N)
    while len(rows) < games * n_players:
        a = net.act(obs, mask)[0]
        env.step(a, obs, mask, actor, done, winner, length, final_vp)
        if done.any():
            st = env.last_game_stats()
            for e in np.nonzero(done)[0]:
                if finished[e] < per_env:
                    finished[e] += 1
                    rows.extend(st[e, :n_players])
    return np.array(rows[: games * n_players], np.float32)


def summarize(st: np.ndarray) -> dict:
    labels = classify(st)
    won = st[:, S["won"]] > 0
    out = {}
    for group, sel in (("all", np.ones(len(st), bool)), ("winners", won)):
        n = max(1, sel.sum())
        for i, name in enumerate(STRATEGIES):
            out[f"{group}:{name}"] = float((labels[sel] == i).sum() / n)
        sp = spending(st[sel])
        total = max(1.0, float(sum(v.sum() for v in sp.values())))
        for k, v in sp.items():
            out[f"{group}:spend_{k}"] = float(v.sum() / total)
    out["draw_rate"] = 1 - won.sum() / (len(st) / 4)
    return out


def analyze(run: str, games: int, random_board: bool, device: str = "cpu") -> list[dict]:
    rdir = Path("runs") / run
    cache_path = rdir / "strategy.csv"
    cache = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for row in csv.DictReader(f):
                cache[row["checkpoint"]] = row
    ckpts = sorted((rdir / "pool").glob("*.pt")) + [rdir / "latest.pt"]
    rows = []
    for ck in ckpts:
        if not ck.exists():
            continue
        key = str(ck.relative_to(rdir))
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        if key in cache and int(cache[key]["steps"]) == meta.get("steps", 0) and int(cache[key]["games"]) == games:
            rows.append({k: (float(v) if k not in ("checkpoint",) else v) for k, v in cache[key].items()})
            continue
        st = selfplay_stats(load(ck).to(device), games, random_board=random_board)
        row = {"checkpoint": key, "steps": meta.get("steps", 0), "games": games, **summarize(st)}
        rows.append(row)
        write_cache(cache_path, rows)  # after every snapshot, so an interrupted run keeps its progress
        print(f"{key:<24} {row['steps'] / 1e6:6.2f}M  winners: " +
              "  ".join(f"{s} {row[f'winners:{s}']:.0%}" for s in STRATEGIES), flush=True)
    rows.sort(key=lambda r: float(r["steps"]))
    write_cache(cache_path, rows)
    return rows


def write_cache(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


SPEND = [("roads", "Roads"), ("settlements", "Extra settlements"), ("cities", "Cities"), ("dev", "Dev cards")]


def plot(run: str, rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from catan_rl.plot_runs import INK, INK2, SERIES, SURFACE

    x = np.array([float(r["steps"]) / 1e6 for r in rows])
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE, sharex=True, sharey=True)
    strat_colors = SERIES[:4] + ["#b9b7b0"]  # four categorical slots + neutral for "Undeveloped"
    layers = [
        (0, STRATEGIES, [f"{{g}}:{s}" for s in STRATEGIES], strat_colors, "strategy mix"),
        # Same hue as the matching strategy above (roads = road builder, cities / dev cards = the OWS
        # variants); extra settlements take the next free categorical slot (magenta).
        (1, [l for _, l in SPEND], [f"{{g}}:spend_{k}" for k, _ in SPEND],
         [SERIES[0], SERIES[4], SERIES[3], SERIES[2]], "where resource cards were spent"),
    ]
    for row_i, names, keys, colors, what in layers:
        for col_i, (group, title) in enumerate((("winners", "Winning players"), ("all", "All players"))):
            ax = axes[row_i, col_i]
            ys = [np.array([float(r[k.format(g=group)]) * 100 for r in rows]) for k in keys]
            ax.set_facecolor(SURFACE)
            ax.stackplot(x, ys, colors=colors, edgecolor=SURFACE, linewidth=1.5, labels=names)
            ax.set_title(f"{title}: {what}", loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
            base = 0.0
            for y in ys:  # direct value labels at the right edge for bands thick enough to hold one
                if y[-1] >= 7:
                    ax.text(x[-1], base + y[-1] / 2, f"{y[-1]:.0f}% ", ha="right", va="center", color=INK,
                            fontsize=9, fontweight="bold")
                base += y[-1]
            if row_i == 1:
                ax.set_xlabel("Training steps (millions)")
        h, l = axes[row_i, 0].get_legend_handles_labels()
        axes[row_i, 1].legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                              labelcolor=INK)
    notes = "\n".join(f"{s}: {DESCRIPTIONS[s]}" for s in STRATEGIES)
    games = int(float(rows[0]["games"]))
    fig.text(0.01, 0.01, f"Each point: {games} self-play games of one snapshot (all seats identical). Spending counts "
             "cards paid for purchases after setup (road 2, settlement 4, city 5, dev card 3).\n"
             f"Strategy classification per player:\n{notes}", color=INK2, fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = Path("plots") / f"{run}_strategy.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = json.loads((Path("runs") / args.run / "config.json").read_text())
    rows = analyze(args.run, args.games, cfg.get("random_board", False), args.device)
    print(f"saved {plot(args.run, rows)}")


if __name__ == "__main__":
    main()
