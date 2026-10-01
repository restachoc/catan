"""Re-evaluate a run's saved snapshots on the fixed evaluation games, for smooth training curves.

    python -m catan_rl.eval_curve <run> ... [--games 400] [--seed 10000]

The in-training evaluations of older runs used a new game set every time (seed 10000 + iter), so their curves
mix policy changes with game-set luck (±4 pp at 400 games). This replays every snapshot in runs/<run>/pool
plus latest.pt on the same games (`evaluate` is fully determined by its seed). Cached per checkpoint in
runs/<run>/eval_curve.csv; `plot_runs --fixed` plots it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from catan_rl.evaluate import evaluate
from catan_rl.model import load

FIELDS = ["checkpoint", "steps", "games", "seed", "win_rate", "avg_vp"]


def eval_curve(run: str, games: int = 400, seed: int = 10_000) -> list[dict]:
    rdir = Path("runs") / run
    random_board = json.loads((rdir / "config.json").read_text()).get("random_board", False)
    path = rdir / "eval_curve.csv"
    cache = {}
    if path.exists():
        with path.open() as f:
            cache = {r["checkpoint"]: r for r in csv.DictReader(f)}
    rows = []
    for ck in sorted((rdir / "pool").glob("*.pt")) + [rdir / "latest.pt"]:
        if not ck.exists():
            continue
        key = str(ck.relative_to(rdir))
        steps = torch.load(ck, map_location="cpu", weights_only=False).get("steps", 0)
        c = cache.get(key)
        if c and int(c["steps"]) == steps and int(c["games"]) == games and int(c["seed"]) == seed:
            rows.append(c)
            continue
        ev = evaluate(load(ck), "heuristic", games=games, random_board=random_board, seed=seed)
        rows.append({"checkpoint": key, "steps": steps, "games": games, "seed": seed,
                     "win_rate": ev["win_rate"], "avg_vp": ev["avg_vp"]})
        write(path, rows)  # after every checkpoint, so an interrupted run keeps its progress
        print(f"{run:<20} {key:<22} {steps / 1e6:6.2f}M  wr {ev['win_rate']:.3f}  vp {ev['avg_vp']:.2f}", flush=True)
    rows.sort(key=lambda r: int(r["steps"]))
    write(path, rows)
    return rows


def write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=10_000)
    args = ap.parse_args()
    for run in args.runs:
        eval_curve(run, args.games, args.seed)


if __name__ == "__main__":
    main()
