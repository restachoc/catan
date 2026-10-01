"""Upload finished runs that trained without --wandb to W&B, under the same metric names live runs use.

    python -m catan_rl.wandb_backfill <run> ... [--project catan] [--replace]

Per run: metrics.csv (training curves); eval_curve.csv as eval/win_rate and eval/vp (the same fixed seed-10000
games live runs evaluate on, so old and new runs compare directly), with the old per-iteration evaluations
under eval_legacy/; strategy.csv as strategy_selfplay/ (self-play games, unlike the live strategy/ metrics
from training games); generalization.json into the run summary. The W&B run id goes to runs/<run>/wandb_id,
so a later `ppo --resume --wandb` continues the same W&B run. --replace deletes a previous backfill first.
Credentials as for live runs (see tracking.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
from pathlib import Path

from catan_rl.tracking import LEGACY

AZ = {
    "steps": "train/steps", "games": "game/finished", "updates": "train/updates", "hours": "perf/hours",
    "samples_per_s": "perf/samples_per_s", "loss_pi": "az/policy_loss", "loss_v": "az/value_loss",
    "entropy": "az/entropy", "eval_wr_heuristic": "eval_legacy/win_rate", "eval_vp": "eval_legacy/vp",
    "eval_search_wr": "eval_legacy/search_win_rate", "eval_search_vp": "eval_legacy/search_vp",
}
OLD_EVAL = {"eval_wr_heuristic": "eval_legacy/win_rate", "eval_vp": "eval_legacy/vp"}


def num(v: str):
    try:
        return float(v)
    except ValueError:
        return None


def rows_by_step(rdir: Path) -> dict[int, dict]:
    """All logged values of a run, merged per training step."""
    by_step: dict[int, dict] = {}

    def add(step: int, vals: dict) -> None:
        by_step.setdefault(step, {"train/steps": step}).update({k: v for k, v in vals.items() if v is not None})

    with (rdir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    names = AZ if "loss_pi" in rows[0] else {**LEGACY, **OLD_EVAL}
    for r in rows:
        add(int(float(r["steps"])), {names[k]: num(v) for k, v in r.items() if k in names and k != "steps"})
    if (rdir / "eval_curve.csv").exists():
        with (rdir / "eval_curve.csv").open() as f:
            for r in csv.DictReader(f):
                add(int(r["steps"]), {"eval/win_rate": num(r["win_rate"]), "eval/vp": num(r["avg_vp"])})
    if (rdir / "strategy.csv").exists():
        with (rdir / "strategy.csv").open() as f:
            for r in csv.DictReader(f):
                add(int(float(r["steps"])), {
                    "strategy_selfplay/" + k.replace(":", "/"): num(v)
                    for k, v in r.items() if k not in ("checkpoint", "steps", "games")})
    return dict(sorted(by_step.items()))


def backfill(run: str, project: str, replace: bool) -> None:
    import wandb

    rdir = Path("runs") / run
    id_file = rdir / "wandb_id"
    if id_file.exists():
        if not replace:
            print(f"{run}: already on W&B (runs/{run}/wandb_id); use --replace to redo it")
            return
        api = wandb.Api()
        try:
            api.run(f"{api.default_entity}/{project}/{id_file.read_text().strip()}").delete()
        except wandb.errors.CommError:
            pass
    cfg = json.loads((rdir / "config.json").read_text())
    arch = "az" if (rdir / "metrics.csv").open().readline().count("loss_pi") else cfg.get("arch", "mlp")
    run_id = secrets.token_hex(6)
    id_file.write_text(run_id)
    wb = wandb.init(project=project, name=run, id=run_id, config={**cfg, "arch": arch}, tags=["backfill", arch],
                    dir=str(rdir), settings=wandb.Settings(save_code=False))
    wb.define_metric("train/steps")
    wb.define_metric("*", step_metric="train/steps")
    rows = rows_by_step(rdir)
    for vals in rows.values():
        wb.log(vals)
    gen = rdir / "generalization.json"
    if gen.exists():
        g = json.loads(gen.read_text())
        for board, key in (("fixed", "beginner_board"), ("random", "random_board")):
            wb.summary[f"final/{key}/win_rate"] = g[board]["win_rate"]
            wb.summary[f"final/{key}/vp"] = g[board]["avg_vp"]
        wb.summary["final/games"] = g["games"]
    wb.finish()
    print(f"{run}: {len(rows)} steps uploaded")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--project", default="catan")
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    for run in args.runs:
        backfill(run, args.project, args.replace)


if __name__ == "__main__":
    main()
