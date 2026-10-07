"""Run directories on W&B, so a run survives losing the machine it trains on (free Colab sessions die).

    python -m catan_rl.wandb_store load --name <run>   # latest uploaded version -> runs/<run>/
    python -m catan_rl.wandb_store save --name <run>   # upload runs/<run>/ (e.g. after the strategy analysis)

`ppo --wandb --save-every 2e6` uploads during training; after a crash, `load` on a fresh machine and continue with
`ppo --name <run> --resume runs/<run>/latest.pt ...` (the restored `wandb_id` continues the same W&B run). Metrics
logged between the last upload and the crash are logged again, so W&B shows that stretch twice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catan_rl.tracking import upload_dir


def load(name: str, project: str) -> Path:
    import wandb

    api = wandb.Api()
    out = Path("runs") / name
    art = api.artifact(f"{api.default_entity}/{project}/{name}-files:latest")
    art.download(root=str(out))
    print(f"restored {art.name} ({art.metadata or ''}) into {out}")
    return out


def save(name: str, project: str) -> None:
    import wandb

    run_dir = Path("runs") / name
    run_id = (run_dir / "wandb_id").read_text().strip()
    run = wandb.init(project=project, id=run_id, resume="allow", dir=str(run_dir),
                     settings=wandb.Settings(save_code=False))
    upload_dir(run, run_dir)
    run.finish()
    print(f"uploaded {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["load", "save"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="catan")
    args = ap.parse_args()
    (load if args.action == "load" else save)(args.name, args.project)


if __name__ == "__main__":
    main()
