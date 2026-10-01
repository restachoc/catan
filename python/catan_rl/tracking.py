"""Optional Weights & Biases logging for training runs (`--wandb`).

The API key never touches the repo (which is public): wandb reads it from ~/.netrc after `wandb login`, or from
the WANDB_API_KEY environment variable (on Colab, filled from a Colab secret; see CLAUDE.md). The entity comes
from WANDB_ENTITY or the account default; W&B projects are private (or team-only) unless made public there.

metrics.csv keeps its flat column names; W&B gets them grouped by prefix (eval/, game/, strategy/, ppo/, nn/,
perf/, league/), with training samples as the x-axis.
"""

from __future__ import annotations

import secrets
from pathlib import Path

# metrics.csv columns from before the grouped names, mapped to their W&B key
LEGACY = {
    "iter": "train/iter", "steps": "train/steps", "train_wr": "train/win_rate",
    "sps": "perf/samples_per_s", "roll_frac": "perf/rollout_frac", "samples": "perf/samples_per_iter",
    "games": "game/finished", "game_len": "game/length",
    "pg": "ppo/policy_loss", "vf": "ppo/value_loss", "ent": "ppo/entropy", "kl": "ppo/approx_kl",
    "clipfrac": "ppo/clip_frac", "explained_var": "ppo/explained_var",
    "eval_wr_heuristic": "eval/win_rate", "eval_vp": "eval/vp",
}


class Tracker:
    def __init__(self, enabled: bool, project: str, run_dir: Path, name: str, config: dict):
        self.run = None
        if not enabled:
            return
        import wandb

        # A stable id per run directory, so --resume continues the same W&B run.
        id_file = run_dir / "wandb_id"
        run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(6)
        id_file.write_text(run_id)
        self.run = wandb.init(project=project, name=name, id=run_id, resume="allow", config=config,
                              dir=str(run_dir), settings=wandb.Settings(save_code=False))
        self.run.define_metric("train/steps")
        self.run.define_metric("*", step_metric="train/steps")

    def log(self, row: dict) -> None:
        if self.run is not None:
            self.run.log({LEGACY.get(k, k): v for k, v in row.items() if v != "" and v is not None})

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
