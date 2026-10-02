# Experiments

Training runs so far and their final numbers. What they showed lives in `findings/` (one file per finding).
Linked from [CLAUDE.md](../CLAUDE.md).

All runs: 4 players, 2×256 MLP unless noted, evaluated with the policy in one seat vs 3 heuristic bots (chance = 25%).
Charts: `plots/all_runs.png` (training curves, every snapshot on the same 400 games; az1 isn't in it since it
kept no snapshots), `plots/generalization.png` (fixed vs random boards),
`plots/<run>_strategy.png` (strategy mix over training).

## Runs

"Final" columns: the run's `best.pt`, 2000 games per cell (`runs/<run>/generalization.json`).

| Run dir | Name in charts | Setup (vs the 6M-step diagnostic PPO recipe) | Fixed board: wr / VP | Random boards: wr / VP |
|---|---|---|---|---|
| `diag` | Run 1 | PPO, mixed opponents (25% heuristic, 25% self-play, 50% league), VP shaping 0.5, fixed board | **43.1% / 8.2** | 0.8% / 3.8 |
| `selfplay` | Run 2 | PPO, pure self-play, win/loss only (`--frac-heuristic 0 --frac-selfplay 1 --vp-coef 0`) | 1.5% / 4.7 | 0.5% / 3.4 |
| `selfplay-randboard` | Run 3 | Run 2 + `--random-board` | 1.2% / 3.8 | 1.6% / 4.1 |
| `diag-randboard` | Run 4 | Run 1 + `--random-board` | 0.1% / 3.4 | 0.5% / 3.7 |
| `gnn-randboard` | Run 5 (GNN) | Run 4 with `--arch gnn --hidden 64 --layers 4` (0.30M params), trained on a Colab T4 with `--amp` | 11.2% / 6.4 | 8.7% / 5.7 |
| `gnn128-randboard` | Run 6 (GNN d128) | Run 5 with `--hidden 128` (1.15M params) | 11.7% / 6.3 | 13.4% / 6.3 |
| `gnn128-l7-randboard` | Run 7 (d128, 7 rounds) | Run 6 with `--layers 7` (2.02M params) **and** warmup + constant LR (run 6 decayed to 5%); `--wandb` | 11.7% / 6.2 | **15.0% / 6.3** |
| `az1` | AlphaZero az1 | AlphaZero, 64 sims, 256 games, 3M samples (~1.6 h), pure self-play, win/loss | 0.0% / 3.1 (raw policy) | 0.1% / 2.8 |

az1 with search (in-training eval, 200 games): 0–1.5% win rate, VP 3.0–3.6, no upward trend.

## Findings from these runs

- [mlp-memorises-board-layouts](../findings/mlp-memorises-board-layouts.md): runs 1, 3, 4.
- [gnn-generalises-across-boards](../findings/gnn-generalises-across-boards.md): run 5 vs run 4.
- [reward-shaping-and-opponents](../findings/reward-shaping-and-opponents.md): run 1 vs run 2.
- [gnn-width-d64-vs-d128](../findings/gnn-width-d64-vs-d128.md): run 6 vs run 5.
- [gnn-rounds-4-vs-7](../findings/gnn-rounds-4-vs-7.md): run 7 vs run 6.
- [bot-strategies](../findings/bot-strategies.md): strategy mix of runs 1–3 and 5–7.
- [alphazero-value-memorisation](../findings/alphazero-value-memorisation.md): az1.
- [evaluation-noise](../findings/evaluation-noise.md): why the in-training curves are bumpy; fixed-game curves.
