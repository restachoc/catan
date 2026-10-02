# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- **Run 7 (`gnn128-l7-randboard`) is done** (zip unpacked, strategy.csv complete). Still to do: `strategy
  gnn128-l7-randboard` (cached, redraws the chart), `generalization` with runs 5–8 once run 8 is in, `plot_runs`
  (next palette slot), finding (7 rounds ≈ 4 rounds: in-training curves match to within noise; +72% params) +
  EXPERIMENTS row.
- **Run 8 (`gnn128-vpkeep-randboard`) is training on Colab** with `--wandb` (started 2026-10-02 ~12:05, ~70 min):
  run 6's recipe (d128, 4 rounds) with `--vp-anneal-frac 0` (VP reward never fades) and the new warmup + constant
  LR. Tests whether the plateau at ~6 VP from ~3.5M steps in runs 6 and 7 comes from the VP reward fading out.
  Vs run 6 two things change (LR, fade), but run 7 showed the LR schedule barely matters. Same pickup steps as run 7.
- **Run 6 strategy analysis** (started 2026-10-01 ~17:00, niced, 7 threads): `strategy gnn128-randboard`, then
  `wandb_backfill gnn128-randboard --replace --fixed-evals` (its evals already used the fixed games; keep the
  flag). Log: `runs/gnn128-randboard/strategy.log`. When done: look at `plots/gnn128-randboard_strategy.png`, add
  run 6 to [bot-strategies](../findings/bot-strategies.md). The analysis caches per snapshot, so a rerun is cheap.
- **Board network: the GNN is chosen.** Run 5 (d64 L4) reaches 8.7% on random boards vs the MLP's 0.5%, still
  rising ([finding](../findings/gnn-generalises-across-boards.md)); d128 (run 6) reaches 13.4%
  ([finding](../findings/gnn-width-d64-vs-d128.md)). The hybrid (GNN + attention to global/seat
  tokens) is the upgrade path if the 64-dim global token proves a bottleneck.
- **Next experiments, proposed to the owner** (launch with `--wandb` from now on):
  1. GNN d64 on the fixed board, 6M steps (~40 min): if it nears run 1's 43%, the architecture is adequate.
  2. The GNN at 20–30M steps (constant LR after warmup now; VP shaping still anneals over half of
     `total_steps`). Redo in one Colab session; runs aren't in git, so resuming needs the checkpoint uploaded.
  3. 6 layers (~1.5× slower) if those stall. Use d128 as the base (run 6 beat d64).
- W&B: everything up to run 6 is backfilled. A toy run `wandb-check` is in the project; the owner may delete it.
  The owner was asked to confirm the project's visibility (team account `gianni-van-de-velde-universiteit-gent`).

## Open decisions (waiting on the owner)

- **Other directions, parked while the network is built:**
  1. **AlphaZero fixes** ([finding](../findings/alphazero-value-memorisation.md)): position subsampling, mixed z/q value target,
     value-head regularisation, held-out value metric. Then a second AZ run (ideally with the new network).
  2. **Pure self-play with VP shaping**, to separate the opponent effect from the reward effect in run 2.
  3. **Continue run 2** with `--resume runs/selfplay/latest.pt --total-steps 30e6` to test whether it's
     stuck or just slow (its learning rate had decayed to ~0).
- **Full v1 PPO run** (3×512, 1B steps, ~2 days on this CPU; faster on the owner's GPU machine): not
  started. The owner wanted to decide after the experiments.
- **League proposal for the long run** (discussed, not implemented): per-game opponent sampling with
  phases: warm-up with the heuristic bot at 60% per seat, annealed to 3%; snapshots every 2M steps with
  weighting toward snapshots the bot struggles against ((1 − p)², where p = how often it finishes ahead);
  anchor snapshots kept permanently; Elo evaluation. Needs a "seat setup for the next game" feature in
  `VecEnv` first. The owner then chose pure self-play for the experiments; revisit before the long run.

## Offered, not started

- **Dynamic thread control** for training runs: a `runs/<name>/threads` control file read each iteration,
  `torch.set_num_threads` plus pre-built rayon pools of several sizes, and a helper to shrink a running
  training while an interactive job runs. Also try `OMP_WAIT_POLICY=PASSIVE`. (`nice` is already the
  default for long runs.)
- **MCTS at play time** against the owner: search with a PPO-trained network in the web UI (a `Game`-level
  search binding plus an `az:`/`search:` bot in `web/server.py`).
