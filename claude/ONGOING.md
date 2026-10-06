# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- **Transformer (run 11) learned much slower than the GNN** ([finding](../findings/transformer-vs-gnn.md)). Proposed
  next: same size, distance bias initialised as a locality prior (−1 per graph-distance step); else the hybrid.
- **Trading is parked, off by default** (`--trading`). Fix the network first (the ~6 VP plateau), then revisit with
  1 proposal per turn, no entropy bonus on trade decisions, or a curriculum from a no-trading model.
- **Colab sessions keep dying before the zip downloads** (run 7's first try at 41 min, run 8 at the usage limit,
  run 9 at 55 min/5.1M steps). Runs 8 and 9 exist only as W&B curves. Before the next GPU run, consider saving the
  run directory to Google Drive during training, or shorter jobs.
- **The GNN plateaus at ~6 VP / ~12–15% from ~3.5M steps** in runs 6–8. Ruled out: LR decay, more rounds, the VP fade, missing
  card counting (run 9).
  Candidates left: the global-token bottleneck (hybrid with seat tokens), the opponent mix (25% heuristic), more steps.
- **Board network: the GNN is chosen.** Run 5 (d64 L4) reaches 8.7% on random boards vs the MLP's 0.5%, still
  rising ([finding](../findings/gnn-generalises-across-boards.md)); d128 (run 6) reaches 13.4%
  ([finding](../findings/gnn-width-d64-vs-d128.md)). The hybrid (GNN + attention to global/seat
  tokens) is the upgrade path if the 64-dim global token proves a bottleneck.
- **Next experiments, proposed to the owner** (launch with `--wandb` from now on):
  1. GNN d64 on the fixed board, 6M steps (~40 min): if it nears run 1's 43%, the architecture is adequate.
  2. The GNN at 20–30M steps (constant LR after warmup and constant VP reward are now the defaults). Redo in
     one Colab session; runs aren't in git, so resuming needs the checkpoint uploaded.
  3. 6 layers (~1.5× slower) if those stall. Use d128 as the base (run 6 beat d64).
- W&B: runs 1–6 are backfilled (run 6's strategy data covers 23 of 29 snapshots; the CPU analysis was stopped);
  run 7 on is logged live. A toy run `wandb-check` is in the project; the owner may delete it.
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
