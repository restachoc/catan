# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- **Run 13 (`tf128-l4h8-randboard`, transformer d128 L4 h8, 0.62M params): resumed from 9M toward 12M on Colab
  (2026-10-08, launched with the cell below; ~70 min).** It was paused at 9M of 12M steps. The
  continuation's session died at 9.81M (2026-10-07 22:19); the last upload (`-files:v9`) is at 9M. A new GPU check
  hung (probably the Colab GPU limit). It is still improving and now slightly above the GNN's plateau (in-training
  means: 4–6M 7.8% / 5.66 VP, 6–8M 9.3% / 5.83, 8–10M 11.6% / 6.02; run 9 GNN 10.4% / 5.97 at 4–5.1M). To finish: on a
  T4, rerun the first Colab cell (`colab.sh wandb_store load --name tf128-l4h8-randboard`, then the same `ppo` command
  with `--resume runs/tf128-l4h8-randboard/latest.pt --total-steps 12e6 --save-every 1e6`, then strategy +
  `wandb_store save`). Afterwards: `wandb_store load` locally (replaces the local 6M copy), 2000-game evals on both
  boards (`evaluate`, not `generalization`, which would redraw the shared chart), strategy chart, and extend
  [transformer-vs-gnn](../findings/transformer-vs-gnn.md) with run 13 + an EXPERIMENTS row update.

## Where the network stands

- **GNN d128 L4 plateaus at ~6 VP / ~10–15% from ~3.5M steps** (runs 6–9). Ruled out as causes: LR decay, more
  rounds, the VP fade, missing card counting.
- **Transformer:** cost-matched d96 L3 was slow without position info (run 11), ~0.5 VP behind the GNN with position
  embeddings + locality prior (run 12). The bigger d128 L4 h8 (run 13, ~1.6× the GNN's cost per sample) kept improving
  past the GNN's plateau. If run 13 ends clearly above the GNN, the transformer becomes the main network.
- Remaining candidates for the plateau: the opponent mix (only 25% of games vs the heuristic it's evaluated on),
  longer training (now practical: runs resume from W&B), the global-token bottleneck (wider helped).
- **Trading is parked, off by default** (`--trading`). Revisit once the network plays better: 1 proposal per turn,
  no entropy bonus on trade decisions, or a curriculum from a no-trading model.
- W&B: runs 1–6 backfilled (run 6's strategy data covers 23 of 29 snapshots); run 7 on logged live. The owner was
  asked to confirm the project's visibility (entity `gianni-van-de-velde-universiteit-gent`).

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
