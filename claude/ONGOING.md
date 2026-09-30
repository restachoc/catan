# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- Nothing is running. (The web server may still be up on port 8765 from an earlier session; restart it
  to pick up new `best.pt` files.)

## Open decisions (waiting on the owner)

- **Next direction after the diagnostics.** Candidates, roughly in the order discussed:
  1. **Board-structured network** (GNN or transformer over hexes/vertices/edges, per-location policy
     heads). Motivated by the generalisation failure (runs 1 and 4) and by AlphaZero's value memorisation.
  2. **AlphaZero fixes** (see EXPERIMENTS.md "AlphaZero"): position subsampling, mixed z/q value target,
     value-head regularisation, held-out value metric. Then a second AZ run.
  3. **Pure self-play with VP shaping**, to separate the opponent effect from the reward effect in run 2.
  4. **Continue run 2** with `--resume runs/selfplay/latest.pt --total-steps 30e6` to test whether it's
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
