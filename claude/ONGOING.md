# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- **Run 6 (`gnn128-randboard`) is training on Colab** (started 2026-10-01 ~15:00, ~70 min): run 5's recipe with
  `--hidden 128`, nothing else changed. Trained without `--wandb` (launched before W&B existed); its in-training
  evals already use the fixed seed-10000 games. Only other difference from run 5: dice now come from their own
  stream (same distribution, different games). When `~/Downloads/gnn128-randboard.zip` exists:
  1. `unzip -qo ~/Downloads/gnn128-randboard.zip` in the repo root, then delete the zip.
  2. Delete the Colab runtime: `open_colab_browser_connection`, then run the `runtime.unassign()` cell (if the
     runtime is already gone, nothing to do).
  3. 2000-game eval: `python -m catan_rl.generalization diag selfplay selfplay-randboard diag-randboard
     gnn-randboard gnn128-randboard --labels "Run 1" "Run 2" "Run 3" "Run 4" "Run 5 (GNN)" "Run 6 (GNN d128)"`
     (half the cores, niced); look at `plots/generalization.png`.
  4. Curves (run 6 = palette slot 7, violet):
     `plot_runs diag-randboard gnn-randboard gnn128-randboard --slots 2 6 7 --labels "Run 4 (MLP)" "Run 5 (GNN d64)"
     "Run 6 (GNN d128)" --out plots/gnn_d64_vs_d128.png`, and add `gnn128-randboard` (slot 7) to `plots/all_runs.png`
     (command: same runs/labels/slots as now, see CLAUDE.md palette note). Look at both charts.
  5. Write `findings/gnn-width-d64-vs-d128.md`, add run 6 to `claude/EXPERIMENTS.md`, index the finding in
     CLAUDE.md, then `python -m catan_rl.wandb_backfill gnn128-randboard`.
  If the zip never arrives (runtime disconnected), the run is lost; tell the owner.
- **Strategy analysis of run 5** (`strategy.py gnn-randboard`, pid 553360, nice 10) was at 27 of 29 snapshots.
  When done: look at `plots/gnn-randboard_strategy.png`, add run 5 to
  [bot-strategies](../findings/bot-strategies.md), and re-upload it:
  `python -m catan_rl.wandb_backfill gnn-randboard --replace` (its W&B strategy data is partial).
- **Board network: the GNN is chosen.** Run 5 (d64 L4) reaches 8.7% on random boards vs the MLP's 0.5%, still
  rising ([finding](../findings/gnn-generalises-across-boards.md)). The hybrid (GNN + attention to global/seat
  tokens) is the upgrade path if the 64-dim global token proves a bottleneck.
- **Next experiments, proposed to the owner** (launch with `--wandb` from now on):
  1. GNN d64 on the fixed board, 6M steps (~40 min): if it nears run 1's 43%, the architecture is adequate.
  2. The GNN at 20–30M steps with stretched LR and VP-shaping schedules (redo in one Colab session; runs aren't
     in git, so resuming needs the checkpoint uploaded).
  3. 6 layers (~1.5× slower) if those stall; d128 depends on run 6.
- W&B: everything up to run 5 is backfilled. A toy run `wandb-check` is in the project; the owner may delete it.
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
