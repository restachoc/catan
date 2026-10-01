# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- **Board network: the GNN is chosen and trained once.** Run 5 (`gnn-randboard`, d64 L4, 6M steps on a Colab T4)
  reaches 8.7% on random boards vs the MLP's 0.5%, still rising
  ([finding](../findings/gnn-generalises-across-boards.md)). GNN + attention to global/seat tokens (the "hybrid")
  stays the upgrade path if the global token proves a bottleneck; transformer and hybrid are benchmark-only.
- **Strategy analysis of run 5** (`strategy.py gnn-randboard`) is running locally at nice 10; when done, look at
  `plots/gnn-randboard_strategy.png` and add run 5 to [bot-strategies](../findings/bot-strategies.md).
- **Next experiments, proposed to the owner:**
  1. GNN d64 on the fixed board, 6M steps (~40 min): if it nears run 1's 43%, the architecture is adequate.
  2. Run 5 continued or redone at 20–30M steps with stretched LR and VP-shaping schedules. Resuming needs
     `runs/gnn-randboard/latest.pt` on Colab (runs aren't in git), so redoing in one session is simpler.
  3. d128 (~2× slower, ~70 min per 6M) or 6 layers (~1.5×) only if both stall.

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
