# Ongoing

What's in flight, open decisions and next steps. **Read this first after a context clear.** Keep it
short and current: delete items when they're done. Linked from [CLAUDE.md](../CLAUDE.md).

## In flight

- Nothing training. Run 13 finished at 12M (2026-10-08). Its strategy step ran on Colab after the local copy was
  fetched; `wandb_store load --name tf128-l4h8-randboard` again to get the full `strategy.csv` + chart.

## Where the network stands

- **Every network plateaus at ~6 VP / ~13–15% on random boards** with the same strategy (dev cards, little
  expansion): GNN d128 (runs 6–9) at 6M, transformer d128 L4 (run 13) at 12M
  ([transformer-vs-gnn](../findings/transformer-vs-gnn.md)). Ruled out: LR decay, more rounds, the VP fade, card
  counting, network type and size. **The network is not the main limit**; next work is on the training setup.
- **Where run 13 loses** (6M `best.pt`, 2000 games per cell, scratch analysis, not yet a finding): swapping setup /
  rest-of-game control with the heuristic gives net/net 10.7%, heur setup 13.7%, heur play 17.5%, heur/heur 26.4%.
  Its setup has 1.9 fewer pips and wood+brick in 42% vs 68% of games; in play it builds 0.95 extra settlements vs 1.75
  and more roads (chases Longest Road). VP is equal to turn ~40; the gap opens late. So: under-expansion, both phases.
  The script was session scratch (lost): many `Game`s in Python, test seat's setup/play by `net.act` or
  `g.bot_action("heuristic")`, seeds 50000+k, seat k%4. Rebuild it in the repo if this becomes a finding.
- Candidate next runs (one change each): `--frac-heuristic ~0.6` (only 25% of games are vs the heuristic it's
  evaluated on); in-game shaping for VP / production gained (potential-based) to credit expansion; setup trained or
  scripted separately. Longer term: search at play time with the PPO network.
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
