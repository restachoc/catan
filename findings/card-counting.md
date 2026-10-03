# Card counting in the observation: no visible effect by 5M steps

*2026-10-03. Run 9 (`gnn128-cards-randboard`) vs run 8 (`gnn128-vpkeep-randboard`). Curves: W&B only (run 9's Colab
session died at 5.1M of 6M steps, so there are no checkpoints and no 2000-game evaluation).*

**Setup.** Run 8's recipe (GNN d128, 4 rounds, random boards, mixed opponents, constant LR after warmup, constant VP
reward) with card counting: every player's expected resource counts, as deduced by the acting seat
(`belief.rs`), added to the observation (OBS_SIZE 1292 → 1312). Nothing else changed.

**Result** (in-training evaluations, the same 400 fixed games for every snapshot; ±1.7 points per point):

| | Win rate, mean 4–5.1M | VP, mean 4–5.1M |
|---|---|---|
| Run 8 (no card counting) | 11.3% | 5.88 |
| Run 9 (card counting) | 10.4% | 5.97 |

Same within noise; run 9 was a little ahead in VP from ~2M (5.5 vs 5.0 at 2.2M) but converged to the same ~6 VP.

**Implications.**
- Knowing opponents' hands isn't what holds the bot at ~6 VP against the heuristic bots, which ignore opponents'
  hands entirely; the information may matter more against stronger opponents or for trading (v3/v4).
- Kept in the observation: it's cheap (~7% training speed on CPU) and needed for later versions.
- With runs 6–9 all at ~6 VP, the remaining suspects are the global-token bottleneck and the opponent mix
  (only 25% of games against the heuristic bots it's evaluated on).
