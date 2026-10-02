# Keeping the VP reward on doesn't lift the ~6 VP plateau

*2026-10-02. Run 8 (`gnn128-vpkeep-randboard`) vs run 6 (`gnn128-randboard`). Curves: W&B only (the checkpoints were
lost when Colab's usage limit cut the runtime, so there is no 2000-game evaluation).*

**Setup.** Run 6's recipe (GNN d128, 4 rounds, random boards, 6M steps, mixed opponents) with `--vp-anneal-frac 0`:
the VP-margin reward (0.5 × (own VP − mean opponent VP) / 10) stays on for the whole run instead of fading to 0 by
3M. **Also changed:** warmup + constant LR instead of run 6's decay to 5%; run 7 showed that schedule barely matters.

**Result** (in-training evaluations, the same 400 fixed games for every snapshot; ±1.7 points per point):

| | Win rate, mean from 4M | VP, mean from 4M | Final (6M) |
|---|---|---|---|
| Run 6 (fades by 3M) | 11.5% | 6.04 | 12.2% / 6.0 VP |
| Run 8 (never fades) | 11.6% | 5.97 | 11.7% / 6.06 VP |

The curves overlap from start to end; both level off at ~6 VP from ~3.5–4M steps.

**Implication.** Neither the VP fade nor the LR decay ([gnn-rounds-4-vs-7](gnn-rounds-4-vs-7.md)) nor more rounds
causes the plateau. Left: the global-token bottleneck (width helped, [gnn-width-d64-vs-d128](gnn-width-d64-vs-d128.md)),
the opponent mix (only 25% of games against the heuristic bots it's evaluated on), or simply too few steps.
