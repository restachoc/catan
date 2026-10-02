# More message-passing rounds (7 vs 4) add nothing

*2026-10-02. Run 7 (`gnn128-l7-randboard`) vs run 6 (`gnn128-randboard`). Charts: `plots/gnn_rounds_4_vs_7.png`,
`plots/generalization.png`, `plots/gnn128-l7-randboard_strategy.png`.*

**Setup.** Run 6's recipe (GNN d128, random boards, 6M steps, mixed opponents + VP shaping fading out by 3M) with
`--layers 7` instead of 4: 2.02M params instead of 1.17M (+72%). **Also changed:** the learning rate warms up over
0.5M samples and then stays constant, where run 6 decayed linearly to 5% at 6M. Two changes at once.

**Result** (`best.pt`, 2000 games vs 3 heuristic bots, chance = 25%; standard error ≈ 0.8 points):

| | Beginner board | Random boards |
|---|---|---|
| Run 6 (4 rounds, LR decay) | 11.7% / 6.3 VP | 13.4% / 6.3 VP |
| Run 7 (7 rounds, constant LR) | 11.7% / 6.2 VP | 15.0% / 6.3 VP |

- Same within noise: +1.6 points on random boards (about 1.5 standard errors), equal VP, equal beginner board.
- The in-training curves (same 400 games each) overlap almost point for point, and both level off at ~6.0–6.1 VP
  from ~3.5–4M steps.
- Same strategy mix as runs 5 and 6 ([bot-strategies](bot-strategies.md)).
- Cost: on a T4, ~1.5× run 6's time (~1070 samples/s, GPU at 100%; 91% of the time is the PPO update).

**Implications.**
- Depth isn't the bottleneck. Every round also passes through the global token, so after ~2 rounds every node
  already sees the whole board; extra rounds add parameters, not information.
- The LR decay didn't cause run 6's plateau either: run 7 levels off at the same VP with a constant LR. The plateau
  starts right after VP shaping ends (3M), which run 8 (VP shaping never fades) tests.
- Width did help (run 5 → 6), which still fits a bottleneck in the global token.
