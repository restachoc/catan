# The GNN learns random boards where the MLP can't

*2026-10-01. Run 5 (`gnn-randboard`) vs run 4 (`diag-randboard`). Charts: `plots/gnn_vs_mlp_randboard_fixed.png`,
`plots/generalization.png`.*

**Setup.** Run 4's recipe (6M steps, mixed opponents, VP shaping, random boards) with `--arch gnn --hidden 64
--layers 4` (0.30M params) instead of the 2×256 MLP. Also changed: trained on a Colab T4 with fp16 autocast
instead of CPU fp32; neither should change what is learned.

**Result** (`best.pt`, 2000 games vs 3 heuristic bots, chance = 25%):

| | Beginner board | Random boards |
|---|---|---|
| Run 4 (MLP) | 0.1% / 3.4 VP | 0.5% / 3.7 VP |
| Run 5 (GNN) | 11.2% / 6.4 VP | 8.7% / 5.7 VP |

- It transfers to the beginner board it never trained on.
- Still far below chance and below run 1's 43% on its single board (an easier task: memorising one layout).
- Not a proven plateau. On fixed evaluation games, VP rises fast to ~5 by 2M steps, then keeps climbing slowly
  (~5.0 → 5.5 from 4M to 6M). The slowdown coincides with VP shaping annealing to 0 at 3M and the learning
  rate decaying to 5% at 6M.

**Open questions.** Does it keep improving with 20–30M steps and stretched schedules? Does it match run 1 on the
fixed board (which would show the architecture is adequate)? Is the 64-dim global token, which scores all 73
non-board actions and the value, a bottleneck?
