# A wider GNN (d128) is better on random boards, not on the beginner board

*2026-10-01. Run 6 (`gnn128-randboard`) vs run 5 (`gnn-randboard`). Charts: `plots/gnn_d64_vs_d128.png`,
`plots/generalization.png`.*

**Setup.** Run 5's recipe (GNN, 4 rounds, random boards, 6M steps, mixed opponents + VP shaping, Colab T4 with
`--amp`) with `--hidden 128` instead of 64: 1.15M params instead of 0.30M. The only other difference is that dice
now come from their own random stream (same distribution, different games), which shouldn't matter.

**Result** (`best.pt`, 2000 games vs 3 heuristic bots, chance = 25%; standard error ≈ 0.7 points):

| | Beginner board | Random boards |
|---|---|---|
| Run 5 (GNN d64) | 11.2% / 6.4 VP | 8.7% / 5.7 VP |
| Run 6 (GNN d128) | 11.7% / 6.3 VP | 13.4% / 6.3 VP |

- On random boards (what it trained on), d128 is clearly better: +4.7 points win rate, +0.6 VP.
- On the beginner board, which neither run trained on, the two are the same.
- On the fixed evaluation games, run 6 pulls ahead after ~2.5M steps and is still rising slowly at 6M (VP ~6.0
  from 4M on), with the same schedule caveats as run 5: VP shaping is gone at 3M and the learning rate is at 5%
  by 6M.
- Cost: about 2× the time per step ([board-network-speed](board-network-speed.md)), ~70 min for 6M steps on a T4.

**Implication.** Width helps, so the 64-dim global token may have been a bottleneck. Both runs are still far
below chance; longer training with stretched schedules is the open question, and d128 is the better base for it.
