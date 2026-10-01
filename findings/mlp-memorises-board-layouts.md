# The flat MLP memorises the board layout

*2026-09-30. Runs 1, 3, 4 (`diag`, `selfplay-randboard`, `diag-randboard`). Chart: `plots/generalization.png`.*

**Finding.** The MLP doesn't generalise to new boards, and can't learn random boards in 6M steps either.

- Run 1 (trained on the beginner board) falls from 43.1% to 0.8% win rate on random boards (2000 games each).
- Run 4 (run 1's recipe on random boards) reaches only 0.5% and ~3.7 VP; run 3 (self-play, random boards) 1.6%.

**Why.** The MLP has separate weights for every vertex and hex, so it learns "settle on vertex 23" for one
layout rather than "settle next to good numbers".

**Implication.** Random boards need a network with weights shared across locations. See
[gnn-generalises-across-boards.md](gnn-generalises-across-boards.md).
