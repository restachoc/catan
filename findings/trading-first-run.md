# First run with trading: it offers constantly and learns slower

*2026-10-03. Run 10 (`gnn128-trade-randboard`) vs run 9 (`gnn128-cards-randboard`). Curves: W&B.*

**Setup.** Run 9's recipe (GNN d128, 4 rounds, random boards, mixed opponents, card counting) with player-to-player
trading: propose (at most 3 per turn), then terms (1:1, 2:1 or 1:2), opponents accept or decline, the proposer picks
an accepter. The built-in heuristic bots never offer and always decline. Plus the pair evaluation (two policy seats
vs two heuristic bots, chance 50%).

**Result** (in-training evaluations, the same 400 fixed games each):

| Steps | Run 9, 1 seat (win / VP) | Run 10, 1 seat | Run 10, pair (win / VP / trades per game) |
|---|---|---|---|
| 2.4M | 8.5% / 5.62 | 2.5% / 4.46 | 6.2% / 4.73 / 4.0 |
| 4.8M | 13.7% / 6.03 | 2.8% / 4.64 | 15.8% / 5.48 / 5.9 |
| 6.0M | ~11% / ~6.0 | 2.8% / 4.93 | 14.0% / 5.21 / 4.3 |

- `best.pt` over 2000 games (trading on): 1 seat 4.2% / 5.05 VP (chance 25%); pair 16.6% / 5.65 VP with 9.6
  trades per game between the two seats (chance 50%).
- Strategy mix (`plots/gnn128-trade-randboard_strategy.png`) is the same as the runs without trading.
- The learner proposes 35–60 trades per game, close to the cap of 3 every turn, and doesn't stop although the
  heuristic bots decline everything; ~3 trades per learner per game complete (against its own snapshots).
- Learning is much slower: at 6M steps it is where run 9 was at ~1.5M.
- The trades don't pay off yet: the pair wins 14–18% against a 50% chance level.

**Likely cause.** Trade decisions (propose, terms, answers, partner choice) fill every turn, spreading the samples
over fewer games and diluting the learning signal; the entropy bonus also rewards spreading over the 60 sets of
terms.

**Decision.** Trading is off by default (`--trading` to enable) until the network itself plays better. Options for
later: 1 proposal per turn, no entropy bonus on trade decisions, or switching trading on for a model trained without
it (possible since the action and observation sizes don't change).
