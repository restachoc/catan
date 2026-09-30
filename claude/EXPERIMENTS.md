# Experiments

Training runs so far, what they showed, and the lessons that still hold. Curated, not a log: when a
lesson is overturned, rewrite it. Linked from [CLAUDE.md](../CLAUDE.md).

All runs: 4 players, 2×256 MLP, evaluated with the policy in one seat vs 3 heuristic bots (chance = 25%).
Charts: `plots/all_runs.png` (training curves), `plots/generalization.png` (fixed vs random boards),
`plots/<run>_strategy.png` (strategy mix over training).

## Runs

"Final" columns: the run's `best.pt`, 2000 games per cell (`runs/<run>/generalization.json`).

| Run dir | Name in charts | Setup (vs the 6M-step diagnostic PPO recipe) | Fixed board: wr / VP | Random boards: wr / VP |
|---|---|---|---|---|
| `diag` | Run 1 | PPO, mixed opponents (25% heuristic, 25% self-play, 50% league), VP shaping 0.5, fixed board | **43.1% / 8.2** | 0.8% / 3.8 |
| `selfplay` | Run 2 | PPO, pure self-play, win/loss only (`--frac-heuristic 0 --frac-selfplay 1 --vp-coef 0`) | 1.5% / 4.7 | 0.5% / 3.4 |
| `selfplay-randboard` | Run 3 | Run 2 + `--random-board` | 1.2% / 3.8 | 1.6% / 4.1 |
| `diag-randboard` | Run 4 | Run 1 + `--random-board` | 0.1% / 3.4 | 0.5% / 3.7 |
| `az1` | AlphaZero az1 | AlphaZero, 64 sims, 256 games, 3M samples (~1.6 h), pure self-play, win/loss | 0.0% / 3.1 (raw policy) | 0.1% / 2.8 |

az1 with search (in-training eval, 200 games): 0–1.5% win rate, VP 3.0–3.6, no upward trend.

## Lessons

**Reward and opponents (PPO)**
- Mixed opponents + VP shaping (run 1) learns fast: win rate passes 25% at ~3M steps and reaches ~43% by 6M.
- Pure self-play with win/loss only (run 2) learns slowly. VP vs the heuristic bot rises 2.5 → 4.5 by ~3M
  steps, then stays flat. The run changed two variables at once (opponents *and* reward), so which one
  matters is still open; the clean test is pure self-play *with* VP shaping.
- The self-play game length fell from ~840 to ~415 steps during run 2, so its bots did get more efficient
  against each other; that improvement barely transferred to beating the heuristic bot.
- Why sparse reward hurts: four equally weak players plus dice means the winner is mostly luck, so each
  update's direction is noisy (advantages are normalised, so the update size is unchanged).

**Board generalisation (the big one)**
- The flat MLP does not generalise: run 1 falls from 43.1% to 0.8% on random boards.
- It also cannot *learn* random boards in 6M steps: run 4 (run 1's recipe on random boards) reaches only
  0.5% and ~3.5 VP. Run 3 (self-play on random boards) is similarly weak.
- Diagnosis: the MLP has separate weights for every vertex/hex, so it memorises layouts. This motivates a
  board-structured network (GNN/transformer over hexes, vertices and edges with shared weights).

**Strategies (from `strategy.py`)**
- Classification by where resource cards were spent after setup (road 2, extra settlement 4, city 5,
  dev card 3): road builder ≥60% on expansion, OWS ≤40% (split into dev cards vs cities), balanced in between.
- In every run, near-random early play spends ~45–50% on roads; trained bots move to ~30% roads, and
  "OWS: dev cards" becomes the largest group (~40% of players, 44–49% of winners on the fixed board).
- Cities without heavy dev-card buying almost never happens (1–2%). Extra settlements get only 11–15% of
  spending: the bots barely expand beyond their two starting settlements, a clear weakness.
- Random-board self-play ends with a more mixed population (~40% balanced, ~40% OWS-dev, ~20% road
  builders among winners).

**AlphaZero (az1)**
- It failed because the value network memorised games: training value loss 0.05–0.12, but on 60 fresh
  held-out games the loss was 2.24 (uniform guessing = 1.39) with 82% average confidence. The search is
  then guided by confidently wrong values and cannot improve the policy.
- Cause: ~350 samples per game share one outcome, a 200k replay holds only ~500 games, and the MLP can
  tell games apart by their board.
- Planned fixes (not yet implemented): keep ~1 in 8 positions per game; value target = 50% game result +
  50% search root value; label smoothing / stronger weight decay on the value head; held-out value loss
  as a standard metric. A board-structured network should also memorise less.
- Cost: ~500 searched moves/s (64 network evaluations each) vs ~7k decisions/s for PPO, so az1 saw
  only ~7.5k games.

**Compute**
- 400-game evaluations are noisy (±3–5 pp); 2000 games for final numbers.
- The PPO learning rate decays to ~0 at `total_steps`; the final third of a 6M run is nearly frozen, so a
  "plateau" there doesn't prove the method has stalled. Continue with `--resume` to check.
