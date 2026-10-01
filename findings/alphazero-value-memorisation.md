# AlphaZero az1 failed because the value net memorised games

*2026-09-30. Run `az1` (64 sims, 256 games, 3M samples, ~1.6 h, pure self-play, win/loss, 2×256 MLP).*

**Result.** Raw policy 0.0% / 3.1 VP on the beginner board; with search 0–1.5% in training evaluations, no trend.

**Cause.** Training value loss was 0.05–0.12, but on 60 fresh held-out games it was 2.24 (uniform guessing =
ln 4 ≈ 1.39) with 82% average confidence. Search is then guided by confidently wrong values and can't improve
the policy. One game yields ~350 samples with the same outcome, a 200k replay holds only ~500 games, and the MLP
can tell games apart by their board.

**Fixes (not implemented).** Keep ~1 in 8 positions per game; value target 50% game result + 50% search root
value; label smoothing or stronger weight decay on the value head; held-out value loss as a standard metric.
A board-structured network should also memorise less.

**Cost.** ~500 searched moves/s on this CPU (64 network evaluations each) vs ~7k decisions/s for PPO, so az1
saw only ~7.5k games.
