# Mixed opponents with VP shaping learn far faster than pure self-play with win/loss

*2026-09-30. Run 1 (`diag`) vs run 2 (`selfplay`), fixed board. Chart: `plots/all_runs.png`.*

- Run 1 (25% heuristic, 25% self-play, 50% league; VP shaping 0.5): win rate passes 25% at ~3M steps and
  reaches ~43% by 6M.
- Run 2 (pure self-play, win/loss only): VP vs the heuristic bot rises 2.5 → 4.5 by ~3M steps, then stays flat.
  Its self-play games got shorter (~840 → ~415 steps), so the bots improved against each other, but that
  barely transferred to beating the heuristic bot.

**Caveat.** Two variables changed at once (opponents and reward), so which one matters is open. The clean test
is pure self-play *with* VP shaping.

**Why sparse reward hurts.** With four equally weak players plus dice, the winner is mostly luck, so each
update's direction is noisy. Advantages are normalised per minibatch, so the update size is unchanged: it's
noise, not small steps, and a larger learning rate isn't the fix.
