# Evaluation curves were bumpy because every evaluation used different games

*2026-10-01. Engine and `evaluate()` changes; re-evaluation with `eval_curve.py`. Charts: `plots/all_runs_fixed.png`,
`plots/gnn_vs_mlp_randboard_fixed.png`.*

**Cause.** PPO evaluated each checkpoint on a new game set (seed 10000 + iter). Even with a fixed seed the games
weren't fixed: dice and steals shared one RNG, so one different steal shifted every later roll, and VecEnv seeded
new games from a step counter, so which games ran depended on when earlier ones ended. Policy sampling was
unseeded.

**Size of the effect.** The same checkpoint (run 1's `best.pt`) scores 37.5% on one set of 400 games and 46% on
another. Binomial noise alone at 400 games is ±1.2 pp at 6.5% and ±2.5 pp at 43%.

**Fix.** Separate dice and steal streams (the k-th roll is fixed whatever the players do), VecEnv seeds slot i's
j-th game from (seed, i, j), and `evaluate()` seeds a forked torch RNG. Evaluations are now exactly reproducible
and PPO uses the same games for every checkpoint. Re-evaluating old snapshots this way gives visibly smoother VP
curves; win rates near 0–10% stay jumpy.

**Still true.** A fixed game set has its own luck (a constant offset across checkpoints), and `best.pt` is the max
of ~30 noisy evaluations, so it's biased upward. Use 2000 fresh games for claims.
