# What strategies the bots learn

*2026-09-30. `strategy.py` on runs 1–3 (300 self-play games per snapshot). Charts: `plots/<run>_strategy.png`.*

Players are classified by where they spent resource cards after setup (road 2, extra settlement 4, city 5,
dev card 3): road builder ≥60% on expansion, OWS ≤40% (split into dev cards vs cities), balanced in between.

- Near-random early play spends ~45–50% on roads; trained bots move to ~30% roads.
- "OWS: dev cards" becomes the largest group (~40% of players, 44–49% of winners on the fixed board).
- Cities without heavy dev-card buying almost never happens (1–2%).
- Extra settlements get only 11–15% of spending: the bots barely expand beyond their two starting settlements,
  a clear weakness.
- Random-board self-play (run 3) ends more mixed: ~40% balanced, ~40% OWS-dev, ~20% road builders among winners.
