# A cost-matched transformer learns much slower than the GNN

*2026-10-06. Run 11 (`tf96-l3-randboard`) vs run 9 (`gnn128-cards-randboard`). Curves: W&B; strategy:
`plots/tf96-l3-randboard_strategy.png`.*

**Setup.** Run 9's recipe (random boards, mixed opponents, constant LR and VP reward, card counting, 6M steps) with
`--arch transformer --hidden 96 --layers 3` (0.28M params) instead of the GNN d128 L4 (1.20M). The transformer keeps
the GNN's encoders and per-location heads and replaces message passing with full self-attention over all 146 tokens,
with a learned per-head bias by graph distance (initialised at 0) and type embeddings, no positional embeddings.

**Cost match** (T4, `benchmarks/policy_cost.py`, est. PPO samples/s): GNN d128 L4 1,788; transformer d96 L3 1,630
and d64 L4 1,626 (0.91×); d64 L3 2,091; d128 L2 2,010; d64 L6 1,079. The real run averaged 1,432 samples/s (72 min).

**Result** (in-training: the same 400 fixed games; final: `best.pt`, 2000 games, chance 25%):

| Steps | Run 9, GNN (win / VP) | Run 11, transformer |
|---|---|---|
| 2M | 5.0% / 5.27 | 0.0% / 3.45 |
| 4M | 9.0% / 5.91 | 1.0% / 3.74 |
| 6M | ~10% / ~5.9 (4.9M) | 2.0% / 4.73 |
| `best.pt`, random / beginner board | (no checkpoint) | 1.9% / 4.90 VP, 2.3% / 5.00 VP |

- Far slower, but still rising at the end (3.7 → 4.7 VP over the last 2M steps).
- Generalises like the GNN: the same on the beginner board it never trained on.
- From ~5M its self-play strategy shifts to dev cards (55% of all players "OWS: dev cards") with fewer road
  builders, unlike the GNN runs whose mix was flat after ~1M.

**Likely cause.** The GNN has board locality built in (each node only hears its neighbours); the transformer must
learn it. With the distance bias at 0, every token starts out attending equally to all 146, averaging the board away.

**Next.** Initialise the distance bias as a locality prior (e.g. −1 per step of graph distance) so it starts
GNN-like and can learn to look further; same size and cost. Failing that, the hybrid (GNN layers plus a few
attention tokens).
