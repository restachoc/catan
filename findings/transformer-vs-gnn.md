# A cost-matched transformer learns slower than the GNN; position embeddings close most of the gap

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

**Run 12 (`tf96-l3-pos-randboard`, same size): fixed.** Changes vs run 11: a learned position embedding per token
slot instead of the type embeddings (the topology is fixed; only what lies on it changes), the distance bias
initialised as a multi-scale locality prior (−slope × distance, slopes 2, 1, 0.5, 0.25 per head; the global token
unpenalised), and a final LayerNorm. Its Colab session died at 4.6M steps (55 min), so no checkpoints.

| Steps | Run 9 GNN | Run 11 transformer | Run 12 transformer + positions |
|---|---|---|---|
| 2M | 5.0% / 5.27 | 0.0% / 3.45 | 4.7% / 4.78 |
| 4M | 9.0% / 5.91 | 1.0% / 3.74 | 8.0% / 5.42 |
| mean 4–5.1M | 10.4% / 5.97 | 0.6% / 3.87 | 7.0% / 5.41 |

- Most of run 11's gap closed (+1.5 VP at 4–5M), but still ~0.5 VP below the GNN at equal cost, and still rising
  when it died. Three changes at once vs run 11, so which one mattered is unknown.
- The GNN remains the better network per GPU-minute at 6M steps; a longer run would show whether the transformer
  overtakes it.
