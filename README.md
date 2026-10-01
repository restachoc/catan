# Catan RL

A fast Catan rules engine in Rust, a browser UI to play against bots, and PPO self-play training with a
board-structured graph network (GNN) that generalises across board layouts.

```
engine/catan-core   rules engine (Rust): bitboard state, flat 253-action space, observations, bots
engine/catan-py     PyO3 bindings: Game (UI) and parallel VecEnv (training)
python/catan_rl     PPO + AlphaZero trainers, evaluation, analysis charts, benchmarks
web/                FastAPI server + canvas UI (play and replays)
```

## Setup

```bash
curl https://sh.rustup.rs -sSf | sh            # Rust
python3 -m venv .venv
.venv/bin/pip install maturin numpy fastapi "uvicorn[standard]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
.venv/bin/maturin develop --release           # build the engine into the venv (rerun after Rust changes)
```

## Play

```bash
.venv/bin/uvicorn web.server:app               # http://localhost:8000
```

The "Opponents" menu lists `heuristic`, `random`, and `ppo:<run>` for every `runs/<run>/best.pt`.
Finished games are saved in `replays/` and can be scrubbed through with the Replay menu (arrow keys step).
Deep link: `/?replay=<name>&step=<n>`. Keys: space/r rolls, Enter ends the turn, Esc closes a picker.

## Train

```bash
nice -n 10 .venv/bin/python -m catan_rl.ppo --name v1          # PPO; defaults: 512 envs, 3x512 MLP, 1e9 steps
nice -n 10 .venv/bin/python -m catan_rl.ppo --name g1 --arch gnn --hidden 64 --layers 4 --random-board --device cuda --amp
nice -n 10 .venv/bin/python -m catan_rl.ppo --name v1 --resume runs/v1/latest.pt
nice -n 10 .venv/bin/python -m catan_rl.az --name az1          # AlphaZero-style MCTS self-play
.venv/bin/python -m catan_rl.evaluate runs/v1/best.pt --games 2000 [--random-board] [--replays 5]
```

Analysis (charts go to `plots/`):

```bash
.venv/bin/python -m catan_rl.plot_runs <run> <run> ...         # training curves side by side
.venv/bin/python -m catan_rl.generalization <run> ...          # fixed vs random boards
.venv/bin/python -m catan_rl.strategy <run>                    # strategy mix over training
```

GPU runs on Colab take one notebook cell (the script builds the engine, trains and zips `runs/<run>/`):

```
!git clone -q https://github.com/restachoc/catan 2>/dev/null; bash catan/scripts/colab.sh ppo --name <run> --device cuda --amp [flags]
from google.colab import files; files.download("/content/<run>.zip")
```

Metrics go to `runs/<name>/metrics.csv`. `eval_wr_heuristic` is the win rate with the policy in one seat
and three heuristic bots in the others (0.25 is chance level).

## Tests and benchmarks

```bash
cd engine && cargo test -p catan-core --release                 # add `-- --ignored` for the 100k-game stress test
cd engine && cargo bench -p catan-core                          # engine throughput
.venv/bin/python -m catan_rl.smoke                              # bindings end to end
.venv/bin/python -m catan_rl.bench_compute [--device cuda]      # RL throughput / time projections
```

## Roadmap

- v1: fixed beginner board, no player trading (current)
- v2: random boards (`--random-board`; the GNN is the first network that generalises across layouts)
- v3: bots accept or reject your trade offers
- v4: bots propose trades
