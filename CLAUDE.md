# CLAUDE.md

Guidance for Claude Code sessions in this repo. **Keep this file clean and current:** when something
here stops being true (a gotcha gets fixed, a milestone is done, a file moves), update or delete it in
the same change. Don't append history; this is a description of the present, not a changelog.

## Claude files

More Claude-facing notes live in `claude/*.md`. Every one of them must be linked here:

| File | Purpose |
|---|---|
| [claude/ONGOING.md](claude/ONGOING.md) | **Read first after a context clear.** What's in flight, open decisions, next steps. Update it as work progresses; remove items when done. |
| [claude/EXPERIMENTS.md](claude/EXPERIMENTS.md) | Training runs so far, their results, and the lessons that still hold. |

**If you find a `.md` file in `claude/` (or elsewhere in the repo, outside `.venv/` and `engine/target/`)
that isn't linked from this file or the README, warn the owner.** Either link it or delete it.

## Working with the owner

- When they say "answer shortly", keep it to a few lines.
- Long jobs (training runs, big evaluations) run in the background. **Don't poll them**; wait for the
  completion notification and then report. Short status lines are fine when a notification arrives.
- For experiments: say exactly what changed versus the baseline run. Point out when two variables
  changed at once, since that makes the result ambiguous.
- Charts go to `plots/` (one file per comparison, clean names). Look at every rendered chart before
  reporting it (label collisions, colours that mean different things in different panels).
- Ask before starting multi-hour runs unless the owner asked for one.
- Don't install tools or system software on the owner's machine without asking; they prefer to do it themselves.
- **Never run anything that can incur costs.** Remote compute stays on prepaid/free Colab (free tier or
  already-bought compute units); no Colab Enterprise, GCP/Vertex, or other billed cloud services, and never
  buy units or upgrade a plan.
- **Always shut down remote sessions when a job finishes** (Colab: `runtime.unassign()`, see "Remote GPU
  runs"; closing the tab is not enough). An idle GPU runtime keeps burning compute units. Copy results off first.
- The owner uses this machine interactively. Heavy jobs make the desktop stutter even when niced (memory
  bandwidth), so keep benchmarks and side jobs to about half the cores.

## Project goal

Settlers of Catan with three parts:
1. **Very fast rules engine** (Rust). Speed matters because RL training throughput depends on it.
2. **Good-looking web UI** that renders fast, for playing against bots and watching replays.
3. **RL bot** that should eventually beat the owner at Catan.

The bot is built in versions of increasing difficulty:

| Version | Board | Trading | Status |
|---|---|---|---|
| v1 | fixed beginner board | none (bank/port only) | **current**: PPO diagnostic run `diag` beats the heuristic bot 43% of the time (chance = 25%); long run not started |
| v2 | random boards | none | the flat MLP fails here (runs 3 and 4); the GNN (`--arch gnn`, run 5) reaches 8.7% after 6M steps on a Colab T4, still below chance; next: longer/bigger GNN runs |
| v3 | either | bots accept/reject the human's offers | not started |
| v4 | either | bots propose structured trades | not started |

Default game: 4 players (the human + 3 bots), configurable from 2 to 4.

## Layout

```
engine/                   Cargo workspace (release profile: lto=fat, codegen-units=1)
  catan-core/src/
    topology.rs           static board graph: 19 hexes, 54 vertices, 72 edges, 30 coast edges (LazyLock TOPO)
    board.rs              per-game layout: tile resources, numbers, ports; beginner + random (no adjacent 6/8)
    state.rs              State (Copy, no heap) + all rules: legal_mask(), step(), try_step()
    actions.rs            flat action-space constants, Mask bitset helpers, action_name()
    obs.rs                actor-relative flat f32 observation (OBS_SIZE)
    bots.rs               random_action, heuristic_action (greedy rules; see the doc comments there)
    mcts.rs               AlphaZero MCTS (PUCT), sampled chance nodes, determinization of hidden dev cards
    stats.rs              end-of-game per-player stats (STAT_NAMES) for strategy analysis
    view.rs               serde views for the UI (board_view, state_view with hidden info per viewer)
    rng.rs                wyrand PRNG (deterministic)
  catan-core/tests/rules.rs   rule tests, invariant stress tests, determinism/replay, MCTS tests
  catan-core/benches/playout.rs   throughput benchmark (std::time; harness = false)
  catan-py/src/lib.rs     PyO3 module `catan_rl._engine`: Game, VecEnv, AzPool, action_names, bot_tournament
python/catan_rl/
  __init__.py             re-exports + ACTIONS offset dict (derived from action names)
  model.py                PolicyNet (MLP), GraphPolicyNet (GNN), AZNet (AlphaZero MLP); save/load; PolicyBot
  graph.py                board graph (from board_json), typed GNN layer, per-location action mapping
  ppo.py                  PPO trainer (Config dataclass = CLI flags)
  az.py                   AlphaZero trainer: AzPool self-play (Rust) + replay buffer + AZNet
  evaluate.py             evaluate() vs bots or checkpoints (--random-board); replay export
  strategy.py             strategy mix of a run's snapshots in self-play -> plots/<run>_strategy.png
  eval_curve.py           every snapshot of a run on the same eval games -> runs/<run>/eval_curve.csv (plot_runs --fixed)
  plot_runs.py            eval curves of several runs side by side
  generalization.py       each run's best.pt on the fixed vs random boards -> plots/generalization.png
  bench_compute.py        rollout/train throughput and time projections
  smoke.py                end-to-end bindings check
scripts/colab.sh          one-command Colab job: build engine, run a catan_rl module, zip runs/<run>
benchmarks/arch_speed.py  speed of candidate board networks (GNN, transformer, hybrid) vs the MLP; reuses graph.py; results/ gitignored
web/server.py             FastAPI + WebSocket; owns the Game; registers bots (incl. runs/*/best.pt)
web/static/               index.html, style.css, board.js (canvas renderer), ui.js (session + panels + replays)
pyproject.toml            maturin config (python-source = python, module = catan_rl._engine)
runs/, replays/, plots/   training output / saved games / charts (all gitignored)
claude/                   Claude notes (see "Claude files" above)
```

## Commands

Always use the venv (`.venv/bin/...`). Rust is installed via rustup; run `source ~/.cargo/env` first in fresh shells.

```bash
.venv/bin/maturin develop --release                       # REQUIRED after any Rust change (see gotchas)
cd engine && cargo test -p catan-core --release           # 18 tests, ~1 s
cd engine && cargo test -p catan-core --release -- --ignored   # 100k-game invariant stress test, ~20 s
cd engine && cargo bench -p catan-core                    # engine throughput
.venv/bin/python -m catan_rl.smoke                        # bindings end to end
.venv/bin/python -m catan_rl.bench_compute                # NN throughput and training-time projections
.venv/bin/python benchmarks/arch_speed.py [--quick] [--csv benchmarks/results/arch_speed.csv]   # niced, half the cores
.venv/bin/uvicorn web.server:app --port 8765              # UI at http://localhost:8765

# Long runs: always under nice, so interactive jobs get priority (see "CPU sharing")
nice -n 10 .venv/bin/python -m catan_rl.ppo --name <run> [flags]   # flags mirror ppo.Config fields
nice -n 10 .venv/bin/python -m catan_rl.az --name <run> [flags]    # flags mirror az.Config fields

.venv/bin/python -m catan_rl.evaluate runs/<run>/best.pt --games 2000 [--random-board] [--replays N]
.venv/bin/python -m catan_rl.strategy <run> [--games 300]   # cached in runs/<run>/strategy.csv
.venv/bin/python -m catan_rl.generalization <run> ... [--labels ...]   # cached in runs/<run>/generalization.json
.venv/bin/python -m catan_rl.eval_curve <run> ...   # then plot_runs --fixed: smooth curves on fixed games
.venv/bin/python -m catan_rl.plot_runs <run> ... [--labels ...] [--out plots/<name>.png]   # default plots/<a>_vs_<b>.png
```

The 6M-step diagnostic PPO recipe used for all comparisons (~15 min):
`--num-envs 256 --rollout 128 --total-steps 6e6 --hidden 256 --layers 2 --snapshot-every 10 --eval-every 10 --eval-games 400`.

## Repository and machines

- Remote: public GitHub repo `restachoc/catan` (Colab can `git clone https://github.com/restachoc/catan` without credentials) (`origin`, branch `master`). Push over HTTPS; the `gh` login
  (account `restachoc`) is the credential helper, set in this repo's local git config only. SSH won't work:
  this machine's SSH key belongs to a different GitHub account. Push when the owner asks.
- This machine: 14 cores, 15 GB RAM, AVX2 only, no GPU. The owner has a separate GPU machine that pulls from
  `origin`. Setup there: rustup, a venv, `maturin develop --release -E train`, then the smoke test.
  `runs/`, `replays/` and `plots/` are gitignored, so checkpoints don't travel with the repo.

## Remote GPU runs (Colab): the default for GPU work

GPU jobs run on free Colab through the `colab-mcp` tools, with as little notebook code as possible. The
runtime type (T4 GPU) is the owner's menu choice; the tools can't change it.
After `runtime.unassign()` the notebook tools disappear; `open_colab_browser_connection` brings them back,
sometimes in a fresh notebook on a CPU runtime. **Check `nvidia-smi` before launching**, and ask the owner to
switch to T4 if there's no GPU.

1. One cell, from `/content`:
   ```
   !git clone -q https://github.com/restachoc/catan 2>/dev/null; bash catan/scripts/colab.sh ppo --name <run> --device cuda --amp [flags]
   from google.colab import files; files.download("/content/<run>.zip")
   ```
   `scripts/colab.sh <module> <args>` pulls master, builds the engine, runs `python -m catan_rl.<module>`,
   and zips `runs/<run>/`. So **commit and push before launching**: Colab runs what's on `origin/master`.
2. The download lands in `~/Downloads/<run>.zip` on this machine (the browser runs here), sometimes a few
   seconds after the cell finishes. `unzip -qo ~/Downloads/<run>.zip` in the repo root gives `runs/<run>/`.
3. Delete the runtime right away: a cell with `from google.colab import runtime; runtime.unassign()`.

Notes: free runtimes have 2 vCPUs, 12 GB RAM, a T4 with 15 GB, and disconnect after ~12 h or when the browser
tab idles too long, so keep single jobs to a few hours. The job runs inside the cell; `run_code_cell`
moves to the background after 2 min and notifies on completion. Don't poll. Measured: GNN d64 L4 with `--amp` and
the diagnostic recipe trains at ~3.3–3.7k samples/s before the league starts and ~2.7–3.2k/s after (6M steps
≈ 35 min, plus ~2 min setup).
End-to-end d64 vs d128 (L4, `--amp`, CUDA-graph rollouts, same recipe, 2026-10-01): d64 3.5k / 3.0k samples/s
(before / with league), d128 1.8k / 1.5k, so d128 is ~1.95× slower. The PPO update doubles (8.3 → 16.4 s per
iteration) and rollout steps take ~1.5× as long (8.8 → 13.4 ms).

## Architecture invariants (don't break these)

- **The UI never implements rules.** The server sends `state.legal` (action ids); the UI only maps ids to
  clickable spots and buttons. Any rule change lives in `state.rs` only.
- **`State` is `Copy` and allocation-free.** Bitboards: `u64` per player for settlements/cities, `u128` per
  player for roads. This keeps stepping fast and makes cloning for search free.
- **Determinism.** A game is fully defined by `(seed, config, action list)`, and replays depend on this. Chance
  comes from independent streams derived from the seed: board and dev deck at setup, `State.dice`, and
  `State.steal_rng`. So the k-th roll and the deck order never depend on the actions (common random numbers:
  evaluations replay the same games whatever the policy does). Never draw dice from another stream or vice
  versa. Search reseeds with `reseed_chance`. VecEnv seeds slot i's j-th game from `(seed, i, j)`, not from
  timing. Bots use their own `Rng`.
- **`step()` checks legality only via `debug_assert`.** Use `try_step()` for untrusted input. `Game.step` in
  the bindings uses `try_step`, and `VecEnv.step` validates every action before stepping.
- **Every non-terminal state has at least one legal action.** The stress test asserts this. Keep phases
  such that a mask can never be empty (e.g. Road Building is only offered when a road can be placed).

## Rules as implemented (decisions worth knowing)

- Setup is snake order; the second settlement grants its adjacent resources.
- Dev cards can be played before rolling (official rule), one per turn, never on the turn bought.
- VP cards count immediately and are hidden from opponents. A player wins only on their own turn: the
  check runs after each action for `cur` and at the start of each turn.
- A 7 makes every player with more than 7 cards discard half (floor), **one card per action**, in turn
  order from the current player. The robber must move to a different hex. The steal victim is chosen
  automatically when there is exactly one candidate; the `Steal` phase appears only with 2+ candidates.
- Bank shortage: if the bank can't cover all demand for a resource, it pays only if exactly one player is owed.
- Year of Plenty offers only pairs the bank can cover.
- Longest Road (LR) is ≥5, broken by opponent settlements, and the holder keeps it on ties. If the holder
  drops and others tie, nobody holds it. Largest Army (LA) is ≥3 knights and strictly more than the holder.
- `max_turns` (default 500) ends the game as a draw (`winner = -1`).
- No player-to-player trading yet. Its action ids don't exist yet; adding them changes `N_ACTIONS` (see below).
- The heuristic bot (`bots.rs`) is greedy and rule-based: city > settlement > useful dev card > road toward a
  new spot > one-card bank trade > buy dev card > end turn. It never blocks leaders or plans ahead, but it
  plays any board equally well, which makes it a fair yardstick for random boards.

## Action space and observation

- Flat action space, `N_ACTIONS = 253`. Offsets are in `actions.rs`: ROLL 0, END_TURN 1, SETTLE 2+v,
  CITY 56+v, ROAD 110+e, BUY_DEV 182, PLAY_KNIGHT 183, PLAY_ROAD_BUILDING 184, MONOPOLY 185+r,
  YOP 190+pair (15 unordered pairs), MOVE_ROBBER 205+h, STEAL 224+k (k = seats after the current
  player), DISCARD 228+r, TRADE 233 + give*4 + (get index skipping give).
- Resources are indexed 0 wood, 1 brick, 2 wool, 3 grain, 4 ore, 5 desert. Port type 5 = 3:1.
- The observation (`OBS_SIZE = 1292`) is **actor-relative**: seat 0 is always the acting player. Blocks: 19 hexes
  × 8, 54 vertices × 14, 72 edges × 4, 4 players × 11, own hand/dev cards/ratios 21, globals 31. Opponents are
  encoded with public info only (card counts, dev-card counts, not contents).
- Python derives offsets from action names (`catan_rl.ACTIONS`) and the web UI fetches them from `/api/meta`.
  **But `ui.js` duplicates the YOP pair order and the `tradeId` formula**, so update both if the layout changes.

## Networks

- Both trainers default to a flat MLP: 1292 → [Linear → LayerNorm → ReLU] × layers → heads. All comparison runs use
  2×256 (~0.46M params). PolicyNet (PPO) has a scalar value head; AZNet has a 4-way value head (win
  probability per seat, relative to the player to move).
- **Known limitation:** the MLP has no notion of board structure (no weight sharing between vertices/hexes), so
  it memorises one layout and does not transfer to random boards.
- `GraphPolicyNet` (`ppo --arch gnn --hidden <d> --layers <rounds>`) is the replacement: typed message passing
  over hexes, vertices and edges plus a global token, shared weights, no positional embeddings. Vertex/edge/hex
  embeddings score settle+city/road/robber; the global token scores the rest and the value. Each node also gets
  its own legal-action bits from the mask and static coast features. Not trained beyond a smoke test yet.
  Checkpoints store `cfg.kind` (`mlp` implied when absent, `gnn`, `az`); `model.load` dispatches on it.
- **Board networks are too slow for this CPU.** They cost 20–30× the MLP's FLOPs per sample (145 nodes × d²
  per layer). Estimated PPO rate (network only, samples/s; CPU = half the cores here, T4 = free Colab):

  | Network | CPU | T4 | T4 + `torch.compile` |
  |---|---|---|---|
  | MLP 2×256 | ~18k | 153k | |
  | GNN d64 L4 | ~250 | 2.4k | 3.3k |
  | GNN d128 L4 | | 1.2k | 1.4k |
  | transformer / hybrid d64 L4 | 60–75 | 0.7–0.9k | |
  | d128 transformer / hybrid | 17–60 | 0.3–0.5k | |

  Full T4 tables: `benchmarks/results/arch_speed_t4.csv` and `batch_sweep_t4.csv` (local only). A 6M-step
  diagnostic with GNN d64 L4 is ~30–40 min of network time on a T4 versus ~7 h on this CPU.
- **Batch size on the T4** (trainer nets, PPO estimate with 4 epochs and the engine included):
  - MLP 2×256 saturates at ~4096 envs for inference (650k/s) and minibatch 8192 for training (1.1M/s):
    ~140k samples/s, versus ~82k/s at the diagnostic recipe's 256 envs.
  - GNN d64 L4 is already saturated at 256 envs and minibatch 1024 (~27k/s inference, ~10k/s training,
    ~2.3k/s end to end), so batch size doesn't help it. d128 being only 2× slower than d64 suggests memory
    bandwidth, not FLOPs, is the limit. fp16 autocast gives ~1.5× (3.5k/s); `torch.compile` gave ~1.4×.
    fp16 needs the masking constant `NEG_INF = -1e9` applied in fp32 (it overflows fp16).
  - Training memory: GNN ~0.75 GB per 1k minibatch (16k fits in 15 GB, 32k OOMs); the transformer d128 OOMs
    at 4096-sample chunks.
  - Free Colab has 2 vCPUs: the engine does ~500k steps/s there up to 4096 envs, then drops (230k/s at 16k).
  - In `ppo.py`: `--device cuda --amp`. For GNN comparisons keep the diagnostic recipe's batch sizes (they
    already saturate the T4), so only the network changes. MLP runs on the GPU: `--num-envs 2048 --minibatch 8192`.
- **Changing `OBS_SIZE` or `N_ACTIONS` invalidates every checkpoint**, and `web/server.py` loads all
  `runs/*/best.pt` at startup. Start a new run name, and delete or move incompatible runs.

## PPO training (ppo.py)

- Env groups by index: `frac_heuristic` (learner in one seat vs 3 Rust heuristic bots played inside the
  engine), `frac_selfplay` (learner in all seats), and the rest league (learner in one seat vs frozen
  snapshots from `runs/<name>/pool/`; the learner itself while the pool is empty). Pure self-play is
  `--frac-heuristic 0 --frac-selfplay 1`.
- League timeline: league envs start out as self-play (the learner plays every seat while the pool is empty).
  The first snapshot is saved at iteration `snapshot_every`; from then on up to `active_opponents` snapshots
  play the other league seats, refreshed every `snapshot_every` iterations, each env switching at its next game
  end. Opponent moves aren't training data, so samples per iteration drop ~37% once the league starts.
- On the GPU, rollout actors (learner and opponents) are `GraphedPolicy` wrappers: CUDA-graph replay padded to
  bucket sizes. An eager GNN `act()` costs ~8 ms at any batch size (host-side launches), so four calls per step
  made the league rollout 3–4× slower. Graphs read the live weights; in-place optimizer steps need no recapture.
- Samples are tagged (env, seat). GAE runs per sequence and bootstraps from the same seat's next decision.
  Unfinished tails are **carried into the next rollout** rather than bootstrapped. League seat assignment
  only changes at game end; changing it mid-game would orphan carried samples.
- Reward is terminal: +1 win, −1/(n−1) loss, 0 draw, plus `vp_coef` × (own VP − mean opponent VP)/10,
  annealed to 0 over `vp_anneal_frac` of training. `--vp-coef 0` gives pure win/loss.
- The learning rate decays linearly to 5% at `total_steps`, so the last third of a short run barely updates.
  Take that into account before calling a plateau, and use `--resume ... --total-steps <larger>` to continue.
- PPO normalises advantages per minibatch, so reward scale does not change the update size; sparse rewards
  mean noisier updates, not smaller ones. Raising the learning rate is not the fix.
- Outputs go to `runs/<name>/`: `metrics.csv`, `latest.pt` (includes optimizer state, used by `--resume`),
  `best.pt` (best eval win rate vs heuristic), `pool/`, and `config.json`.
- Throughput on this machine (14 cores, no GPU): ~6.5–7k samples/s for 2×256 with league inference,
  ~10k/s for pure self-play, ~7.5k/s for 3×512 before league overhead. The network is the bottleneck; the
  engine does 13M raw steps/s per core.

## AlphaZero (az.py, mcts.rs)

- `AzPool` searches all games in lockstep: `select()` returns one leaf per searching game, Python
  evaluates the batch with `AZNet.evaluate`, `expand()` backs it up; when `select()` returns 0,
  `advance()` plays the chosen moves. Forced moves (one legal action) are played without search and
  are not training samples.
- Chance: each edge traversal replays the action with a fresh RNG seed; children are keyed by
  `state_key` (a position hash without the RNG), so dice, dev draws and steals branch naturally.
- Hidden info: `determinize` reshuffles opponents' unplayed dev cards and the deck before each
  search. Resource hands are treated as known (approximates card counting).
- Value targets are the winner one-hot (draw = uniform). Samples carry the true legal mask for the policy
  loss (masking to visited moves only would leave unvisited legal moves unconstrained).
- **Value memorisation is the main failure mode:** one game yields ~150–350 samples with the same outcome,
  and the MLP can recognise a game from its board, so it learns "this game → seat 2 wins". Always measure the
  value loss on fresh held-out games (uniform guessing = ln 4 ≈ 1.39); the training loss is meaningless.
  See EXPERIMENTS.md for the planned fixes.
- Throughput on this CPU (64 sims, 256 games): ~500 searched moves/s with 2×256, ~300/s with 3×512;
  about 60% of the time is the network, 40% the Rust search.

## CPU sharing

- Every heavy job (training, evaluation, strategy analysis) spawns ~14 torch threads plus ~14 rayon
  threads. Two such jobs at full width thrash: training fell from 7.3k to 1.2k samples/s while a strategy
  analysis ran next to it.
- Run long jobs under `nice -n 10` (or `renice -n 10 -p <pid>` for a running one; no root needed). When
  running a second job alongside, cap it: `RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4` and
  `torch.set_num_threads(4)`. That combination slowed a running training by only ~10%.

## Gotchas

- **Benchmarks can freeze the desktop:** a transformer over ~146 tokens at batch 4096 materialises several GB of
  attention and swaps (RAM is 15 GB). Train in micro-batches, leave cores free, and run under `nice`
  (`arch_speed.py` does all three by default).

- **Stale extension:** after editing Rust, Python keeps importing the old `_engine` until you rerun
  `maturin develop --release`. There is no error, just old behaviour. Rebuilding while a training runs is
  safe (the running process keeps the library it loaded).
- **`pkill -f` / `pgrep -f` self-match:** patterns like `"catan_rl.ppo"` also match the shell running the
  command, which kills it or makes `until ! pgrep` loops never end. Anchor the pattern to the process's
  own command line (`pgrep -f "^.venv/bin/python -m catan_rl.ppo --name <run>"`) or use background task IDs.
- The web server registers bots only at import time, so restart it after a new `best.pt` appears.
- pyo3 0.29: `py.detach` releases the GIL (formerly `allow_threads`). `Game` uses
  `#[pyclass(skip_from_py_object)]` to silence the Clone/FromPyObject deprecation.
- `VecEnv.step` signature: `(actions, obs, mask, actor, done, winner, length, final_vp)`. All are
  caller-owned numpy buffers, written in place, with auto-reset on game end. `last_game_stats()` returns
  the end-of-game stats of each env's last finished game.
- Bot seats set with `VecEnv.set_seats` act inside Rust during `step`/`reset`, so Python only sees states
  where an `"external"` seat acts.
- `evaluate()` is fully determined by its seed (engine streams plus a forked, seeded torch RNG), and PPO evaluates
  every checkpoint on the same games (seed 10000). Different game sets still differ by ±4 pp at 400 games (one
  checkpoint: 37.5% vs 46% on two seeds), so use 2000 games for claims. Runs before 2026-10-01 used a new game
  set per evaluation; their `metrics.csv` curves are noisier than `eval_curve.csv`.
- Replays and seeds from before the chance-stream split (2026-10-01) don't reproduce: the same seed now gives
  different dice. Checkpoints are unaffected.
- No Node.js on this machine, so the dataviz palette validator can't run; the charts use slots 1–4 of its
  documented reference palette plus magenta (slot 5) and a neutral grey.

## Verifying changes

- Rules or engine: `cargo test --release` plus the `--ignored` stress test. Add a targeted test in
  `tests/rules.rs` for any new rule. Run the MCTS tests in debug once too (`cargo test -p catan-core mcts`),
  since debug builds check the legality of every replayed move.
- Performance: `cargo bench` before and after. Current numbers are 13M steps/s per core (random bot) and
  ~1.4–1.8M/s per core including observation encoding.
- Bindings: `python -m catan_rl.smoke`.
- UI: take headless screenshots with Chrome and look at them:
  `google-chrome --headless=new --window-size=1440,900 --virtual-time-budget=5000 --screenshot=out.png "http://localhost:8765/?replay=<name>&step=<n>"`
  Also play a game over the WebSocket (see `smoke`-style scripts) to check the server flow.
- PPO: the diagnostic recipe above should show `eval_vp` rising within ~1M steps and eval win rate above
  0.25 after ~3M steps (mixed opponents + VP shaping, fixed board).

## Conventions

- Match the surrounding style: terse doc comments on modules and non-obvious functions, with no narration.
- Rust: keep hot paths allocation-free; iterate bitmasks with `bits64`/`bits128`/`bits32`.
- Python: type hints, dataclass config, numpy buffers over per-env Python loops where possible.
- Commit with the attribution trailer given by the harness; commit at milestones the owner has approved.
