# CLAUDE.md

Guidance for Claude Code sessions in this repo. **Keep this file clean and current:** when something
here stops being true (a gotcha gets fixed, a milestone is done, a file moves), update or delete it in
the same change. Don't append history; this is a description of the present, not a changelog.

## Project goal

Settlers of Catan with three parts:
1. **Very fast rules engine** (Rust). Speed matters because RL training throughput depends on it.
2. **Good-looking web UI** that renders fast, for playing against bots and watching replays.
3. **RL bot** (PPO self-play + league) that should eventually beat the owner at Catan.

The bot is built in versions of increasing difficulty:

| Version | Board | Trading | Status |
|---|---|---|---|
| v1 | fixed beginner board | none (bank/port only) | **current**: trainer works; diagnostic run beats the heuristic bot 44% of the time (chance = 25%); full run not started |
| v2 | random boards | none | engine supports `random_board`; needs training (likely a GNN/transformer, ideally on GPU) |
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
    view.rs               serde views for the UI (board_view, state_view with hidden info per viewer)
    rng.rs                wyrand PRNG (deterministic)
  catan-core/tests/rules.rs   rule unit tests, invariant stress tests, determinism/replay tests
  catan-core/benches/playout.rs   throughput benchmark (std::time; harness = false)
  catan-py/src/lib.rs     PyO3 module `catan_rl._engine`: Game, VecEnv, action_names, bot_tournament
python/catan_rl/
  __init__.py             re-exports + ACTIONS offset dict (derived from action names)
  model.py                PolicyNet (MLP + LayerNorm, masked policy head, value head), save/load, PolicyBot
  ppo.py                  trainer (Config dataclass = CLI flags)
  evaluate.py             evaluate() vs bots or checkpoints; replay export
  bench_compute.py        rollout/train throughput and time projections
  smoke.py                end-to-end bindings check
web/server.py             FastAPI + WebSocket; owns the Game; registers bots (incl. runs/*/best.pt)
web/static/               index.html, style.css, board.js (canvas renderer), ui.js (session + panels + replays)
pyproject.toml            maturin config (python-source = python, module = catan_rl._engine)
runs/, replays/           training output / saved games (gitignored)
```

## Commands

Always use the venv (`.venv/bin/...`). Rust is installed via rustup; run `source ~/.cargo/env` first in fresh shells.

```bash
.venv/bin/maturin develop --release                       # REQUIRED after any Rust change (see gotchas)
cd engine && cargo test -p catan-core --release           # 15 tests, ~1 s
cd engine && cargo test -p catan-core --release -- --ignored   # 100k-game invariant stress test, ~20 s
cd engine && cargo bench -p catan-core                    # engine throughput
.venv/bin/python -m catan_rl.smoke                        # bindings end to end
.venv/bin/python -m catan_rl.bench_compute                # NN throughput and training-time projections
.venv/bin/uvicorn web.server:app --port 8765              # UI at http://localhost:8765
.venv/bin/python -m catan_rl.ppo --name <run> [flags]     # train; flags mirror ppo.Config fields
.venv/bin/python -m catan_rl.evaluate runs/<run>/best.pt --games 2000 [--replays N]
.venv/bin/python -m catan_rl.plot_runs <run> <run> ...  # eval curves side by side -> runs/compare.png
```

## Architecture invariants (don't break these)

- **The UI never implements rules.** The server sends `state.legal` (action ids); the UI only maps ids to
  clickable spots and buttons. Any rule change lives in `state.rs` only.
- **`State` is `Copy` and allocation-free.** Bitboards: `u64` per player for settlements/cities, `u128` per
  player for roads. This keeps stepping fast and makes cloning for search (MCTS later) free.
- **Determinism.** A game is fully defined by `(seed, config, action list)`, and replays depend on this. All
  randomness (dice, dev deck, steals, random board) comes from `State.rng`; bots use their own `Rng`.
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
- Longest Road is ≥5, broken by opponent settlements, and the holder keeps it on ties. If the holder
  drops and others tie, nobody holds it. Largest Army is ≥3 knights and strictly more than the holder.
- `max_turns` (default 500) ends the game as a draw (`winner = -1`).
- No player-to-player trading yet. Its action ids don't exist yet; adding them changes `N_ACTIONS` (see below).

## Action space and observation

- Flat action space, `N_ACTIONS = 253`. Offsets are in `actions.rs`: ROLL 0, END_TURN 1, SETTLE 2+v,
  CITY 56+v, ROAD 110+e, BUY_DEV 182, PLAY_KNIGHT 183, PLAY_ROAD_BUILDING 184, MONOPOLY 185+r,
  YOP 190+pair (15 unordered pairs), MOVE_ROBBER 205+h, STEAL 224+k (k = seats after the current
  player), DISCARD 228+r, TRADE 233 + give*4 + (get index skipping give).
- Resources are indexed 0 wood, 1 brick, 2 wool, 3 grain, 4 ore, 5 desert. Port type 5 = 3:1.
- The observation (`OBS_SIZE = 1292`) is **actor-relative**: seat 0 is always the acting player. Opponents
  are encoded with public info only (card counts, dev-card counts, not contents).
- Python derives offsets from action names (`catan_rl.ACTIONS`) and the web UI fetches them from `/api/meta`.
  **But `ui.js` duplicates the YOP pair order and the `tradeId` formula**, so update both if the layout changes.

## RL training (ppo.py)

- Env groups by index: `frac_heuristic` (learner in one seat vs 3 Rust heuristic bots played inside the
  engine), `frac_selfplay` (learner in all seats), and the rest league (learner in one seat vs frozen
  snapshots from `runs/<name>/pool/`; the learner itself while the pool is empty).
- Samples are tagged (env, seat). GAE runs per sequence and bootstraps from the same seat's next decision.
  Unfinished tails are **carried into the next rollout** rather than bootstrapped. League seat assignment
  only changes at game end; changing it mid-game would orphan carried samples.
- Reward is terminal: +1 win, −1/(n−1) loss, 0 draw, plus `vp_coef` × (own VP − mean opponent VP)/10,
  annealed to 0 over `vp_anneal_frac` of training. The shaping term matters: a fresh policy never beats
  the heuristic bot, so pure win/loss gives no signal early on.
- Outputs go to `runs/<name>/`: `metrics.csv`, `latest.pt` (includes optimizer state, used by `--resume`),
  `best.pt` (best eval win rate vs heuristic), `pool/`, and `config.json`.
- Measured on this machine (14 cores, no GPU): about 7k samples/s with a 2×256 MLP including league
  inference, and about 7.5k/s for a 3×512 MLP before league overhead. The network is the bottleneck; the
  engine does 13M raw steps/s per core. Diagnostic run `runs/diag` (2×256, 6M steps, ~15 min) won 44%
  against 3 heuristic bots.
- **Changing `OBS_SIZE` or `N_ACTIONS` invalidates every checkpoint**, and `web/server.py` loads all
  `runs/*/best.pt` at startup. Start a new run name, and delete or move incompatible runs.

## Gotchas

- **Stale extension:** after editing Rust, Python keeps importing the old `_engine` until you rerun
  `maturin develop --release`. There is no error, just old behaviour.
- **`pkill -f` / `pgrep -f` self-match:** patterns like `"uvicorn web.server:app"` or `"catan_rl.ppo"` also
  match the shell running the command, which kills it or makes `until ! pgrep` loops never end. Use
  background task IDs or a PID file instead.
- The web server registers bots only at import time, so restart it after a new `best.pt` appears.
- pyo3 0.29: `py.detach` releases the GIL (formerly `allow_threads`). `Game` uses
  `#[pyclass(skip_from_py_object)]` to silence the Clone/FromPyObject deprecation.
- `VecEnv.step` signature: `(actions, obs, mask, actor, done, winner, length, final_vp)`. All are
  caller-owned numpy buffers, written in place, with auto-reset on game end.
- Bot seats set with `VecEnv.set_seats` act inside Rust during `step`/`reset`, so Python only sees states
  where an `"external"` seat acts.

## Verifying changes

- Rules or engine: `cargo test --release` plus the `--ignored` stress test. Add a targeted test in
  `tests/rules.rs` for any new rule.
- Performance: `cargo bench` before and after. Current numbers are 13M steps/s per core (random bot) and
  ~1.4–1.8M/s per core including observation encoding.
- Bindings: `python -m catan_rl.smoke`.
- UI: take headless screenshots with Chrome and look at them:
  `google-chrome --headless=new --window-size=1440,900 --virtual-time-budget=5000 --screenshot=out.png "http://localhost:8765/?replay=<name>&step=<n>"`
  Also play a game over the WebSocket (see `smoke`-style scripts) to check the server flow.
- Training: a short run (e.g. `--total-steps 6e6 --hidden 256 --layers 2 --num-envs 256`) should show
  `eval_vp` rising within ~1M steps and eval win rate above 0.25 after ~3M steps.

## Conventions

- Match the surrounding style: terse doc comments on modules and non-obvious functions, with no narration.
- Rust: keep hot paths allocation-free; iterate bitmasks with `bits64`/`bits128`/`bits32`.
- Python: type hints, dataclass config, numpy buffers over per-env Python loops where possible.
- Commit with the attribution trailer given by the harness; commit only when asked or at milestones the
  owner has approved.
