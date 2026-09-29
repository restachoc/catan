"""End-to-end smoke test of the Python bindings.

    python -m catan_rl.smoke

Plays full games through `VecEnv` with a uniformly random masked policy in every external seat,
checks the buffers stay consistent, and reports throughput. Also checks `Game` + built-in bots.
"""

from __future__ import annotations

import json
import time

import numpy as np

from catan_rl import ACTIONS, N_ACTIONS, OBS_SIZE, Game, VecEnv, bot_tournament


def random_masked(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Sample uniformly among legal actions per row.
    noise = rng.random(mask.shape, dtype=np.float32)
    noise[~mask] = -1.0
    return noise.argmax(axis=1).astype(np.int64)


def vecenv_games(num_envs: int = 256, target_games: int = 10_000, seats=None) -> None:
    env = VecEnv(num_envs, seed=123)
    if seats:
        for i in range(num_envs):
            env.set_seats(i, seats)
    obs = np.zeros((num_envs, OBS_SIZE), np.float32)
    mask = np.zeros((num_envs, N_ACTIONS), np.bool_)
    actor = np.zeros(num_envs, np.int64)
    done = np.zeros(num_envs, np.bool_)
    winner = np.zeros(num_envs, np.int64)
    length = np.zeros(num_envs, np.int64)
    env.reset(obs, mask, actor)
    rng = np.random.default_rng(0)

    games = steps = 0
    wins = np.zeros(5, np.int64)
    t = time.perf_counter()
    while games < target_games:
        assert mask.any(axis=1).all(), "an env has no legal action"
        assert np.isfinite(obs).all()
        a = random_masked(mask, rng)
        env.step(a, obs, mask, actor, done, winner, length)
        steps += num_envs
        n_done = int(done.sum())
        games += n_done
        for w in winner[done]:
            wins[w if w >= 0 else 4] += 1
    dt = time.perf_counter() - t
    label = "random policy, all seats" if not seats else f"random policy vs {seats}"
    print(f"VecEnv[{label}]: {games} games in {dt:.1f}s -> {games / dt:,.0f} games/s, "
          f"{steps / dt / 1e6:.2f}M policy steps/s | wins per seat {wins[:4].tolist()} draws {wins[4]}")


def game_api() -> None:
    g = Game(seed=7)
    board = json.loads(g.board_json())
    assert len(board["hex_res"]) == 19 and len(board["vertex_xy"]) == 54 and len(board["edge_vertices"]) == 72
    while not g.is_over:
        legal = g.legal_actions()
        a = g.bot_action("heuristic")
        assert a in legal
        g.step(a)
    st = json.loads(g.state_json(0))
    assert st["winner"] == g.winner
    # Replaying the history reproduces the game exactly.
    r = Game(seed=7)
    for a in g.history:
        r.step(a)
    assert r.winner == g.winner and r.state_json(None) == g.state_json(None)
    # Illegal actions are rejected.
    try:
        Game(seed=1).step(ACTIONS["END_TURN"])
        raise AssertionError("illegal action accepted")
    except ValueError:
        pass
    print(f"Game API ok: heuristic self-play game won by seat {g.winner} in {len(g.history)} actions")


def main() -> None:
    game_api()
    counts = bot_tournament(["heuristic", "random", "random", "random"], 2000)
    print(f"heuristic vs 3x random over 2000 games: {counts} (seat wins..., draws)")
    vecenv_games()
    vecenv_games(seats=["external", "heuristic", "heuristic", "heuristic"], target_games=3000)


if __name__ == "__main__":
    main()
