"""Evaluate a policy against built-in bots (or another checkpoint).

    python -m catan_rl.evaluate runs/v1/best.pt [--opponent heuristic|random|<ckpt.pt>] [--games 1000]
    python -m catan_rl.evaluate runs/v1/best.pt --replays 5     # also save replays for the web UI
    python -m catan_rl.evaluate runs/v1/best.pt --random-board  # on random boards instead of the beginner board

The policy holds one seat (rotating over seats across envs); all other seats are the opponent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_rl import N_ACTIONS, OBS_SIZE, Game, VecEnv
from catan_rl.model import PolicyBot, PolicyNet, load


def evaluate(net: PolicyNet, opponent: str | PolicyNet = "heuristic", games: int = 400, n_players: int = 4,
             random_board: bool = False, seed: int = 12345, greedy: bool = False) -> dict:
    N = min(games, 256)
    env = VecEnv(N, seed=seed, n_players=n_players, random_board=random_board)
    learner_seat = np.arange(N) % n_players
    bot_opp = isinstance(opponent, str)
    for i in range(N):
        kind = opponent if bot_opp else "external"
        env.set_seats(i, ["external" if s == learner_seat[i] else kind for s in range(4)])
    obs = np.zeros((N, OBS_SIZE), np.float32)
    mask = np.zeros((N, N_ACTIONS), np.bool_)
    actor = np.zeros(N, np.int64)
    done = np.zeros(N, np.bool_)
    winner = np.zeros(N, np.int64)
    length = np.zeros(N, np.int64)
    final_vp = np.zeros((N, 4), np.int64)
    env.reset(obs, mask, actor)
    net.eval()
    finished = np.zeros(N, np.int64)
    per_env = -(-games // N)
    wins = draws = played = 0
    vps = []
    while played < games:
        mine = actor == learner_seat
        actions = np.zeros(N, np.int64)
        if mine.any():
            actions[mine] = net.act(obs[mine], mask[mine], greedy=greedy)[0]
        if not bot_opp and (~mine).any():
            actions[~mine] = opponent.act(obs[~mine], mask[~mine], greedy=greedy)[0]
        env.step(actions, obs, mask, actor, done, winner, length, final_vp)
        for e in np.nonzero(done)[0]:
            if finished[e] >= per_env:
                continue
            finished[e] += 1
            played += 1
            wins += int(winner[e] == learner_seat[e])
            draws += int(winner[e] < 0)
            vps.append(final_vp[e, learner_seat[e]])
    return {"games": played, "win_rate": wins / played, "draw_rate": draws / played, "avg_vp": float(np.mean(vps))}


def save_replays(ckpt: str, count: int, opponent: str = "heuristic", n_players: int = 4, seed: int = 777) -> None:
    """Play `count` games (policy in seat 0) through `Game` and store them for the web replay viewer."""
    bot = PolicyBot(ckpt, greedy=False)
    out = Path("replays")
    out.mkdir(exist_ok=True)
    for k in range(count):
        g = Game(seed=seed + k, n_players=n_players)
        seat = k % n_players
        while not g.is_over:
            g.step(bot(g) if g.actor == seat else g.bot_action(opponent))
        players = ["ppo" if s == seat else opponent for s in range(n_players)]
        name = f"ppo-{Path(ckpt).parent.name}-{seed + k}"
        (out / f"{name}.json").write_text(json.dumps({
            "seed": seed + k, "config": {"n_players": n_players}, "actions": g.history,
            "players": players, "winner": g.winner,
        }))
        print(f"saved replays/{name}.json  winner={players[g.winner] if g.winner >= 0 else 'draw'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--random-board", action="store_true")
    ap.add_argument("--replays", type=int, default=0)
    args = ap.parse_args()
    net = load(args.checkpoint)
    opp = args.opponent if args.opponent in ("heuristic", "random") else load(args.opponent)
    res = evaluate(net, opp, games=args.games, n_players=args.players, greedy=args.greedy,
                   random_board=args.random_board)
    print(json.dumps(res, indent=2))
    print(f"(chance level with {args.players} players: {1 / args.players:.3f})")
    if args.replays:
        save_replays(args.checkpoint, args.replays, n_players=args.players)


if __name__ == "__main__":
    main()
