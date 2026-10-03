"""Evaluate a policy against built-in bots (or another checkpoint).

    python -m catan_rl.evaluate runs/v1/best.pt [--opponent heuristic|random|<ckpt.pt>] [--games 1000]
    python -m catan_rl.evaluate runs/v1/best.pt --replays 5     # also save replays for the web UI
    python -m catan_rl.evaluate runs/v1/best.pt --random-board  # on random boards instead of the beginner board
    python -m catan_rl.evaluate runs/v1/best.pt --pair          # two policy seats (opposite) vs two bots

The policy holds one seat (rotating over seats across envs), or with `pair` two opposite seats (0+2 or 1+3,
alternating), so that it can trade with itself (the built-in bots never trade); all other seats are the opponent.
With a pair, win_rate counts games won by either policy seat (chance 50%) and trades counts trades between them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from catan_rl import N_ACTIONS, OBS_SIZE, Game, VecEnv
from catan_rl._engine import STAT_NAMES
from catan_rl.model import PolicyBot, PolicyNet, load


def evaluate(net: PolicyNet, opponent: str | PolicyNet = "heuristic", games: int = 400, n_players: int = 4,
             random_board: bool = False, seed: int = 12345, greedy: bool = False, pair: bool = False,
             trading: bool = False) -> dict:
    """Win rate and VP of `net` in one seat (rotating by env), or two opposite seats, against `opponent`.

    Fully determined by `seed`: the same boards, dev decks and dice sequences every call (the engine keeps
    chance streams independent of the actions), and the policy's sampling uses a torch RNG seeded here and
    forked, so evaluating doesn't disturb the caller's random state.
    """
    dev = next(net.parameters()).device
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        return _evaluate(net, opponent, games, n_players, random_board, seed, greedy, pair, trading)


def _evaluate(net, opponent, games: int, n_players: int, random_board: bool, seed: int, greedy: bool,
              pair: bool, trading: bool) -> dict:
    assert not pair or n_players == 4, "pair evaluation needs 4 players"
    N = min(games, 256)
    env = VecEnv(N, seed=seed, n_players=n_players, random_board=random_board, trading=trading)
    learner = np.zeros((N, 4), np.bool_)  # [env, seat] held by the policy
    for i in range(N):
        seats = [i % 2, i % 2 + 2] if pair else [i % n_players]
        learner[i, seats] = True
    bot_opp = isinstance(opponent, str)
    for i in range(N):
        kind = opponent if bot_opp else "external"
        env.set_seats(i, ["external" if learner[i, s] else kind for s in range(4)])
    trades_col = STAT_NAMES.index("trades")
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
    vps, trades = [], []
    rows = np.arange(N)
    while played < games:
        mine = learner[rows, actor]
        actions = np.zeros(N, np.int64)
        if mine.any():
            actions[mine] = net.act(obs[mine], mask[mine], greedy=greedy)[0]
        if not bot_opp and (~mine).any():
            actions[~mine] = opponent.act(obs[~mine], mask[~mine], greedy=greedy)[0]
        env.step(actions, obs, mask, actor, done, winner, length, final_vp)
        stats = env.last_game_stats() if done.any() else None
        for e in np.nonzero(done)[0]:
            if finished[e] >= per_env:
                continue
            finished[e] += 1
            played += 1
            wins += int(winner[e] >= 0 and learner[e, winner[e]])
            draws += int(winner[e] < 0)
            vps.extend(final_vp[e, learner[e]])
            trades.append(stats[e, learner[e], trades_col].sum() / (2 if pair else 1))
    return {"games": played, "win_rate": wins / played, "draw_rate": draws / played, "avg_vp": float(np.mean(vps)),
            "trades": float(np.mean(trades))}


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
    ap.add_argument("--pair", action="store_true", help="two policy seats (opposite) vs two opponents")
    ap.add_argument("--trading", action="store_true", help="allow player-to-player trading")
    args = ap.parse_args()
    net = load(args.checkpoint)
    opp = args.opponent if args.opponent in ("heuristic", "random") else load(args.opponent)
    res = evaluate(net, opp, games=args.games, n_players=args.players, greedy=args.greedy,
                   random_board=args.random_board, pair=args.pair, trading=args.trading)
    print(json.dumps(res, indent=2))
    print(f"(chance level: {(2 if args.pair else 1) / args.players:.3f})")
    if args.replays:
        save_replays(args.checkpoint, args.replays, n_players=args.players)


if __name__ == "__main__":
    main()
