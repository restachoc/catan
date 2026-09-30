"""AlphaZero-style training: MCTS self-play with the current network, trained on search targets.

    python -m catan_rl.az --name az1 [--sims 64] [--num-games 256] [--total-samples 5e6] ...

Loop: every game in an `AzPool` (Rust) is searched in lockstep, one leaf per game per round, so the
network evaluates a whole batch at once. Each searched decision becomes a sample (observation,
root visit distribution, eventual winner relative to the player who moved). Samples go into a replay
buffer; the network trains on them at a fixed reuse ratio (each sample is seen ~`reuse` times).

Losses: cross-entropy between the policy and the visit distribution, and cross-entropy between the
4-way value head and the winner. No reward shaping, no heuristic opponents: pure self-play.
Evaluation: raw policy vs 3 heuristic bots (cheap) and MCTS vs 3 heuristic bots (what matters).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from catan_rl import N_ACTIONS, OBS_SIZE
from catan_rl._engine import AzPool
from catan_rl.evaluate import evaluate
from catan_rl.model import AZNet, save


@dataclass
class Config:
    name: str = "az1"
    seed: int = 0
    n_players: int = 4
    random_board: bool = False
    hidden: int = 256
    layers: int = 2
    # self-play
    num_games: int = 256
    sims: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    noise_frac: float = 0.25
    temp_moves: int = 30
    total_samples: float = 5e6
    # training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 1024
    replay: int = 200_000
    min_replay: int = 100_000
    reuse: float = 4.0
    # eval / io
    eval_every: int = 100_000
    eval_games: int = 400
    eval_search_games: int = 200
    threads: int = 0
    resume: str = ""


class Replay:
    """Ring buffer; observations stored as float16 to halve memory."""

    def __init__(self, cap: int):
        self.obs = np.zeros((cap, OBS_SIZE), np.float16)
        self.mask = np.zeros((cap, N_ACTIONS), np.bool_)
        self.pi = np.zeros((cap, N_ACTIONS), np.float32)
        self.z = np.zeros((cap, 4), np.float32)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, obs, mask, pi, z) -> None:
        idx = (self.i + np.arange(len(z))) % self.cap
        self.obs[idx], self.mask[idx], self.pi[idx], self.z[idx] = obs, mask, pi, z
        self.i = int((self.i + len(z)) % self.cap)
        self.n = min(self.n + len(z), self.cap)

    def sample(self, b: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, b)
        return (torch.from_numpy(self.obs[idx].astype(np.float32)), torch.from_numpy(self.mask[idx]),
                torch.from_numpy(self.pi[idx]), torch.from_numpy(self.z[idx]))


def eval_search(net: AZNet, cfg: Config, games: int, seed: int) -> dict:
    """MCTS (no noise, greedy after the opening) in one seat vs 3 heuristic bots."""
    n = min(games, 128)
    pool = AzPool(n, seed=seed, n_players=cfg.n_players, random_board=cfg.random_board, sims=cfg.sims,
                  c_puct=cfg.c_puct, noise_frac=0.0, temp_moves=0, record=False)
    for i in range(n):
        seat = i % cfg.n_players
        pool.set_seats(i, ["search" if s == seat else "heuristic" for s in range(4)])
    obs = np.zeros((n, OBS_SIZE), np.float32)
    mask = np.zeros((n, N_ACTIONS), np.bool_)
    results = []
    while len(results) < games:
        k = pool.select(obs, mask)
        while k:
            p, v = net.evaluate(obs[:k], mask[:k])
            pool.expand(p, v)
            k = pool.select(obs, mask)
        pool.advance()
        results += pool.take_results()
    results = results[:games]
    wins = sum(r[2][r[0]] == "search" for r in results if r[0] >= 0)
    vps = [vp for w, vps, seats in results for vp, s in zip(vps, seats) if s == "search"]
    return {"win_rate": wins / len(results), "avg_vp": float(np.mean(vps))}


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.dir = Path("runs") / cfg.name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        self.net = AZNet(cfg.hidden, cfg.layers)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.samples = 0
        self.games = 0
        self.updates = 0
        self.best = -1.0
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            self.net.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.samples, self.games, self.best = ck["samples"], ck["games"], ck.get("best", -1.0)
        self.pool = AzPool(cfg.num_games, seed=cfg.seed * 1_000_003 + self.samples, n_players=cfg.n_players,
                           random_board=cfg.random_board, sims=cfg.sims, c_puct=cfg.c_puct,
                           dirichlet_alpha=cfg.dirichlet_alpha, noise_frac=cfg.noise_frac,
                           temp_moves=cfg.temp_moves)
        self.replay = Replay(cfg.replay)
        self.obs = np.zeros((cfg.num_games, OBS_SIZE), np.float32)
        self.mask = np.zeros((cfg.num_games, N_ACTIONS), np.bool_)
        self.metrics = self.dir / "metrics.csv"

    def selfplay_round(self) -> int:
        """One searched move in every game; returns the number of finished-game samples collected."""
        self.net.eval()
        k = self.pool.select(self.obs, self.mask)
        while k:
            p, v = self.net.evaluate(self.obs[:k], self.mask[:k])
            self.pool.expand(p, v)
            k = self.pool.select(self.obs, self.mask)
        self.pool.advance()
        obs, mask, pi, z = self.pool.take_samples()
        self.games += len(self.pool.take_results())
        if len(z):
            self.replay.add(obs, mask, pi, z)
            self.samples += len(z)
        return len(z)

    def train(self, n_updates: int) -> dict:
        self.net.train()
        logs = {"loss_pi": [], "loss_v": [], "entropy": []}
        for _ in range(n_updates):
            obs, mask, pi, z = self.replay.sample(self.cfg.batch, self.rng)
            logits, v = self.net(obs, mask)
            logp = torch.log_softmax(logits, -1)
            loss_pi = -(pi * logp.clamp(min=-30)).sum(-1).mean()
            loss_v = -(z * torch.log_softmax(v, -1)).sum(-1).mean()
            loss = loss_pi + loss_v
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            with torch.no_grad():
                logs["loss_pi"].append(loss_pi.item())
                logs["loss_v"].append(loss_v.item())
                logs["entropy"].append(-(logp.exp() * logp.clamp(min=-30)).sum(-1).mean().item())
            self.updates += 1
        return {k: float(np.mean(v)) if v else 0.0 for k, v in logs.items()}

    def log(self, row: dict) -> None:
        new = not self.metrics.exists()
        with self.metrics.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def checkpoint(self, path: Path) -> None:
        save(self.net, path, opt=self.opt.state_dict(), samples=self.samples, games=self.games, best=self.best,
             steps=self.samples)

    def run(self) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        start_samples = self.samples
        next_eval = (self.samples // cfg.eval_every + 1) * cfg.eval_every
        owed = 0.0
        info = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}
        while self.samples < cfg.total_samples:
            new = self.selfplay_round()
            if self.replay.n >= cfg.min_replay and new:
                owed += new * cfg.reuse / cfg.batch
                n_up = int(owed)
                owed -= n_up
                if n_up:
                    info = self.train(n_up)
            if self.samples >= next_eval:
                next_eval += cfg.eval_every
                elapsed = time.perf_counter() - t0
                raw = evaluate(self.net, "heuristic", games=cfg.eval_games, n_players=cfg.n_players,
                               random_board=cfg.random_board, seed=20_000 + self.samples)
                srch = eval_search(self.net, cfg, cfg.eval_search_games, seed=30_000 + self.samples)
                row = {
                    "steps": self.samples,
                    "games": self.games,
                    "updates": self.updates,
                    "hours": round(elapsed / 3600, 3),
                    "samples_per_s": round((self.samples - start_samples) / elapsed),
                    **{k: round(v, 4) for k, v in info.items()},
                    "eval_wr_heuristic": round(raw["win_rate"], 3),
                    "eval_vp": round(raw["avg_vp"], 2),
                    "eval_search_wr": round(srch["win_rate"], 3),
                    "eval_search_vp": round(srch["avg_vp"], 2),
                }
                self.log(row)
                self.checkpoint(self.dir / "latest.pt")
                if srch["win_rate"] > self.best:
                    self.best = srch["win_rate"]
                    self.checkpoint(self.dir / "best.pt")
                print(f"{self.samples / 1e6:6.2f}M samples | {self.games} games | {row['samples_per_s']} samples/s | "
                      f"loss pi {info['loss_pi']:.3f} v {info['loss_v']:.3f} ent {info['entropy']:.2f} | "
                      f"raw wr {row['eval_wr_heuristic']:.3f} vp {row['eval_vp']} | "
                      f"SEARCH wr {row['eval_search_wr']:.3f} vp {row['eval_search_vp']}", flush=True)
        self.checkpoint(self.dir / "latest.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        t = f.type if isinstance(f.type, type) else eval(f.type)
        name = "--" + f.name.replace("_", "-")
        if t is bool:
            ap.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(name, type=t, default=f.default)
    Trainer(Config(**vars(ap.parse_args()))).run()


if __name__ == "__main__":
    main()
