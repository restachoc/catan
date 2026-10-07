"""PPO self-play with a league of past snapshots.

    python -m catan_rl.ppo --name v1 [--total-steps 1e9] [--num-envs 512] ...

Env groups (fixed per env index):
  * heuristic: the learner holds one seat (rotating by env index); the other seats are the Rust
    heuristic bot, played inside the engine.
  * selfplay: the learner holds every seat, so every decision becomes training data.
  * league: the learner holds one random seat; the other seats are drawn from a small active set of
    frozen past snapshots (or the learner itself while the pool is empty).

Turn-based multi-agent bookkeeping: each learner decision is a sample tagged (env, seat). GAE runs
per (env, seat) sequence, bootstrapping from the same seat's next decision. A sequence's last,
unfinished sample is carried into the next rollout instead of being bootstrapped with a guess.
Rewards are terminal: +1 win / -1/(n-1) loss (0 on a turn-limit draw), plus a VP-margin term that gives
denser signal (constant by default; can be annealed with --vp-anneal-frac).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from catan_rl import N_ACTIONS, OBS_SIZE, VecEnv
from catan_rl._engine import STAT_NAMES
from catan_rl.evaluate import evaluate
from catan_rl.model import GraphedPolicy, PolicyNet, load, make_policy, save
from catan_rl.strategy import STRATEGIES, classify, spending
from catan_rl.tracking import LEGACY, Tracker

RUST_BOT = -1
LEARNER = 0


@dataclass
class Config:
    name: str = "v1"
    seed: int = 0
    n_players: int = 4
    random_board: bool = False
    # model: "mlp" (hidden = width), "gnn" (hidden = node embedding size, layers = message-passing rounds) or
    # "transformer" (hidden = token size, layers = attention blocks over all board tokens, heads = attention heads)
    arch: str = "mlp"
    hidden: int = 512
    layers: int = 3
    heads: int = 4  # transformer only
    # rollout
    num_envs: int = 512
    rollout: int = 128
    total_steps: float = 1e9
    frac_heuristic: float = 0.25
    frac_selfplay: float = 0.25
    trading: bool = False  # player-to-player trading (off: the trade actions are never legal)
    # ppo
    lr: float = 3e-4
    warmup_steps: float = 5e5  # linear LR warmup over this many samples, then constant
    gamma: float = 0.999
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # reward shaping: vp_coef * (own VP - mean opponent VP) / 10, annealed to 0 over vp_anneal_frac of
    # total_steps (0 = never annealed, the default: fading it out didn't help, see findings/)
    vp_coef: float = 0.5
    vp_anneal_frac: float = 0.0
    # league
    snapshot_every: int = 20
    pool_size: int = 40
    active_opponents: int = 3
    # eval / io
    eval_every: int = 20
    eval_games: int = 400
    eval_both_boards: bool = True  # also evaluate on the other board type (generalisation curve)
    eval_pair: bool = False  # also evaluate two policy seats vs two heuristic bots (they can trade; chance 50%)
    threads: int = 0
    resume: str = ""
    save_every: float = 0  # with --wandb: upload the run directory to W&B every this many samples (and at the end)
    # hardware: "cpu" or "cuda"; amp = fp16 autocast (cuda only, ~1.5x for the GNN on a T4)
    device: str = "cpu"
    amp: bool = False
    # Weights & Biases (off by default; key from `wandb login` or WANDB_API_KEY, never from the repo)
    wandb: bool = False
    wandb_project: str = "catan"


class Buffer:
    """Flat sample storage for one rollout (plus samples carried over from the last one)."""

    def __init__(self, cap: int):
        self.obs = np.zeros((cap, OBS_SIZE), np.float32)
        self.mask = np.zeros((cap, N_ACTIONS), np.bool_)
        self.act = np.zeros(cap, np.int64)
        self.logp = np.zeros(cap, np.float32)
        self.val = np.zeros(cap, np.float32)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.bool_)
        self.env = np.zeros(cap, np.int64)
        self.seat = np.zeros(cap, np.int64)
        self.n = 0

    FIELDS = ("obs", "mask", "act", "logp", "val", "rew", "done", "env", "seat")

    def add(self, obs, mask, act, logp, val, env, seat) -> np.ndarray:
        k = len(act)
        s = slice(self.n, self.n + k)
        self.obs[s], self.mask[s], self.act[s] = obs, mask, act
        self.logp[s], self.val[s], self.env[s], self.seat[s] = logp, val, env, seat
        self.rew[s] = 0.0
        self.done[s] = False
        self.n += k
        return np.arange(s.start, s.stop)

    def take(self, idx: np.ndarray) -> dict:
        return {f: getattr(self, f)[idx].copy() for f in self.FIELDS}

    def put(self, data: dict) -> np.ndarray:
        k = len(data["act"])
        s = slice(self.n, self.n + k)
        for f in self.FIELDS:
            getattr(self, f)[s] = data[f]
        self.n += k
        return np.arange(s.start, s.stop)


def gae(buf: Buffer, gamma: float, lam: float):
    """Returns (advantages, returns, trainable mask, carry-over indices)."""
    n = buf.n
    order = np.lexsort((np.arange(n), buf.seat[:n], buf.env[:n]))
    key = buf.env[:n][order] * 8 + buf.seat[:n][order]
    same_next = np.zeros(n, np.bool_)
    same_next[:-1] = key[:-1] == key[1:]
    adv = np.zeros(n, np.float32)
    train = np.zeros(n, np.bool_)
    carry = []
    val, rew, done = buf.val, buf.rew, buf.done
    next_adv = 0.0
    for pos in range(n - 1, -1, -1):
        i = order[pos]
        if done[i]:
            a = rew[i] - val[i]
        elif same_next[pos]:
            j = order[pos + 1]
            a = rew[i] + gamma * val[j] - val[i] + gamma * lam * next_adv
        else:
            carry.append(i)
            next_adv = 0.0
            continue
        adv[i] = a
        train[i] = True
        next_adv = a
    ret = adv + val[:n]
    return adv, ret, train, np.array(carry, np.int64)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.dir = Path("runs") / cfg.name
        self.pool_dir = self.dir / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

        self.dev = torch.device(cfg.device)
        self.amp = cfg.amp and self.dev.type == "cuda"
        self.net = make_policy(cfg.arch, cfg.hidden, cfg.layers, cfg.heads).to(self.dev)
        self.net.amp = self.amp
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.scaler = torch.amp.GradScaler(self.dev.type, enabled=self.amp)
        # rollout actors: CUDA-graph replay on the GPU (per-call launch overhead dominates there)
        self.actor_net = GraphedPolicy(self.net) if self.dev.type == "cuda" else self.net
        self.steps = 0
        self.iter = 0
        self.best_wr = -1.0
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            self.net.load_state_dict(ck["model"])
            if "opt" in ck:
                self.opt.load_state_dict(ck["opt"])
            self.steps, self.iter = ck.get("steps", 0), ck.get("iter", 0)
            self.best_wr = ck.get("best_wr", -1.0)

        N = cfg.num_envs
        self.env = VecEnv(N, seed=cfg.seed * 1_000_003, n_players=cfg.n_players, random_board=cfg.random_board,
                          trading=cfg.trading)
        self.n_heur = int(N * cfg.frac_heuristic)
        self.n_self = int(N * cfg.frac_selfplay)
        self.ctrl = np.zeros((N, 4), np.int64)
        P = cfg.n_players
        for i in range(N):
            if i < self.n_heur:
                seat = i % P
                self.env.set_seats(i, ["external" if s == seat else "heuristic" for s in range(4)])
                self.ctrl[i] = RUST_BOT
                self.ctrl[i, seat] = LEARNER
        self.opponents: list = []  # active league opponents (actors); ctrl id k -> opponents[k-1]
        self.league = np.arange(self.n_heur + self.n_self, N)
        for i in self.league:
            self.assign_league(i)

        self.obs = np.zeros((N, OBS_SIZE), np.float32)
        self.mask = np.zeros((N, N_ACTIONS), np.bool_)
        self.actor = np.zeros(N, np.int64)
        self.done = np.zeros(N, np.bool_)
        self.winner = np.zeros(N, np.int64)
        self.length = np.zeros(N, np.int64)
        self.final_vp = np.zeros((N, 4), np.int64)
        self.env.reset(self.obs, self.mask, self.actor)

        self.buf = Buffer(N * cfg.rollout + N * 4)
        self.carry: dict | None = None
        self.last_idx = np.full((N, 4), -1, np.int64)
        self.reset_stats()
        self.metrics_path = self.dir / "metrics.csv"
        self.tracker = Tracker(cfg.wandb, cfg.wandb_project, self.dir, cfg.name, dataclasses.asdict(cfg))

    def reset_stats(self) -> None:
        self.stats = {"games": 0, "len": [], "learner_wins": 0, "learner_games": 0, "draws": 0, "players": []}

    # -------------------------------------------------------------- league

    def assign_league(self, i: int) -> None:
        P = self.cfg.n_players
        seat = self.rng.integers(P)
        k = len(self.opponents)
        for s in range(4):
            if s == seat:
                self.ctrl[i, s] = LEARNER
            elif s < P:
                self.ctrl[i, s] = LEARNER if k == 0 else 1 + self.rng.integers(k)
            else:
                self.ctrl[i, s] = RUST_BOT

    def refresh_opponents(self) -> None:
        pool = sorted(self.pool_dir.glob("*.pt"))
        if not pool:
            return
        k = min(self.cfg.active_opponents, len(pool))
        # Always include the newest snapshot, the rest uniformly from the pool.
        picks = [pool[-1]] + random.sample(pool[:-1], k - 1) if len(pool) > 1 else [pool[-1]]
        # Seat assignments only change at game end (assign_league), so running games keep their
        # learner seat; opponent ids stay valid because the active set never shrinks.
        self.opponents = [self.on_device(load(p)) for p in picks]

    def on_device(self, net: PolicyNet):
        net.amp = self.amp
        net = net.to(self.dev)
        return GraphedPolicy(net) if self.dev.type == "cuda" else net

    def snapshot(self) -> None:
        save(self.net, self.pool_dir / f"iter_{self.iter:06d}.pt", iter=self.iter, steps=self.steps)
        pool = sorted(self.pool_dir.glob("*.pt"))
        for old in pool[: max(0, len(pool) - self.cfg.pool_size)]:
            old.unlink()

    # -------------------------------------------------------------- rollout

    def reward(self, e: int, seat: int) -> float:
        P = self.cfg.n_players
        w = self.winner[e]
        win = 1.0 if w == seat else (-1.0 / (P - 1) if w >= 0 else 0.0)
        vp = self.final_vp[e, :P]
        margin = (vp[seat] - (vp.sum() - vp[seat]) / (P - 1)) / 10.0
        return win + self.vp_weight() * margin

    def vp_weight(self) -> float:
        cfg = self.cfg
        if cfg.vp_anneal_frac <= 0:
            return cfg.vp_coef
        return cfg.vp_coef * max(0.0, 1.0 - self.steps / (cfg.total_steps * cfg.vp_anneal_frac))

    def collect(self) -> None:
        buf = self.buf
        buf.n = 0
        self.last_idx[:] = -1
        if self.carry is not None and len(self.carry["act"]):
            idx = buf.put(self.carry)
            self.last_idx[self.carry["env"], self.carry["seat"]] = idx
        N = self.cfg.num_envs
        ar = np.arange(N)
        self.net.eval()
        for _ in range(self.cfg.rollout):
            ctrl_now = self.ctrl[ar, self.actor]
            actions = np.zeros(N, np.int64)
            li = np.nonzero(ctrl_now == LEARNER)[0]
            if len(li):
                a, lp, v = self.actor_net.act(self.obs[li], self.mask[li])
                actions[li] = a
                idx = buf.add(self.obs[li], self.mask[li], a, lp, v, li, self.actor[li])
                self.last_idx[li, self.actor[li]] = idx
            for k, opp in enumerate(self.opponents, start=1):
                oi = np.nonzero(ctrl_now == k)[0]
                if len(oi):
                    actions[oi] = opp.act(self.obs[oi], self.mask[oi])[0]
            self.env.step(actions, self.obs, self.mask, self.actor, self.done, self.winner, self.length, self.final_vp)
            self.steps += len(li)
            ended = np.nonzero(self.done)[0]
            game_stats = self.env.last_game_stats() if len(ended) else None
            for e in ended:
                self.stats["games"] += 1
                self.stats["len"].append(self.length[e])
                self.stats["draws"] += int(self.winner[e] < 0)
                learner_seats = [s for s in range(self.cfg.n_players) if self.ctrl[e, s] == LEARNER]
                self.stats["players"].extend(game_stats[e, learner_seats])
                for s in range(self.cfg.n_players):
                    j = self.last_idx[e, s]
                    if self.ctrl[e, s] == LEARNER and j >= 0:
                        buf.rew[j] += self.reward(e, s)
                        buf.done[j] = True
                        self.last_idx[e, s] = -1
                if e < self.n_heur or e >= self.n_heur + self.n_self:
                    seat = int(np.nonzero(self.ctrl[e] == LEARNER)[0][0])
                    self.stats["learner_games"] += 1
                    self.stats["learner_wins"] += int(self.winner[e] == seat)
                if e >= self.n_heur + self.n_self:
                    self.assign_league(e)

    # -------------------------------------------------------------- update

    def update(self) -> dict:
        cfg, buf = self.cfg, self.buf
        adv, ret, train, carry = gae(buf, cfg.gamma, cfg.lam)
        self.carry = buf.take(carry)
        idx = np.nonzero(train)[0]
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr * min(1.0, (self.steps + 1) / max(cfg.warmup_steps, 1))

        # obs/mask stay on the host (largest arrays); each minibatch is copied to the device
        obs = torch.from_numpy(buf.obs[idx])
        mask = torch.from_numpy(buf.mask[idx])
        act, old_lp, adv_t, ret_t = (torch.from_numpy(x).to(self.dev) for x in
                                     (buf.act[idx], buf.logp[idx], adv[idx], ret[idx]))
        n = len(idx)
        mb = min(cfg.minibatch, n)
        self.net.train()
        params_before = torch.nn.utils.parameters_to_vector(self.net.parameters()).detach().clone()
        logs = {k: [] for k in ("pg", "vf", "ent", "kl", "clipfrac", "grad_norm", "max_prob", "ent_norm", "n_legal")}
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n - mb + 1, mb):
                b = perm[s : s + mb]
                bd = b.to(self.dev)
                with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
                    logits, v = self.net(obs[b].to(self.dev), mask[b].to(self.dev))
                logp_all = torch.log_softmax(logits, -1)
                lp = logp_all.gather(-1, act[bd, None]).squeeze(-1)
                probs = logp_all.exp()
                ent_rows = -(probs * logp_all.clamp(min=-30)).sum(-1)
                ent = ent_rows.mean()
                ratio = (lp - old_lp[bd]).exp()
                a = adv_t[bd]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = 0.5 * (v - ret_t[bd]).pow(2).mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
                with torch.no_grad():
                    logs["pg"].append(pg.item())
                    logs["vf"].append(vf.item())
                    logs["ent"].append(ent.item())
                    logs["kl"].append(((ratio - 1) - ratio.log()).mean().item())
                    logs["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    logs["grad_norm"].append(grad_norm.item())  # before clipping (inf/nan = fp16 overflow step)
                    n_legal = mask[b].sum(-1).to(self.dev).float()
                    multi = n_legal > 1  # entropy relative to its maximum, log(#legal), where there is a choice
                    logs["ent_norm"].append((ent_rows[multi] / n_legal[multi].log()).mean().item() if multi.any() else 0.0)
                    logs["max_prob"].append(probs.max(-1).values.mean().item())
                    logs["n_legal"].append(n_legal.mean().item())
        out = {k: float(np.mean(v)) for k, v in logs.items() if k not in ("grad_norm", "max_prob", "ent_norm", "n_legal")}
        out["samples"] = n
        ev = 1 - np.var(ret[idx] - buf.val[idx]) / (np.var(ret[idx]) + 1e-8)
        out["explained_var"] = float(ev)
        g = np.array(logs["grad_norm"])
        params_after = torch.nn.utils.parameters_to_vector(self.net.parameters()).detach()
        out.update({
            "ppo/grad_norm": float(np.mean(g[np.isfinite(g)])) if np.isfinite(g).any() else float("nan"),
            "ppo/grad_norm_max": float(g[np.isfinite(g)].max()) if np.isfinite(g).any() else float("nan"),
            "ppo/grad_clipped_frac": float((g > cfg.max_grad_norm).mean()),
            "ppo/entropy_norm": float(np.mean(logs["ent_norm"])),
            "ppo/max_prob": float(np.mean(logs["max_prob"])),
            "ppo/legal_actions": float(np.mean(logs["n_legal"])),
            "ppo/adv_mean": float(adv[idx].mean()), "ppo/adv_std": float(adv[idx].std()),
            "ppo/return_mean": float(ret[idx].mean()), "ppo/value_mean": float(buf.val[idx].mean()),
            "ppo/lr": self.opt.param_groups[0]["lr"],
            "ppo/vp_coef": self.vp_weight(),
            "ppo/carried_frac": len(carry) / max(1, buf.n),
            "nn/param_norm": float(params_after.norm()),
            "nn/update_ratio": float((params_after - params_before).norm() / (params_before.norm() + 1e-12)),
            "nn/amp_scale": float(self.scaler.get_scale()) if self.amp else 1.0,
        })
        return out

    def game_metrics(self) -> dict:
        """Learner players in the training games finished this iteration: results, VP and strategy mix."""
        st = self.stats
        out = {"game/draw_rate": st["draws"] / max(1, st["games"])}
        if not st["players"]:
            return out
        p = np.array(st["players"], np.float32)
        S = {n: i for i, n in enumerate(STAT_NAMES)}
        won = p[:, S["won"]] > 0
        out.update({
            "game/learner_vp": float(p[:, S["vp"]].mean()),
            "game/longest_road_rate": float(p[:, S["longest_road"]].mean()),
            "game/largest_army_rate": float(p[:, S["largest_army"]].mean()),
            "game/knights": float(p[:, S["knights"]].mean()),
            "game/cities": float(p[:, S["cities"]].mean()),
            "game/settlements": float(p[:, S["settlements"]].mean()),
            "game/offers": float(p[:, S["offers"]].mean()),
            "game/trades": float(p[:, S["trades"]].mean()),
        })
        labels = classify(p)
        for group, sel in (("all", np.ones(len(p), bool)), ("winners", won)):
            if not sel.any():
                continue
            for i, name in enumerate(STRATEGIES):
                out[f"strategy/{group}/{name}"] = float((labels[sel] == i).mean())
            sp = spending(p[sel])
            total = max(1.0, float(sum(v.sum() for v in sp.values())))
            for k, v in sp.items():
                out[f"strategy/{group}/spend_{k}"] = float(v.sum() / total)
        return out

    # -------------------------------------------------------------- main loop

    def log(self, row: dict) -> None:
        """metrics.csv: the fixed legacy columns (plot_runs reads these); metrics.jsonl: every metric."""
        new = not self.metrics_path.exists()
        # keep an existing file's column order (resumed runs)
        fields = list(LEGACY) if new else self.metrics_path.open().readline().strip().split(",")
        with self.metrics_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in row.items() if v != ""}) + "\n")

    def checkpoint(self, path: Path) -> None:
        save(self.net, path, opt=self.opt.state_dict(), steps=self.steps, iter=self.iter, best_wr=self.best_wr)

    def run(self) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        steps0 = self.steps
        saved = int(self.steps // cfg.save_every) if cfg.save_every else 0
        while self.steps < cfg.total_steps:
            t = time.perf_counter()
            self.collect()
            t_roll = time.perf_counter() - t
            info = self.update()
            t_all = time.perf_counter() - t
            t_upd = t_all - t_roll
            self.iter += 1
            if self.iter % cfg.snapshot_every == 0:
                self.snapshot()
                self.refresh_opponents()
            st = self.stats
            row = {
                "iter": self.iter,
                "steps": self.steps,
                "sps": round(info["samples"] / t_all),
                "roll_frac": round(t_roll / t_all, 2),
                "games": st["games"],
                "game_len": round(float(np.mean(st["len"])) if st["len"] else 0, 1),
                "train_wr": round(st["learner_wins"] / max(1, st["learner_games"]), 3),
                **{k: (round(v, 4) if k in LEGACY else v) for k, v in info.items()},
                "eval_wr_heuristic": "",
                "eval_vp": "",
                "perf/rollout_s": round(t_roll, 2),
                "perf/update_s": round(t_upd, 2),
                "league/pool": len(list(self.pool_dir.glob("*.pt"))),
                "league/active_opponents": len(self.opponents),
                **self.game_metrics(),
            }
            if self.dev.type == "cuda":
                row["perf/gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                torch.cuda.reset_peak_memory_stats()
            self.reset_stats()
            if self.iter % cfg.eval_every == 0:
                te = time.perf_counter()
                ev = evaluate(self.net, "heuristic", games=cfg.eval_games, n_players=cfg.n_players,
                              random_board=cfg.random_board, seed=10_000, trading=cfg.trading)  # same games every eval
                row["eval_wr_heuristic"] = round(ev["win_rate"], 3)
                row["eval_vp"] = round(ev["avg_vp"], 2)
                if cfg.eval_both_boards:
                    other = "beginner" if cfg.random_board else "random"
                    ev2 = evaluate(self.net, "heuristic", games=cfg.eval_games, n_players=cfg.n_players,
                                   random_board=not cfg.random_board, seed=10_000, trading=cfg.trading)
                    row[f"eval/{other}_board/win_rate"] = round(ev2["win_rate"], 3)
                    row[f"eval/{other}_board/vp"] = round(ev2["avg_vp"], 2)
                if cfg.eval_pair and cfg.n_players == 4:
                    ev3 = evaluate(self.net, "heuristic", games=cfg.eval_games, n_players=4,
                                   random_board=cfg.random_board, seed=10_000, pair=True, trading=cfg.trading)
                    row["eval/pair/win_rate"] = round(ev3["win_rate"], 3)
                    row["eval/pair/vp"] = round(ev3["avg_vp"], 2)
                    row["eval/pair/trades"] = round(ev3["trades"], 2)
                row["perf/eval_s"] = round(time.perf_counter() - te, 1)
                self.checkpoint(self.dir / "latest.pt")
                if ev["win_rate"] > self.best_wr:
                    self.best_wr = ev["win_rate"]
                    self.checkpoint(self.dir / "best.pt")
            self.log(row)
            self.tracker.log(row)
            if cfg.save_every and self.steps // cfg.save_every > saved:
                saved = int(self.steps // cfg.save_every)
                self.checkpoint(self.dir / "latest.pt")
                self.tracker.save_files(self.dir)
            elapsed = time.perf_counter() - t0
            rate = (self.steps - steps0) / elapsed
            eta = (cfg.total_steps - self.steps) / max(rate, 1)
            print(
                f"it {self.iter:5d} | {self.steps / 1e6:8.2f}M steps | {rate:6.0f} sps | "
                f"len {row['game_len']:5} | train wr {row['train_wr']:.2f} | ent {info['ent']:.2f} "
                f"kl {info['kl']:.4f} ev {info['explained_var']:.2f}"
                + (f" | EVAL vs heuristic wr {row['eval_wr_heuristic']:.3f} vp {row['eval_vp']}" if row["eval_vp"] != "" else "")
                + (f" | pair wr {row['eval/pair/win_rate']:.3f} trades {row['eval/pair/trades']}" if "eval/pair/win_rate" in row else "")
                + f" | eta {eta / 3600:.1f}h",
                flush=True,
            )
        self.checkpoint(self.dir / "latest.pt")
        if cfg.save_every:
            self.tracker.save_files(self.dir)
        self.tracker.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        t = f.type if isinstance(f.type, type) else eval(f.type)
        name = "--" + f.name.replace("_", "-")
        if t is bool:
            ap.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(name, type=float if f.name in ("total_steps", "warmup_steps", "save_every") else t,
                            default=f.default)
    cfg = Config(**vars(ap.parse_args()))
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
