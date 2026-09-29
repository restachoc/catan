"""Measure what RL training would cost on this machine.

    python -m catan_rl.bench_compute [--device cuda]

Reports, for a few MLP sizes:
  * rollout throughput: VecEnv + batched policy inference (the self-play data-collection loop)
  * PPO update throughput: forward + backward + Adam over minibatches
  * the resulting end-to-end PPO rate (each sample is collected once, trained on `epochs` times)
and projects wall-clock time for a range of training budgets.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from catan_rl import N_ACTIONS, OBS_SIZE, VecEnv


def mlp(hidden: int, layers: int) -> nn.Module:
    mods, d = [], OBS_SIZE
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    body = nn.Sequential(*mods)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.pi, self.v = body, nn.Linear(d, N_ACTIONS), nn.Linear(d, 1)

        def forward(self, x):
            h = self.body(x)
            return self.pi(h), self.v(h)

    return Net()


def bench_rollout(net: nn.Module, device: str, num_envs: int, steps: int) -> float:
    env = VecEnv(num_envs, seed=0)
    obs = np.zeros((num_envs, OBS_SIZE), np.float32)
    mask = np.zeros((num_envs, N_ACTIONS), np.bool_)
    actor = np.zeros(num_envs, np.int64)
    done = np.zeros(num_envs, np.bool_)
    winner = np.zeros(num_envs, np.int64)
    length = np.zeros(num_envs, np.int64)
    env.reset(obs, mask, actor)
    t = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            logits, _ = net(torch.from_numpy(obs).to(device))
            logits = logits.masked_fill(~torch.from_numpy(mask).to(device), -1e9)
            a = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            env.step(a, obs, mask, actor, done, winner, length)
    return num_envs * steps / (time.perf_counter() - t)


def bench_train(net: nn.Module, device: str, batch: int, iters: int) -> float:
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    x = torch.randn(batch, OBS_SIZE, device=device)
    m = torch.rand(batch, N_ACTIONS, device=device) < 0.1
    m[:, 0] = True
    a = torch.zeros(batch, dtype=torch.long, device=device)
    adv = torch.randn(batch, device=device)
    ret = torch.randn(batch, device=device)
    old_lp = torch.zeros(batch, device=device) - 2.0

    def step():
        logits, v = net(x)
        logits = logits.masked_fill(~m, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        ratio = (dist.log_prob(a) - old_lp).exp()
        loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
        loss = loss + 0.5 * (v.squeeze(-1) - ret).pow(2).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    step()  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - t)


def fmt_time(sec: float) -> str:
    if sec < 3600:
        return f"{sec / 60:.0f} min"
    if sec < 86400 * 2:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} days"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs per batch of experience")
    args = ap.parse_args()
    dev = args.device
    print(f"device={dev}  torch threads={torch.get_num_threads()}  obs={OBS_SIZE}  actions={N_ACTIONS}\n")

    budgets = [100e6, 1e9, 5e9]
    print(f"{'model':<18}{'params':>9}{'rollout/s':>12}{'train/s':>12}{'PPO steps/s':>13}  " +
          "  ".join(f"{b / 1e6:>6.0f}M" if b < 1e9 else f"{b / 1e9:>6.0f}B" for b in budgets))
    for hidden, layers in [(256, 2), (512, 3), (1024, 3)]:
        torch.manual_seed(0)
        net = mlp(hidden, layers).to(dev)
        params = sum(p.numel() for p in net.parameters())
        roll = bench_rollout(net, dev, args.envs, 60)
        train = bench_train(net, dev, 4096, 15)
        # Time per sample: collect once + train `epochs` times.
        e2e = 1.0 / (1.0 / roll + args.epochs / train)
        print(f"{f'{layers}x{hidden}':<18}{params / 1e6:>8.2f}M{roll:>12,.0f}{train:>12,.0f}{e2e:>13,.0f}  " +
              "  ".join(f"{fmt_time(b / e2e):>7}" for b in budgets))


if __name__ == "__main__":
    main()
