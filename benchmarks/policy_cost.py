"""GPU cost of the production policy networks under the PPO settings, to size one architecture like another.

    python benchmarks/policy_cost.py [--configs gnn:128:4 transformer:64:4 ...]

Per network: one PPO minibatch step (forward + backward + Adam, fp16 autocast, minibatch 4096) and one rollout
batch (256 rows through a CUDA-graph `GraphedPolicy`). The estimate combines them like a PPO iteration: 4 epochs of
updates per sample plus about two rollout forwards per sample (learner and league opponents).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catan_rl import N_ACTIONS, OBS_SIZE, VecEnv
from catan_rl.model import GraphedPolicy, make_policy


def sample_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Real observations and masks from random play (so masks have realistic legal sets)."""
    env = VecEnv(256, seed=0, random_board=True)
    obs = np.zeros((256, OBS_SIZE), np.float32)
    mask = np.zeros((256, N_ACTIONS), np.bool_)
    actor = np.zeros(256, np.int64)
    bufs = [np.zeros(256, np.bool_), np.zeros(256, np.int64), np.zeros(256, np.int64), np.zeros((256, 4), np.int64)]
    env.reset(obs, mask, actor)
    rng = np.random.default_rng(0)
    os, ms = [], []
    while sum(len(o) for o in os) < n:
        os.append(obs.copy())
        ms.append(mask.copy())
        env.step(np.argmax(rng.random(mask.shape) * mask, 1), obs, mask, actor, *bufs)
    return torch.from_numpy(np.concatenate(os)[:n]), torch.from_numpy(np.concatenate(ms)[:n])


def timed(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["gnn:128:4", "transformer:64:4", "transformer:64:3",
                                                      "transformer:96:3", "transformer:128:2"])
    ap.add_argument("--minibatch", type=int, default=4096)
    args = ap.parse_args()
    dev = torch.device("cuda")
    obs, mask = sample_batch(args.minibatch)
    obs_d, mask_d = obs.to(dev), mask.to(dev)
    act = torch.randint(0, 1, (args.minibatch,), device=dev)
    print(f"{'network':<20}{'params':>9}{'update ms':>11}{'rollout ms':>12}{'est. samples/s':>16}{'GB':>6}")
    for c in args.configs:
        arch, d, layers = c.split(":")
        net = make_policy(arch, int(d), int(layers)).to(dev)
        net.amp = True
        opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
        scaler = torch.amp.GradScaler()
        legal_first = mask_d.float().argmax(-1)

        def update():
            with torch.autocast("cuda", dtype=torch.float16):
                logits, v = net(obs_d, mask_d)
                loss = -torch.log_softmax(logits, -1).gather(-1, legal_first[:, None]).mean() + v.square().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        torch.cuda.reset_peak_memory_stats()
        t_up = timed(update, 10)
        gp = GraphedPolicy(net)
        o256, m256 = obs[:256].numpy(), mask[:256].numpy()
        t_roll = timed(lambda: gp.act(o256, m256), 30)
        per_sample = 4 * t_up / args.minibatch + 2 * t_roll / 256
        params = sum(p.numel() for p in net.parameters()) / 1e6
        gb = torch.cuda.max_memory_allocated() / 2**30
        print(f"{c:<20}{params:>8.2f}M{t_up * 1e3:>11.1f}{t_roll * 1e3:>12.2f}{1 / per_sample:>16,.0f}{gb:>6.1f}", flush=True)
        del net, opt, gp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
