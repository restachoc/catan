"""Forward/backward speed of candidate board-structured networks versus the current MLP.

    .venv/bin/python benchmarks/arch_speed.py [--quick] [--device cuda] [--threads N] [--csv benchmarks/results/arch_speed.csv]

Runs niced on half the cores by default, and trains in micro-batches (gradient accumulation over the
PPO minibatch) so the transformer's attention matrices stay small; a full 4096 batch needs several GB.

Prototypes (speed only, untrained) over the real board graph from `Game.board_json()`:
  * mlp:  the current PolicyNet torso (baseline)
  * gnn:  heterogeneous message passing over hexes, vertices and edges plus one global token
  * tf:   transformer over all 146 tokens with a learned per-head graph-distance attention bias
  * hyb:  gnn layers plus attention between the nodes and 5 tokens (global + one per seat)

All graph networks read the existing flat observation (no engine change) and emit the flat 253-way
policy: vertex embeddings score settlements and cities, edge embeddings roads, hex embeddings the
robber, and the global token everything else plus the value.

Reports per network: inference samples/s at the rollout batch size (PPO acting), training samples/s
for forward + backward + Adam at the PPO minibatch size, and an estimated end-to-end PPO rate
(each sample is acted on once and trained on `epochs` times; engine time is ignored).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from catan_rl import N_ACTIONS, OBS_SIZE
from catan_rl.graph import (BOARD_OBS, EDGE_F, GLOBAL_F, HEX_F, MAX_P, N_EDGE, N_GLOBAL_ACT, N_HEX, N_TOK, N_VERT,
                            PLAYER_F, VERT_F,
                            GNNLayer, Topology, TopoBuffers, action_perm, dense, split_obs)


# ------------------------------------------------------------------ networks


class MLP(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        mods, d = [], OBS_SIZE
        for _ in range(layers):
            mods.append(dense(d, hidden))
            d = hidden
        self.torso = nn.Sequential(*mods)
        self.pi, self.v = nn.Linear(d, N_ACTIONS), nn.Linear(d, 1)

    def forward(self, obs):
        h = self.torso(obs)
        return self.pi(h), self.v(h).squeeze(-1)


class GraphHeads(nn.Module):
    """Encoders from the split observation and per-location policy heads shared by the graph networks."""

    def __init__(self, d: int):
        super().__init__()
        self.enc_h, self.enc_v, self.enc_e = dense(HEX_F, d), dense(VERT_F, d), dense(EDGE_F, d)
        self.enc_g = dense(GLOBAL_F, d)
        self.pi_v, self.pi_e, self.pi_h = nn.Linear(d, 2), nn.Linear(d, 1), nn.Linear(d, 1)
        self.pi_g, self.v = nn.Linear(d, N_GLOBAL_ACT), nn.Linear(d, 1)
        self.register_buffer("perm", action_perm())

    def encode(self, obs):
        h, v, e, g = split_obs(obs)
        return self.enc_h(h), self.enc_v(v), self.enc_e(e), self.enc_g(g)

    def decode(self, h, v, e, g):
        pv = self.pi_v(v)
        flat = torch.cat([pv[..., 0], pv[..., 1], self.pi_e(e)[..., 0], self.pi_h(h)[..., 0], self.pi_g(g)], -1)
        return flat[:, self.perm], self.v(g).squeeze(-1)


def node_major(*xs):
    return [x.transpose(0, 1).contiguous() for x in xs]


def batch_major(*xs):
    return [x.transpose(0, 1) for x in xs]


class GNN(nn.Module):
    def __init__(self, topo: Topology, d: int, layers: int):
        super().__init__()
        self.io = GraphHeads(d)
        self.topo = TopoBuffers(topo)
        self.layers = nn.ModuleList(GNNLayer(d) for _ in range(layers))

    def forward(self, obs):
        h, v, e, g = self.io.encode(obs)
        h, v, e = node_major(h, v, e)
        x = {"h": h, "v": v, "e": e}
        for layer in self.layers:
            x, g = layer(x, g, self.topo)
        return self.io.decode(*batch_major(x["h"], x["v"], x["e"]), g)


class Attn(nn.Module):
    """Pre-LN multi-head attention block (queries from `x`, keys/values from `kv`) with an FFN."""

    def __init__(self, d: int, heads: int, ffn: int = 2):
        super().__init__()
        self.h = heads
        self.ln_q, self.ln_kv, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn * d), nn.ReLU(), nn.Linear(ffn * d, d))

    def forward(self, x, kv=None, bias=None):
        B, T, d = x.shape
        xq = self.ln_q(x)
        kvn = xq if kv is None else self.ln_kv(kv)
        q = self.q(xq).view(B, T, self.h, -1).transpose(1, 2)
        k, v = self.kv(kvn).view(B, kvn.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.ln_f(x))


class Transformer(nn.Module):
    """Full attention over hex, vertex, edge and one global token; bias[head, graph distance]."""

    MAX_D = 8

    def __init__(self, topo: Topology, d: int, layers: int, heads: int = 4):
        super().__init__()
        self.io = GraphHeads(d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        self.heads = heads
        dist = topo.dist.clamp(max=self.MAX_D)
        full = torch.full((N_TOK + 1, N_TOK + 1), self.MAX_D + 1, dtype=torch.long)
        full[:N_TOK, :N_TOK] = dist
        self.register_buffer("dist", full)
        self.bias = nn.ParameterList(nn.Parameter(torch.zeros(heads, self.MAX_D + 2)) for _ in range(layers))
        self.layers = nn.ModuleList(Attn(d, heads) for _ in range(layers))

    def forward(self, obs):
        h, v, e, g = self.io.encode(obs)
        t = self.type_emb
        x = torch.cat([h + t[0], v + t[1], e + t[2], g[:, None]], 1)
        for layer, b in zip(self.layers, self.bias):
            x = layer(x, bias=b[:, self.dist][None])
        return self.io.decode(x[:, :N_HEX], x[:, N_HEX:N_HEX + N_VERT], x[:, N_HEX + N_VERT:N_TOK], x[:, N_TOK])


class Hybrid(nn.Module):
    """GNN layers; after each, 5 tokens (global + one per seat) attend to all nodes, then nodes attend to the tokens."""

    def __init__(self, topo: Topology, d: int, layers: int, heads: int = 4):
        super().__init__()
        self.io = GraphHeads(d)
        self.topo = TopoBuffers(topo)
        self.enc_p = dense(PLAYER_F, d)
        self.seat_emb = nn.Parameter(torch.zeros(MAX_P, d))
        self.gnn = nn.ModuleList(GNNLayer(d) for _ in range(layers))
        self.tok_read = nn.ModuleList(Attn(d, heads) for _ in range(layers))
        self.node_read = nn.ModuleList(Attn(d, heads, ffn=1) for _ in range(layers))

    def forward(self, obs):
        h, v, e, g = self.io.encode(obs)
        players = obs[:, BOARD_OBS:BOARD_OBS + MAX_P * PLAYER_F].view(-1, MAX_P, PLAYER_F)
        tok = torch.cat([g[:, None], self.enc_p(players) + self.seat_emb], 1)
        nodes = torch.cat([h, v, e], 1)
        for gnn, tr, nr in zip(self.gnn, self.tok_read, self.node_read):
            nm = nodes.transpose(0, 1).contiguous()
            x = {"h": nm[:N_HEX], "v": nm[N_HEX:N_HEX + N_VERT], "e": nm[N_HEX + N_VERT:]}
            x, g0 = gnn(x, tok[:, 0], self.topo)
            nodes = torch.cat([x["h"], x["v"], x["e"]], 0).transpose(0, 1)
            tok = torch.cat([g0[:, None], tok[:, 1:]], 1)
            tok = tr(tok, kv=torch.cat([nodes, tok], 1))
            nodes = nr(nodes, kv=tok)
        return self.io.decode(nodes[:, :N_HEX], nodes[:, N_HEX:N_HEX + N_VERT], nodes[:, N_HEX + N_VERT:], tok[:, 0])


# ------------------------------------------------------------------ benchmark


def timed(fn, min_time: float, sync=lambda: None) -> float:
    """Seconds per call, after one warm-up call, averaged over at least `min_time` seconds (and 2 calls)."""
    fn()
    sync()
    n, t0 = 0, time.perf_counter()
    while (el := time.perf_counter() - t0) < min_time or n < 2:
        fn()
        sync()
        n += 1
    return el / n


def bench(net: nn.Module, infer_batch: int, train_batch: int, chunk: int, min_time: float,
          device: str) -> tuple[float, float]:
    """Inference samples/s (including the host-to-device copy of the observations) and training samples/s."""
    obs_i = torch.rand(infer_batch, OBS_SIZE)
    obs_t = torch.rand(train_batch, OBS_SIZE, device=device).split(chunk)
    act = torch.randint(0, N_ACTIONS, (train_batch,), device=device).split(chunk)
    ret = torch.randn(train_batch, device=device).split(chunk)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    def infer():
        with torch.inference_mode():
            net(obs_i.to(device))[0].cpu()

    def train():
        opt.zero_grad(set_to_none=True)
        for o, a, r in zip(obs_t, act, ret):
            logits, v = net(o)
            (F.cross_entropy(logits, a) + F.mse_loss(v, r)).mul(len(o) / train_batch).backward()
        opt.step()

    return infer_batch / timed(infer, min_time, sync), train_batch / timed(train, min_time, sync)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer-batch", type=int, default=256, help="rollout batch (num_envs of the diagnostic recipe)")
    p.add_argument("--train-batch", type=int, default=4096, help="PPO minibatch")
    p.add_argument("--train-chunk", type=int, default=512, help="micro-batch for forward/backward (caps memory)")
    p.add_argument("--epochs", type=int, default=4, help="PPO epochs, for the end-to-end estimate")
    p.add_argument("--min-time", type=float, default=2.0, help="seconds per measurement")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--threads", type=int, default=0, help="torch threads (default: half the cores)")
    p.add_argument("--nice", type=int, default=10, help="niceness increment for this process")
    p.add_argument("--quick", action="store_true", help="only the smallest size of each family")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    os.nice(args.nice)
    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2) // 2))
    torch.manual_seed(0)

    topo = Topology()
    nets = {
        "mlp 2x256": lambda: MLP(256, 2),
        "mlp 3x512": lambda: MLP(512, 3),
        "gnn d64 L4": lambda: GNN(topo, 64, 4),
        "gnn d64 L6": lambda: GNN(topo, 64, 6),
        "gnn d128 L4": lambda: GNN(topo, 128, 4),
        "gnn d128 L6": lambda: GNN(topo, 128, 6),
        "tf d64 L4": lambda: Transformer(topo, 64, 4),
        "tf d128 L4": lambda: Transformer(topo, 128, 4),
        "hyb d64 L4": lambda: Hybrid(topo, 64, 4),
        "hyb d128 L4": lambda: Hybrid(topo, 128, 4),
        "hyb d128 L6": lambda: Hybrid(topo, 128, 6),
    }
    if args.quick:
        nets = {k: f for k, f in nets.items() if k in ("mlp 2x256", "gnn d64 L4", "tf d64 L4", "hyb d64 L4")}

    print(f"device={args.device} threads={torch.get_num_threads()} infer_batch={args.infer_batch} train_batch={args.train_batch} "
          f"(chunks of {args.train_chunk}) epochs={args.epochs}")
    hdr = f"{'net':<13}{'params':>9}{'infer/s':>11}{'train/s':>10}{'ppo est/s':>11}{'vs mlp':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows, base = [], None
    for name, make in nets.items():
        net = make().to(args.device)
        params = sum(p.numel() for p in net.parameters())
        inf, trn = bench(net, args.infer_batch, args.train_batch, args.train_chunk, args.min_time, args.device)
        ppo = 1.0 / (1.0 / inf + args.epochs / trn)
        base = base or ppo
        rows.append({"net": name, "params": params, "infer_per_s": round(inf), "train_per_s": round(trn),
                     "ppo_est_per_s": round(ppo), "vs_mlp": round(ppo / base, 3)})
        print(f"{name:<13}{params:>9,}{inf:>11,.0f}{trn:>10,.0f}{ppo:>11,.0f}{ppo / base:>8.2f}", flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
