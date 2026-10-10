"""Board graph and message-passing blocks for board-structured networks.

The board has three node types: 19 hexes, 54 vertices and 72 edges. The graph is the same for every
layout (only tile contents differ), so it is built once from `Game.board_json()`. Weights are shared
across all nodes of a type and there are no positional embeddings, which is what lets a network trained
on one layout play another.
"""

from __future__ import annotations

import json
from collections import deque
from functools import cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from catan_rl import ACTIONS, N_ACTIONS, OBS_SIZE, Game

N_HEX, N_VERT, N_EDGE, MAX_P = 19, 54, 72, 4
HEX_F, VERT_F, EDGE_F, PLAYER_F = 8, 14, 4, 16  # per-node blocks of the flat observation (obs.rs)
BOARD_OBS = N_HEX * HEX_F + N_VERT * VERT_F + N_EDGE * EDGE_F
GLOBAL_F = OBS_SIZE - BOARD_OBS  # players, own hand/dev cards, globals
N_TOK = N_HEX + N_VERT + N_EDGE
N_GLOBAL_ACT = N_ACTIONS - 2 * N_VERT - N_EDGE - N_HEX


def _padded(lists: list[list[int]], pad: int) -> torch.Tensor:
    """Neighbour lists -> (N, K) index tensor; missing slots point at the zero row `pad`."""
    k = max(len(l) for l in lists)
    return torch.tensor([l + [pad] * (k - len(l)) for l in lists], dtype=torch.long)


class Topology:
    """Fixed board graph as padded neighbour indices per relation, static node features, and token distances."""

    def __init__(self):
        b = json.loads(Game(0, 4, 10, False, 500).board_json())
        hv = b["hex_vertices"]
        ev = b["edge_vertices"]
        v_h = [[] for _ in range(N_VERT)]
        v_e = [[] for _ in range(N_VERT)]
        for h, vs in enumerate(hv):
            for v in vs:
                v_h[v].append(h)
        for e, (a, c) in enumerate(ev):
            v_e[a].append(e)
            v_e[c].append(e)
        v_v = [[ev[e][0] + ev[e][1] - v for e in v_e[v]] for v in range(N_VERT)]
        e_e = [sorted({f for v in ev[e] for f in v_e[v] if f != e}) for e in range(N_EDGE)]
        h_e = [sorted({e for v in hv[h] for e in v_e[v] if set(ev[e]) <= set(hv[h])}) for h in range(N_HEX)]
        assert all(len(x) == 6 for x in h_e)
        e_h = [[h for h in range(N_HEX) if e in h_e[h]] for e in range(N_EDGE)]

        # relation "target<source" -> (N_target, K) neighbour indices; pad index = size of the source type
        self.rel = {
            "h<v": _padded(hv, N_VERT), "h<e": _padded(h_e, N_EDGE),
            "v<h": _padded(v_h, N_HEX), "v<v": _padded(v_v, N_VERT), "v<e": _padded(v_e, N_EDGE),
            "e<v": _padded([list(x) for x in ev], N_VERT), "e<e": _padded(e_e, N_EDGE),
        }
        # Mean aggregation hides how many neighbours a node has, so coast position is given explicitly.
        self.vert_static = torch.tensor([[len(v_h[v]) / 3, len(v_e[v]) / 3] for v in range(N_VERT)])
        self.edge_static = torch.tensor([[float(len(e_h[e]) == 1)] for e in range(N_EDGE)])

        # BFS distances over the token graph (hex-vertex and vertex-edge links), tokens ordered hex, vertex, edge
        adj = [[] for _ in range(N_TOK)]
        for h, vs in enumerate(hv):
            for v in vs:
                adj[h].append(N_HEX + v)
                adj[N_HEX + v].append(h)
        for e, (a, c) in enumerate(ev):
            for v in (a, c):
                adj[N_HEX + N_VERT + e].append(N_HEX + v)
                adj[N_HEX + v].append(N_HEX + N_VERT + e)
        dist = np.zeros((N_TOK, N_TOK), np.int64)
        for s in range(N_TOK):
            d = [-1] * N_TOK
            d[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if d[w] < 0:
                        d[w] = d[u] + 1
                        q.append(w)
            dist[s] = d
        self.dist = torch.from_numpy(dist)


@cache
def topology() -> Topology:
    return Topology()


def action_perm() -> torch.Tensor:
    """Index into cat([settle 54, city 54, road 72, robber 19, global rest]) giving flat action order."""
    spans = [("SETTLE", N_VERT), ("CITY", N_VERT), ("ROAD", N_EDGE), ("MOVE_ROBBER", N_HEX)]
    src = [-1] * N_ACTIONS
    pos = 0
    for name, n in spans:
        for i in range(n):
            src[ACTIONS[name] + i] = pos + i
        pos += n
    for a in range(N_ACTIONS):
        if src[a] < 0:
            src[a] = pos
            pos += 1
    assert pos == N_ACTIONS
    return torch.tensor(src)


def global_actions() -> torch.Tensor:
    """Flat ids of the actions not tied to a board location, in flat order (the global head's outputs)."""
    perm = action_perm()
    return (perm >= N_ACTIONS - N_GLOBAL_ACT).nonzero().squeeze(-1)


def split_obs(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat observation -> (hex (B,19,8), vertex (B,54,14), edge (B,72,4), global (B,GLOBAL_F))."""
    B = obs.shape[0]
    a = N_HEX * HEX_F
    b = a + N_VERT * VERT_F
    return (obs[:, :a].view(B, N_HEX, HEX_F), obs[:, a:b].view(B, N_VERT, VERT_F),
            obs[:, b:BOARD_OBS].view(B, N_EDGE, EDGE_F), obs[:, BOARD_OBS:])


def pip_targets(obs: torch.Tensor, v_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Auxiliary targets from the observation: pips per resource of every vertex (B, 54, 5; its up to 3 hexes summed)
    and of the whole board (B, 5). `v_h` is the (54, 3) vertex->hex index padded with N_HEX. The robber is ignored."""
    h = split_obs(obs)[0]
    res = h[..., :5] * (h[..., 6:7] * 5)  # desert (one-hot slot 5) drops out
    padded = torch.cat([res, res.new_zeros(res.shape[0], 1, 5)], 1)
    return padded[:, v_h].sum(2), res.sum(1)


def dense(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.ReLU())


class TopoBuffers(nn.Module):
    """Row-normalised dense adjacency (target x source) per relation, as buffers."""

    SIZE = {"h": N_HEX, "v": N_VERT, "e": N_EDGE}

    def __init__(self, topo: Topology):
        super().__init__()
        for k, idx in topo.rel.items():
            a = torch.zeros(self.SIZE[k[0]], self.SIZE[k[2]] + 1)
            a.scatter_(1, idx, 1.0)
            a = a[:, :-1]
            self.register_buffer(k, a / a.sum(1, keepdim=True), persistent=False)

    def __getitem__(self, k):
        return getattr(self, k)


class GNNLayer(nn.Module):
    """One round of typed message passing on node-major tensors (N, B, d).

    Equivalent to Linear(cat[self, neighbour mean per relation, global]) -> LayerNorm -> ReLU with a
    residual, computed as: project each source type once (self + all outgoing relations in one Linear),
    then average neighbours with a fixed row-normalised adjacency matrix as a single 2D GEMM per relation.
    The global token is then updated from itself and the mean of each node type.
    """

    OUT = {"h": ("h", "v<h"), "v": ("v", "h<v", "v<v", "e<v"), "e": ("e", "h<e", "v<e", "e<e")}  # source -> uses
    T = ("h", "v", "e")

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.proj = nn.ModuleDict({s: nn.Linear(d, d * len(u)) for s, u in self.OUT.items()})
        self.glob = nn.Linear(d, 3 * d, bias=False)
        self.ln = nn.ModuleDict({t: nn.LayerNorm(d) for t in self.T})
        self.upd_g = dense(4 * d, d)

    def forward(self, x: dict[str, torch.Tensor], g: torch.Tensor, adj: TopoBuffers):
        d, B = self.d, g.shape[0]
        pre = {}
        msgs = []  # (relation, projected source (N_src, B, d))
        for s, uses in self.OUT.items():
            y = self.proj[s](x[s])
            pre[s] = y[..., :d]
            msgs += [(r, y[..., (i + 1) * d:(i + 2) * d]) for i, r in enumerate(uses[1:])]
        gt = self.glob(g).view(B, 3, d)
        for r, y in msgs:
            t = r[0]
            n = y.shape[0]
            pre[t] = pre[t] + (adj[r] @ y.reshape(n, B * d)).view(-1, B, d)
        out = {t: x[t] + F.relu(self.ln[t](pre[t] + gt[:, i])) for i, t in enumerate(self.T)}
        g = g + self.upd_g(torch.cat([g, out["h"].mean(0), out["v"].mean(0), out["e"].mean(0)], -1))
        return out, g


class AttnBlock(nn.Module):
    """Pre-LN transformer block on batch-major tokens (B, T, d): self-attention with an additive bias, then an FFN."""

    def __init__(self, d: int, heads: int, ffn: int = 2):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn * d), nn.ReLU(), nn.Linear(ffn * d, d))

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.ln2(x))
