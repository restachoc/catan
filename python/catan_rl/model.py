"""Policy/value network and helpers to load it as a bot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from catan_rl import ACTIONS, N_ACTIONS, OBS_SIZE
from catan_rl import graph as G

NEG_INF = -1e9


class PolicyNet(nn.Module):
    """MLP torso over the flat observation with a masked policy head and a value head."""

    amp = False  # fp16 autocast in act() on CUDA; set by the trainer, not saved

    def __init__(self, hidden: int = 512, layers: int = 3):
        super().__init__()
        self.cfg = {"hidden": hidden, "layers": layers}
        mods, d = [], OBS_SIZE
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        self.torso = nn.Sequential(*mods)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(obs)
        logits = self.pi(h).float().masked_fill(~mask, NEG_INF)  # fp32: -1e9 overflows fp16
        return logits, self.v(h).float().squeeze(-1)

    @torch.inference_mode()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        """Batched action selection from numpy arrays on the net's device. Returns (actions, logp, value) as numpy."""
        dev = next(self.parameters()).device
        o = torch.from_numpy(obs).to(dev)
        m = torch.from_numpy(mask).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=self.amp and dev.type == "cuda"):
            logits, v = self(o, m)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(-1, a[:, None]).squeeze(-1)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()


class GraphPolicyNet(nn.Module):
    """Board-structured PPO network with the same interface as PolicyNet.

    Hexes, vertices and edges are nodes of a typed GNN (`graph.GNNLayer`) with one global token for the
    players, own hand and phase. Each node also sees whether its own actions are legal (from the mask).
    Vertex embeddings score settlements and cities, edge embeddings roads, hex embeddings the robber,
    and the global token the remaining actions and the value.
    """

    amp = False

    def __init__(self, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.cfg = {"kind": "gnn", "hidden": hidden, "layers": layers}
        d = hidden
        topo = G.topology()
        self.register_buffer("vert_static", topo.vert_static, persistent=False)
        self.register_buffer("edge_static", topo.edge_static, persistent=False)
        self.register_buffer("perm", G.action_perm(), persistent=False)
        self.register_buffer("glob_act", G.global_actions(), persistent=False)
        self.adj = G.TopoBuffers(topo)
        self.enc_h = G.dense(G.HEX_F + 1, d)
        self.enc_v = G.dense(G.VERT_F + 2 + 2, d)
        self.enc_e = G.dense(G.EDGE_F + 1 + 1, d)
        self.enc_g = G.dense(G.GLOBAL_F + G.N_GLOBAL_ACT, d)
        self.layers = nn.ModuleList(G.GNNLayer(d) for _ in range(layers))
        self.pi_v, self.pi_e, self.pi_h = nn.Linear(d, 2), nn.Linear(d, 1), nn.Linear(d, 1)
        self.pi_g = nn.Linear(d, G.N_GLOBAL_ACT)
        self.v = nn.Sequential(G.dense(d, d), nn.Linear(d, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in (self.pi_v, self.pi_e, self.pi_h, self.pi_g):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.v[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, g = self.encode(obs, mask)
        x, g = self.trunk(x, g)
        return self.decode(x, g, mask)

    def encode(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Node embeddings, node-major {"h": (19, B, d), "v": (54, B, d), "e": (72, B, d)}, and the global token (B, d)."""
        B = obs.shape[0]
        h, v, e, g = G.split_obs(obs)
        m = mask.to(obs.dtype)
        s, c, r, rb = ACTIONS["SETTLE"], ACTIONS["CITY"], ACTIONS["ROAD"], ACTIONS["MOVE_ROBBER"]
        h = torch.cat([h, m[:, rb:rb + G.N_HEX, None]], -1)
        v = torch.cat([v, m[:, s:s + G.N_VERT, None], m[:, c:c + G.N_VERT, None],
                       self.vert_static.expand(B, -1, -1)], -1)
        e = torch.cat([e, m[:, r:r + G.N_EDGE, None], self.edge_static.expand(B, -1, -1)], -1)
        g = self.enc_g(torch.cat([g, m[:, self.glob_act]], -1))
        x = {k: enc(t).transpose(0, 1).contiguous()
             for k, enc, t in (("h", self.enc_h, h), ("v", self.enc_v, v), ("e", self.enc_e, e))}
        return x, g

    def trunk(self, x: dict[str, torch.Tensor], g: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        for layer in self.layers:
            x, g = layer(x, g, self.adj)
        return x, g

    def decode(self, x: dict[str, torch.Tensor], g: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pv = self.pi_v(x["v"])
        flat = torch.cat([pv[..., 0], pv[..., 1], self.pi_e(x["e"])[..., 0], self.pi_h(x["h"])[..., 0]], 0).T
        logits = torch.cat([flat, self.pi_g(g)], -1)[:, self.perm]
        return logits.float().masked_fill(~mask, NEG_INF), self.v(g).float().squeeze(-1)

    act = PolicyNet.act


class TransformerPolicyNet(GraphPolicyNet):
    """GraphPolicyNet's encoders and heads with full self-attention over all 146 tokens (hexes, vertices, edges,
    global) instead of message passing. Attention gets a learned per-head bias by graph distance between tokens
    (the global token is its own distance bucket) and a per-type embedding; there are no positional embeddings, so
    weights stay shared across locations as in the GNN.
    """

    MAX_D = 8  # distances beyond this share one bias

    def __init__(self, hidden: int = 64, layers: int = 4, heads: int = 4):
        super().__init__(hidden, 0)
        self.cfg = {"kind": "transformer", "hidden": hidden, "layers": layers, "heads": heads}
        d = hidden
        dist = torch.full((G.N_TOK + 1, G.N_TOK + 1), self.MAX_D + 1, dtype=torch.long)
        dist[:G.N_TOK, :G.N_TOK] = G.topology().dist.clamp(max=self.MAX_D)
        self.register_buffer("dist", dist, persistent=False)
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        self.attn_bias = nn.Parameter(torch.zeros(layers, heads, self.MAX_D + 2))
        self.blocks = nn.ModuleList(G.AttnBlock(d, heads) for _ in range(layers))
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def trunk(self, x: dict[str, torch.Tensor], g: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        te = self.type_emb
        t = torch.cat([x["h"] + te[0], x["v"] + te[1], x["e"] + te[2], (g + te[3])[None]], 0).transpose(0, 1)
        for blk, b in zip(self.blocks, self.attn_bias):
            t = blk(t, b[:, self.dist][None])
        t = t.transpose(0, 1)
        a, c = G.N_HEX, G.N_HEX + G.N_VERT
        return {"h": t[:a], "v": t[a:c], "e": t[c:G.N_TOK]}, t[G.N_TOK]


class GraphedPolicy:
    """CUDA-graph replay of a policy's forward for rollouts, with `act()` like PolicyNet.

    An eager GNN forward costs ~8 ms on a T4 whatever the batch (8 to 256 rows): the host launches ~350 small
    kernels per call, half of them autocast weight casts. Replaying a captured graph is one launch. Batches
    are padded to a bucket size, one graph per bucket, captured on first use. The graph reads the live
    weights, so in-place optimizer updates are picked up without recapturing.
    """

    BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096)

    def __init__(self, net: nn.Module):
        self.net = net
        self.graphs: dict[int, tuple] = {}

    def _capture(self, B: int) -> None:
        o = torch.zeros(B, OBS_SIZE, device="cuda")
        m = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device="cuda")
        m[:, 0] = True  # padding rows need one legal action
        amp = torch.autocast("cuda", dtype=torch.float16, enabled=self.net.amp)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), amp:
            for _ in range(3):  # warm-up on a side stream, as required before capture
                self.net(o, m)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), amp:
            out = self.net(o, m)
        self.graphs[B] = (g, o, m, out)

    @torch.inference_mode()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        n = len(obs)
        B = next((b for b in self.BUCKETS if b >= n), None)
        if B is None:
            return self.net.act(obs, mask, greedy)
        if B not in self.graphs:
            self._capture(B)
        g, o, m, (logits, v) = self.graphs[B]
        o[:n].copy_(torch.from_numpy(obs), non_blocking=True)
        m[:n].copy_(torch.from_numpy(mask), non_blocking=True)
        g.replay()
        lg = logits[:n]
        a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
        logp = torch.log_softmax(lg, -1).gather(-1, a[:, None]).squeeze(-1)
        out = torch.stack([a.float(), logp, v[:n]]).cpu().numpy()  # one device sync
        return out[0].astype(np.int64), out[1], out[2]


class AZNet(nn.Module):
    """AlphaZero network: MLP torso, masked policy head, and a value head predicting each seat's win
    probability (seat 0 = the player to act, others in turn order)."""

    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.cfg = {"kind": "az", "hidden": hidden, "layers": layers}
        mods, d = [], OBS_SIZE
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        self.torso = nn.Sequential(*mods)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 4)
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (masked policy logits, value logits over the 4 relative seats)."""
        h = self.torso(obs)
        return self.pi(h).masked_fill(~mask, NEG_INF), self.v(h)

    @torch.inference_mode()
    def evaluate(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Priors and win probabilities for MCTS leaves, as float32 numpy arrays."""
        logits, v = self(torch.from_numpy(obs), torch.from_numpy(mask))
        return torch.softmax(logits, -1).numpy(), torch.softmax(v, -1).numpy()

    @torch.inference_mode()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        """Raw-policy play (no search), same interface as PolicyNet.act."""
        logits, v = self(torch.from_numpy(obs), torch.from_numpy(mask))
        a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(-1, a[:, None]).squeeze(-1)
        return a.numpy(), logp.numpy(), torch.softmax(v, -1)[:, 0].numpy()


def save(net: nn.Module, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": net.state_dict(), "cfg": net.cfg, **meta}, tmp)
    tmp.replace(path)


NETS = {"mlp": PolicyNet, "gnn": GraphPolicyNet, "transformer": TransformerPolicyNet, "az": AZNet}


def make_policy(arch: str, hidden: int, layers: int) -> PolicyNet | GraphPolicyNet:
    return NETS[arch](hidden, layers)


def load(path: Path | str) -> PolicyNet | GraphPolicyNet | AZNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    net = NETS[cfg.pop("kind", "mlp")](**cfg)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


class PolicyBot:
    """Adapter so a trained network can play through the `Game` API (web UI, replays)."""

    def __init__(self, path: Path | str, greedy: bool = True):
        self.net = load(path)
        self.greedy = greedy

    def __call__(self, game) -> int:
        obs = game.observation()[None]
        mask = game.action_mask()[None]
        a, _, _ = self.net.act(obs, mask, greedy=self.greedy)
        return int(a[0])
