"""Policy/value network and helpers to load it as a bot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from catan_rl import N_ACTIONS, OBS_SIZE

NEG_INF = -1e9


class PolicyNet(nn.Module):
    """MLP torso over the flat observation with a masked policy head and a value head."""

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
        logits = self.pi(h).masked_fill(~mask, NEG_INF)
        return logits, self.v(h).squeeze(-1)

    @torch.inference_mode()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        """Batched action selection from numpy arrays. Returns (actions, logp, value) as numpy."""
        o = torch.from_numpy(obs)
        m = torch.from_numpy(mask)
        logits, v = self(o, m)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(-1, a[:, None]).squeeze(-1)
        return a.numpy(), logp.numpy(), v.numpy()


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


def load(path: Path | str) -> PolicyNet | AZNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    net = AZNet(**{k: v for k, v in cfg.items() if k != "kind"}) if cfg.get("kind") == "az" else PolicyNet(**cfg)
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
