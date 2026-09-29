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


def save(net: PolicyNet, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": net.state_dict(), "cfg": net.cfg, **meta}, tmp)
    tmp.replace(path)


def load(path: Path | str) -> PolicyNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyNet(**ck["cfg"])
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
