"""Play Catan in the browser against bots, or watch saved replays.

    uvicorn web.server:app --reload      (from the repo root, with the venv active)

The server owns the Rust `Game`; the browser only renders state and sends action ids chosen from
the legal list the server provides. The UI never implements rules.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from catan_rl import ACTION_NAMES, ACTIONS, Game

ROOT = Path(__file__).resolve().parent
REPLAY_DIR = ROOT.parent / "replays"

app = FastAPI(title="Catan")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# Pluggable bot registry: name -> callable(game) -> action. RL policies register here later.
BOTS = {
    "heuristic": lambda g: g.bot_action("heuristic"),
    "random": lambda g: g.bot_action("random"),
}


def register_bot(name: str, fn) -> None:
    BOTS[name] = fn


def register_policies() -> None:
    """Expose trained checkpoints as bots: $CATAN_POLICY, plus runs/<name>/best.pt for each run."""
    try:
        from catan_rl.model import PolicyBot
    except ImportError:  # torch not installed
        return
    paths = {f"ppo:{p.parent.name}": p for p in sorted((ROOT.parent / "runs").glob("*/best.pt"))}
    if os.environ.get("CATAN_POLICY"):
        paths["ppo"] = Path(os.environ["CATAN_POLICY"])
    for name, path in paths.items():
        try:
            register_bot(name, PolicyBot(path, greedy=False))
        except RuntimeError as e:  # trained on an older observation/action layout
            print(f"skipping {name}: incompatible checkpoint ({str(e).splitlines()[0]})")


register_policies()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {"actions": ACTIONS, "action_names": ACTION_NAMES, "bots": sorted(BOTS)}


@app.get("/api/replays")
def list_replays():
    if not REPLAY_DIR.exists():
        return []
    return sorted((p.stem for p in REPLAY_DIR.glob("*.json")), reverse=True)


@app.get("/api/replays/{name}")
def get_replay(name: str):
    path = (REPLAY_DIR / f"{name}.json").resolve()
    if path.parent != REPLAY_DIR.resolve() or not path.exists():
        raise HTTPException(404)
    rep = json.loads(path.read_text())
    cfg = rep.get("config", {})
    g = Game(seed=rep["seed"], **cfg)
    frames = [json.loads(g.state_json(None))]
    log = []
    for a in rep["actions"]:
        actor = g.actor
        g.step(a)
        st = json.loads(g.state_json(None))
        frames.append(st)
        log.append(describe(actor, a, st))
    return {
        "board": json.loads(g.board_json()),
        "frames": frames,
        "log": log,
        "players": rep.get("players", []),
    }


PRETTY = {
    "settlement": "built a settlement",
    "city": "built a city",
    "road": "built a road",
    "robber": "moved the robber",
    "buy_dev": "bought a development card",
    "play_knight": "played a knight",
    "play_road_building": "played road building",
    "end_turn": "ended their turn",
}


def describe(actor: int, a: int, st: dict) -> dict:
    """One log line for action `a` taken by `actor`; `st` is the state after the action."""
    name = ACTION_NAMES[a]
    head, _, arg = name.replace("@", ":").partition(":")
    if a == ACTIONS["ROLL"]:
        d = st["last_roll"]
        text = f"rolled {d[0]} + {d[1]} = {d[0] + d[1]}"
    elif head in PRETTY:
        text = PRETTY[head]
    elif head == "steal":
        text = f"stole from P{(actor + int(arg)) % st['n_players']}"
    elif head == "trade":
        give, get = arg.split("->")
        text = f"traded {give} for {get} with the bank"
    elif head == "monopoly":
        text = f"played monopoly on {arg}"
    elif head == "year_of_plenty":
        text = f"played year of plenty ({arg.replace('+', ' + ')})"
    elif head == "discard":
        text = f"discarded {arg}"
    else:
        text = name
    return {"player": actor, "action": a, "text": text}


class Session:
    def __init__(self, ws: WebSocket, seed: int, human: int, bots: list[str], n_players: int, delay: float):
        self.ws = ws
        self.game = Game(seed=seed, n_players=n_players)
        self.human = human
        self.bots = bots  # per seat; ignored for the human seat
        self.delay = delay

    async def send_state(self, log_entry: dict | None = None):
        await self.ws.send_json(
            {
                "type": "state",
                "state": json.loads(self.game.state_json(self.human)),
                "log": log_entry,
            }
        )

    async def apply(self, a: int):
        actor = self.game.actor
        self.game.step(a)
        st = json.loads(self.game.state_json(self.human))
        await self.ws.send_json({"type": "state", "state": st, "log": describe(actor, a, st)})
        if self.game.is_over:
            self.save_replay()

    def save_replay(self):
        REPLAY_DIR.mkdir(exist_ok=True)
        rep = {
            "seed": self.game.seed,
            "config": {"n_players": self.game.n_players},
            "actions": self.game.history,
            "players": ["you" if i == self.human else self.bots[i] for i in range(self.game.n_players)],
            "winner": self.game.winner,
        }
        name = f"human-{self.game.seed}"
        (REPLAY_DIR / f"{name}.json").write_text(json.dumps(rep))

    async def run_bots(self):
        """Let bot seats act until it is the human's move or the game ends."""
        while not self.game.is_over and self.game.actor != self.human:
            await asyncio.sleep(self.delay)
            bot = BOTS[self.bots[self.game.actor]]
            await self.apply(bot(self.game))


@app.websocket("/ws")
async def play(ws: WebSocket):
    await ws.accept()
    session: Session | None = None
    try:
        while True:
            msg = await ws.receive_json()
            if msg["type"] == "new":
                n = int(msg.get("n_players", 4))
                seed = int(msg.get("seed") or random.randrange(1 << 31))
                bot = msg.get("bot", "heuristic")
                if bot not in BOTS:
                    bot = "heuristic"
                session = Session(ws, seed, 0, [bot] * n, n, float(msg.get("delay", 0.35)))
                await ws.send_json({"type": "board", "board": json.loads(session.game.board_json()), "seed": seed})
                await session.send_state()
                await session.run_bots()
            elif msg["type"] == "action" and session is not None:
                if session.game.actor != session.human or session.game.is_over:
                    continue
                try:
                    await session.apply(int(msg["action"]))
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
                await session.run_bots()
            elif msg["type"] == "speed" and session is not None:
                session.delay = float(msg["delay"])
    except WebSocketDisconnect:
        pass
