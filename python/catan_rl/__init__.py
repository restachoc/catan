"""Catan RL: fast Rust engine bindings, bots, training and evaluation."""

from catan_rl._engine import N_ACTIONS, OBS_SIZE, Game, VecEnv, action_names, bot_tournament

ACTION_NAMES = action_names()


def _first(prefix: str) -> int:
    return next(i for i, n in enumerate(ACTION_NAMES) if n.startswith(prefix))


# Offsets of each action family in the flat action space (mirrors engine/catan-core/src/actions.rs).
ACTIONS = {
    "ROLL": _first("roll"),
    "END_TURN": _first("end_turn"),
    "SETTLE": _first("settlement@"),
    "CITY": _first("city@"),
    "ROAD": _first("road@"),
    "BUY_DEV": _first("buy_dev"),
    "PLAY_KNIGHT": _first("play_knight"),
    "PLAY_ROAD_BUILDING": _first("play_road_building"),
    "PLAY_MONOPOLY": _first("monopoly:"),
    "PLAY_YOP": _first("year_of_plenty:"),
    "MOVE_ROBBER": _first("robber@"),
    "STEAL": _first("steal:"),
    "DISCARD": _first("discard:"),
    "TRADE": _first("trade:"),
    "N_ACTIONS": N_ACTIONS,
}

__all__ = ["Game", "VecEnv", "bot_tournament", "ACTIONS", "ACTION_NAMES", "N_ACTIONS", "OBS_SIZE"]
