"""Task-agnostic helpers for MyoChallenge policy examples.

Action size is read from JSON — no shell environment variables required.

During server evaluation the harness writes ``/shared/eval_meta.json``.
For local testing, bundle ``task_config.json`` next to ``policy.py`` (see
``task_config.json`` in this directory for elbow flexion).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_POLICY_DIR = Path(__file__).resolve().parent
_SHARED_META = Path("/shared/eval_meta.json")
_TASK_CONFIG = _POLICY_DIR / "task_config.json"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _action_dim_from_env_id(env_id: str) -> int:
    import gymnasium as gym
    import myosuite

    myosuite.register_all_envs()
    env = gym.make(env_id)
    try:
        return int(env.action_space.shape[0])
    finally:
        env.close()


def _action_dim_from_config(config: dict[str, Any]) -> int:
    if "action_dim" in config:
        return int(config["action_dim"])
    if env_id := config.get("env_id"):
        return _action_dim_from_env_id(str(env_id))
    raise KeyError("task_config.json must include 'action_dim' or 'env_id'")


@lru_cache(maxsize=1)
def get_action_dim() -> int:
    """Return the number of muscle activations for the current task."""
    if _SHARED_META.is_file():
        meta = _load_json(_SHARED_META)
        return int(meta["action_dim"])

    if _TASK_CONFIG.is_file():
        return _action_dim_from_config(_load_json(_TASK_CONFIG))

    raise RuntimeError(
        "Could not determine action dimension. During evaluation the harness "
        "writes /shared/eval_meta.json automatically. For local testing, add "
        "task_config.json beside policy.py (see mc26eval/examples/task_config.json)."
    )
