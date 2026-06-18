"""Constant mid-range activation baseline (any task via policy_utils)."""

import numpy as np

from policy_utils import get_action_dim


class Policy:
    def __init__(self):
        self._action_dim = get_action_dim()

    def reset(self):
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.full(self._action_dim, 0.5, dtype=np.float32)
