"""Minimal random policy — baseline to verify the Docker submission pipeline on any task.

Copy policy_utils.py and task_config.json into /policy alongside policy.py.
"""

import numpy as np

from policy_utils import get_action_dim


class Policy:
    def __init__(self):
        self.rng = np.random.default_rng()
        self._action_dim = get_action_dim()

    def reset(self):
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Harness clips actions to [0, 1] for muscle activations.
        return self.rng.uniform(0.0, 1.0, size=self._action_dim).astype(np.float32)
