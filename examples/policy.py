"""
MyoChallenge 2026 — Policy contract (Docker agent images).

Your Docker agent image MUST expose a file named policy.py under /policy or /app.
That file MUST define a class named Policy with the interface below.

The evaluation harness will:
  1. Add /policy (your agent mount) to sys.path
  2. Run:  from policy import Policy; p = Policy(); p.reset()
  3. Step: action = p.get_action(obs)  for every environment step

Any additional files in the image next to policy.py (weights, configs, your own
.py modules) are available relative to the same directory as policy.py.

Dependencies:
  - Libraries come from the *task eval image* and/or packages you install in the
    agent Dockerfile — not from a bare requirements.txt at score time.
  - Do not ship a host venv/site-packages or expect requirements.txt to be installed.
  - Allowed imports per task: tutorials/SUBMISSION_DEPENDENCIES.md
  - Build/push guide: DOCKER_SUBMISSION.md
  - First-time submit: tutorials/NEWBIE_QUICKSTART.md
"""

from pathlib import Path

import numpy as np

from policy_utils import get_action_dim


class Policy:
    """Minimal example — replace with your trained policy."""

    def __init__(self):
        # Load weights from a file bundled next to policy.py, e.g.:
        # weights_path = Path(__file__).parent / "weights.npz"
        # self.weights = np.load(weights_path)
        self._action_dim = get_action_dim()

    def reset(self):
        """Called at the start of every episode."""
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return an action given the current observation.

        Args:
            obs: 1-D numpy array, shape defined by the task environment.

        Returns:
            action: 1-D numpy array, shape defined by the task environment.
        """
        # TODO: replace with your policy inference
        return np.zeros(self._action_dim, dtype=np.float32)
