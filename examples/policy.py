"""
MyoChallenge 2026 — Policy contract.

Your submission zip MUST contain a file named policy.py at its root.
That file MUST define a class named Policy with the interface below.

The evaluation harness will:
  1. Add /policy (your unzipped submission) to sys.path
  2. Run:  from policy import Policy; p = Policy(); p.reset()
  3. Step: action = p.get_action(obs)  for every environment step

Any additional files in your zip (weights, configs, etc.) are available
relative to the same directory as policy.py.
"""

from pathlib import Path
import numpy as np


class Policy:
    """Minimal example — replace with your trained policy."""

    def __init__(self):
        # Load weights from a file bundled in the zip, e.g.:
        # weights_path = Path(__file__).parent / "weights.npz"
        # self.weights = np.load(weights_path)
        pass

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
        return np.zeros(39)  # placeholder: zero action
