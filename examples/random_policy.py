"""Minimal random policy — useful as a submission baseline to verify the pipeline."""

import numpy as np


class Policy:
    def __init__(self):
        self.rng = np.random.default_rng()

    def reset(self):
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=39).astype(np.float32)
