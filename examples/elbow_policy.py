"""Constant 0.5 activation policy for myoElbowPose1D6MRandom-v0 (6 muscles)."""

import numpy as np


class Policy:
    def reset(self):
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.full(6, 0.5, dtype=np.float32)
