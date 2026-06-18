"""Isolated test: SB3 train → export → ONNX inference (no MyoSuite)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from export_sb3_onnx import export_sb3_to_onnx  # noqa: E402
from onnx_contract import assert_elbow_onnx_contract  # noqa: E402

OBS_DIM = 9
ACTION_DIM = 6


class DummyElbowEnv(gym.Env):
    """Elbow-shaped spaces without MuJoCo / MyoSuite."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(0.0, 1.0, (ACTION_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


def test_sb3_train_export_onnx_pipeline(tmp_path: Path):
    vec_env = DummyVecEnv([lambda: DummyElbowEnv()])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        n_steps=64,
        batch_size=32,
        learning_rate=3e-4,
        seed=0,
    )
    model.learn(total_timesteps=128)

    checkpoint = tmp_path / "elbow_ppo.zip"
    onnx_path = tmp_path / "elbow_sb3.onnx"
    model.save(str(checkpoint))
    vec_env.close()

    export_sb3_to_onnx(checkpoint, onnx_path)
    assert onnx_path.is_file()
    assert_elbow_onnx_contract(onnx_path)
