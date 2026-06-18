"""End-to-end: two separate users train, export ONNX, and submit.

Offline (default): embedded mcEvalProd SQLite API — no network.
Live: set LIVE_API_BASE=https://api.mc.myosuite.org

  pytest tests/test_e2e_two_user_submit.py -v
  LIVE_API_BASE=https://api.mc.myosuite.org pytest tests/test_e2e_two_user_submit.py -v -k live
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from e2e_api import TASK, PHASE, ApiClient, live_api_client, offline_api_client  # noqa: E402
from export_mjlab_onnx import ElbowActor, export_actor_to_elbow_onnx  # noqa: E402
from export_sb3_onnx import export_sb3_to_onnx  # noqa: E402
from onnx_contract import assert_elbow_onnx_contract  # noqa: E402

OBS_DIM = 9
ACTION_DIM = 6


class DummyElbowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(0.0, 1.0, (ACTION_DIM,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def _train_sb3_onnx(tmp_path: Path) -> Path:
    vec_env = DummyVecEnv([lambda: DummyElbowEnv()])
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=64, batch_size=32, seed=1)
    model.learn(total_timesteps=128)
    checkpoint = tmp_path / "sb3_elbow.zip"
    onnx_path = tmp_path / "sb3_elbow.onnx"
    model.save(str(checkpoint))
    vec_env.close()
    export_sb3_to_onnx(checkpoint, onnx_path)
    assert_elbow_onnx_contract(onnx_path)
    return onnx_path


def _train_mjlab_onnx(tmp_path: Path) -> Path:
    actor = ElbowActor()
    # Tiny "training" — random grad step so weights differ from init
    opt = th.optim.Adam(actor.parameters(), lr=1e-2)
    for _ in range(10):
        obs = th.randn(8, OBS_DIM)
        loss = actor(obs).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    onnx_path = tmp_path / "mjlab_elbow.onnx"
    export_actor_to_elbow_onnx(actor, onnx_path)
    assert_elbow_onnx_contract(onnx_path)
    return onnx_path


def _submit_as_user(api: ApiClient, prefix: str, onnx_path: Path) -> dict:
    uid = uuid.uuid4().hex[:10]
    email = f"{prefix}-{uid}@example.com"
    team = f"{prefix}-team-{uid}"
    api.ensure_task_phase()
    user = api.register(email, team)
    result = api.submit_onnx(user, onnx_path)
    assert result["submission_id"] > 0
    return {"user": user, "submission": result, "onnx": onnx_path}


@pytest.fixture(scope="session")
def offline_api():
    return offline_api_client()


def test_offline_sb3_user_trains_and_submits(tmp_path: Path, offline_api: ApiClient):
    onnx = _train_sb3_onnx(tmp_path)
    out = _submit_as_user(offline_api, "sb3-user", onnx)
    assert out["submission"]["submission_type"] == "onnx"


def test_offline_mjlab_user_trains_and_submits(tmp_path: Path, offline_api: ApiClient):
    onnx = _train_mjlab_onnx(tmp_path)
    out = _submit_as_user(offline_api, "mjlab-user", onnx)
    assert out["submission"]["submission_type"] == "onnx"


@pytest.fixture
def live_api():
    client = live_api_client()
    if client is None:
        pytest.skip("Set LIVE_API_BASE to run live submission tests")
    return client


@pytest.mark.live
def test_live_sb3_user_trains_and_submits(tmp_path: Path, live_api: ApiClient):
    onnx = _train_sb3_onnx(tmp_path)
    out = _submit_as_user(live_api, "sb3-user", onnx)
    print(f"SB3 user {out['user'].email} submission_id={out['submission']['submission_id']}")


@pytest.mark.live
def test_live_mjlab_user_trains_and_submits(tmp_path: Path, live_api: ApiClient):
    onnx = _train_mjlab_onnx(tmp_path)
    out = _submit_as_user(live_api, "mjlab-user", onnx)
    print(f"mjlab user {out['user'].email} submission_id={out['submission']['submission_id']}")
