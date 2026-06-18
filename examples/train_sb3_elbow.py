#!/usr/bin/env python3
"""Train a PPO policy for elbow flexion with Stable-Baselines3.

Requires MyoSuite ms3:
  git clone -b ms3 https://github.com/MyoHub/myosuite.git && pip install -e .

Usage:
  python train_sb3_elbow.py --timesteps 200000 --save-dir runs/elbow_sb3
  python export_sb3_onnx.py --checkpoint runs/elbow_sb3/elbow_ppo.zip -o elbow_model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "myoElbowPose1D6MRandom-v0"


def make_env():
    myosuite.register_all_envs()
    return Monitor(gym.make(ENV_ID))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/elbow_sb3"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps)

    checkpoint = args.save_dir / "elbow_ppo.zip"
    model.save(str(checkpoint))
    vec_env.close()
    print(f"Saved {checkpoint}")
    print(f"Export: python export_sb3_onnx.py --checkpoint {checkpoint} -o elbow_model.onnx")


if __name__ == "__main__":
    main()
