#!/usr/bin/env python3
"""Run one ChaseTag episode with a local policy.py wrapper (local debugging only).

Submissions are ONNX models only; this script does not test the submission path.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=Path.cwd())
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    policy_path = args.policy_dir.resolve() / "policy.py"
    if not policy_path.is_file():
        parser.error(f"policy.py not found: {policy_path}")

    try:
        import gymnasium as gym
        import myosuite
        import numpy as np
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install MyoSuite 3.1+.", file=sys.stderr)
        return 1

    spec = importlib.util.spec_from_file_location("submission_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {policy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(policy_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(policy_path.parent))
    policy = module.Policy()

    myosuite.register_all_envs()
    env = gym.make("myoChallengeChaseTagFBVs-v0", disable_env_checker=True)
    try:
        obs, _ = env.reset(seed=0)
        policy.reset()
        steps = 0
        for steps in range(1, args.max_steps + 1):
            action = np.asarray(policy.get_action(obs["agent_0"]), dtype=np.float32).reshape(-1)
            if action.shape != (354,):
                raise ValueError(f"Expected 354 actions, got {action.shape}")
            actions = {"agent_0": np.clip(action, 0, 1), "agent_1": np.zeros(354, np.float32)}
            obs, _, terminated, truncated, _info = env.step(actions)
            if any(terminated.values()) or any(truncated.values()):
                break
        print(f"ChaseTag policy ran for {steps} steps")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
