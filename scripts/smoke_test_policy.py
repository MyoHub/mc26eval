#!/usr/bin/env python3
"""Smoke-test a local policy.py against a MyoSuite env (newbie onboarding).

Usage (from mc26eval/examples, with policy.py present):

  python ../scripts/smoke_test_policy.py --task elbowflexion
  python ../scripts/smoke_test_policy.py --task chasetag --max-steps 50

Requires MyoSuite (ms3) installed and importable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASKS = {
    "elbowflexion": {
        "env_id": "myoElbowPose1D6MRandom-v0",
        "config": "task_config.json",
        "obs_dim": 9,
        "action_dim": 6,
    },
    "chasetag": {
        "env_id": "myoChallengeChaseTagVsFullbody-v0",
        "config": "chasetag_task_config.json",
        "obs_dim": 546,
        "action_dim": 354,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="elbowflexion",
        help="Which task contract to exercise",
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing policy.py (default: cwd)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Cap episode length for a fast smoke test",
    )
    args = parser.parse_args()
    meta = TASKS[args.task]
    policy_dir = args.policy_dir.resolve()

    if not (policy_dir / "policy.py").is_file():
        print(f"ERROR: {policy_dir / 'policy.py'} not found", file=sys.stderr)
        print("Hint: cp random_policy.py policy.py  (from examples/)", file=sys.stderr)
        return 1

    cfg_path = policy_dir / meta["config"]
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        print(f"Loaded {cfg_path.name}: task={cfg.get('task')} env={cfg.get('env_id')}")
    else:
        print(f"Note: {cfg_path.name} missing — using built-in dims for checks only")

    try:
        import gymnasium as gym
        import myosuite  # noqa: F401
        import numpy as np
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print(
            "Install MyoSuite ms3: https://github.com/MyoHub/myosuite/tree/ms3",
            file=sys.stderr,
        )
        return 1

    if str(policy_dir) not in sys.path:
        sys.path.insert(0, str(policy_dir))

    from policy import Policy  # type: ignore  # noqa: E402

    myosuite.register_all_envs()
    env = gym.make(meta["env_id"])
    policy = Policy()

    obs, _ = env.reset()
    policy.reset()
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if obs.shape[0] != meta["obs_dim"]:
        print(
            f"WARNING: obs dim {obs.shape[0]} != expected {meta['obs_dim']} "
            f"for {args.task}"
        )

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < args.max_steps:
        action = np.asarray(policy.get_action(obs), dtype=np.float64).reshape(-1)
        if action.shape[0] != meta["action_dim"]:
            print(
                f"ERROR: action dim {action.shape[0]} != expected {meta['action_dim']}",
                file=sys.stderr,
            )
            env.close()
            return 1
        obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        steps += 1

    env.close()
    print(f"OK — {args.task}: {steps} steps on {meta['env_id']}")
    print("Next: bake policy.py (+ helpers/weights) into a Docker image and run evalmc push …")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
