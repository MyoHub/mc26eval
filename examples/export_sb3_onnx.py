#!/usr/bin/env python3
"""Export a Stable-Baselines3 PPO checkpoint to elbow-flexion ONNX.

Output contract (MyoChallenge eval):
  input  obs    [batch, 9]  float32
  output action [batch, 6]  float32

Usage:
  python export_sb3_onnx.py --checkpoint runs/elbow_sb3/elbow_ppo.zip -o elbow_model.onnx
  evalmc submit-onnx elbow_model.onnx -t elbowflexion -p 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch as th
from stable_baselines3 import PPO

OBS_DIM = 9
ACTION_DIM = 6


class OnnxablePolicy(th.nn.Module):
    """Wrap SB3 actor for ONNX — returns actions only."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.policy(obs, deterministic=True)[0]


def export_sb3_to_onnx(checkpoint: Path, output: Path, obs_dim: int = OBS_DIM) -> None:
    model = PPO.load(str(checkpoint), device="cpu")
    onnx_policy = OnnxablePolicy(model.policy)
    onnx_policy.eval()

    dummy = th.randn(1, obs_dim, dtype=th.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    th.onnx.export(
        onnx_policy,
        dummy,
        str(output),
        opset_version=17,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    print(f"Wrote {output} (obs [batch, {obs_dim}] -> action [batch, {ACTION_DIM}])")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("elbow_model.onnx"))
    parser.add_argument("--obs-dim", type=int, default=OBS_DIM)
    args = parser.parse_args()
    export_sb3_to_onnx(args.checkpoint, args.output, obs_dim=args.obs_dim)


if __name__ == "__main__":
    main()
