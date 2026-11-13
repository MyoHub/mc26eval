#!/usr/bin/env python3
"""Check ChaseTag ONNX shape contract and, when available, a real reset observation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit("Expected exactly one ONNX input and one output")
    input_shape = inputs[0].shape
    output_shape = outputs[0].shape
    if len(input_shape) != 2 or input_shape[-1] != 546:
        raise SystemExit(f"Expected input [batch, 546], got {input_shape}")
    if len(output_shape) != 2 or output_shape[-1] != 354:
        raise SystemExit(f"Expected output [batch, 354], got {output_shape}")

    try:
        import gymnasium as gym
        import myosuite
    except ImportError:
        print("OK — ChaseTag ONNX metadata contract [batch, 546] → [batch, 354]; "
              "runtime skipped (install MyoSuite 3.1+ for an environment-backed check)")
        return

    myosuite.register_all_envs()
    env = gym.make("myoChallengeChaseTagFBVs-v0", disable_env_checker=True)
    try:
        obs, _ = env.reset(seed=0)
        batch = np.asarray(obs["agent_0"], dtype=np.float32).reshape(1, 546)
        action = session.run(None, {inputs[0].name: batch})[0]
    finally:
        env.close()
    if action.shape != (1, 354) or not np.isfinite(action).all():
        raise SystemExit(f"Invalid output: shape={action.shape}, finite={np.isfinite(action).all()}")
    print("OK — ChaseTag ONNX contract [batch, 546] → [batch, 354]")


if __name__ == "__main__":
    main()
