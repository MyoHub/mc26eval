#!/usr/bin/env python3
"""Check ChaseTag ONNX shape contract and, when available, a real reset observation.

Recurrent models are accepted: every extra input ``<x>_in`` needs a matching
output ``<x>_out`` with the same static shape; state starts at zeros.
"""

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
    out_by_name = {o.name: o for o in outputs}
    state = {}
    state_pairs = []
    for inp in inputs[1:]:
        if not inp.name.endswith("_in"):
            raise SystemExit(
                f"Extra input '{inp.name}' must be recurrent state named '<x>_in'"
            )
        out_name = inp.name[: -len("_in")] + "_out"
        if out_name not in out_by_name:
            raise SystemExit(f"State input '{inp.name}' has no matching output '{out_name}'")
        shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
        state[inp.name] = np.zeros(shape, dtype=np.float32)
        state_pairs.append((inp.name, out_name))
    kind = f"recurrent, state {[p[0] for p in state_pairs]}" if state_pairs else "stateless"
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
              f"({kind}); runtime skipped (install MyoSuite 3.1+ for an environment-backed check)")
        return

    myosuite.register_all_envs()
    env = gym.make("myoChallengeChaseTagFBVs-v0", disable_env_checker=True)
    try:
        obs, _ = env.reset(seed=0)
        batch = np.asarray(obs["agent_0"], dtype=np.float32).reshape(1, 546)
        fetch = [outputs[0].name] + [o for _i, o in state_pairs]
        for _ in range(3):  # a few steps so state is fed back
            outs = session.run(fetch, {inputs[0].name: batch, **state})
            for (i_name, _o), value in zip(state_pairs, outs[1:]):
                if value.shape != state[i_name].shape:
                    raise SystemExit(f"State '{_o}' shape {value.shape} != {state[i_name].shape}")
                state[i_name] = value
        action = outs[0]
    finally:
        env.close()
    if action.shape != (1, 354) or not np.isfinite(action).all():
        raise SystemExit(f"Invalid output: shape={action.shape}, finite={np.isfinite(action).all()}")
    print(f"OK — ChaseTag ONNX contract [batch, 546] → [batch, 354] ({kind})")


if __name__ == "__main__":
    main()
