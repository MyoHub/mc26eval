#!/usr/bin/env python3
"""Smoke-test an ONNX file against the MyoChallenge I/O contract.

Usage:

  python scripts/smoke_test_onnx.py examples/elbow_model.onnx --task elbowflexion
  python scripts/smoke_test_onnx.py chase.onnx --task chasetag

Requires: pip install onnxruntime numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TASKS = {
    "elbowflexion": {"obs_dim": 9, "action_dim": 6},
    "chasetag": {"obs_dim": 546, "action_dim": 354},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path, help="Path to .onnx model")
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="elbowflexion",
        help="Which I/O contract to check",
    )
    args = parser.parse_args()
    meta = TASKS[args.task]

    if not args.onnx_path.is_file():
        print(f"ERROR: file not found: {args.onnx_path}", file=sys.stderr)
        return 1

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Fix: pip install onnxruntime numpy", file=sys.stderr)
        return 1

    try:
        session = ort.InferenceSession(
            str(args.onnx_path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"ERROR: could not load ONNX: {exc}", file=sys.stderr)
        if "IR version" in msg or "Unsupported model" in msg:
            print(
                "Hint: upgrade onnxruntime (pip install -U onnxruntime) — "
                "your install is older than the exported model IR.",
                file=sys.stderr,
            )
        return 1
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        print("ERROR: model has no inputs/outputs", file=sys.stderr)
        return 1

    in_name = inputs[0].name
    obs = np.random.randn(1, meta["obs_dim"]).astype(np.float32)
    try:
        out = session.run(None, {in_name: obs})[0]
    except Exception as exc:  # noqa: BLE001 — newbie-facing diagnostic
        print(f"ERROR: inference failed: {exc}", file=sys.stderr)
        print(
            f"Hint: expected input '{in_name}' with shape [1, {meta['obs_dim']}]",
            file=sys.stderr,
        )
        return 1

    out = np.asarray(out)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.shape[-1] != meta["action_dim"]:
        print(
            f"ERROR: action last-dim {out.shape[-1]} != expected {meta['action_dim']}",
            file=sys.stderr,
        )
        return 1

    print(f"OK — {args.task}: input '{in_name}' {tuple(obs.shape)} → action {tuple(out.shape)}")
    print(f"Next: evalmc submit-onnx {args.onnx_path} -t {args.task} -p 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
