#!/usr/bin/env python3
"""Export a minimal valid ONNX model for elbowflexion (obs=9, action=6).

After export:
  evalmc submit-onnx elbow_model.onnx -t elbowflexion -p 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_elbow_onnx(obs_dim: int = 9, action_dim: int = 6) -> onnx.ModelProto:
    """Map obs -> constant 0.5 activations via Gemm (W=0, b=0.5)."""
    w = numpy_helper.from_array(np.zeros((obs_dim, action_dim), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.full(action_dim, 0.5, dtype=np.float32), name="b")
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, obs_dim])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, action_dim])
    gemm = helper.make_node("Gemm", ["obs", "W", "b"], ["action"])
    graph = helper.make_graph([gemm], "elbow_constant", [obs], [action], [w, b])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="mc26eval"
    )
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("elbow_model.onnx"), help="output path"
    )
    args = parser.parse_args()
    model = build_elbow_onnx()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Wrote {args.output} (input obs [batch, 9] -> action [batch, 6])")


if __name__ == "__main__":
    main()
