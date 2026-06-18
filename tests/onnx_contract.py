"""Shared ONNX contract checks for elbow-flexion submissions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

OBS_DIM = 9
ACTION_DIM = 6


class OnnxPolicy:
    """Minimal copy of the eval-harness ONNX runner (no myosuite needed)."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path))
        self._input_name = self._session.get_inputs()[0].name

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[np.newaxis, :]
        out = self._session.run(None, {self._input_name: obs_arr})[0]
        return np.asarray(out).squeeze().astype(np.float32)


def assert_elbow_onnx_contract(onnx_path: Path) -> None:
    """Validate an ONNX file matches the elbowflexion eval I/O contract."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    inputs = model.graph.input
    outputs = model.graph.output
    assert len(inputs) >= 1, "ONNX model must have at least one input"
    assert len(outputs) >= 1, "ONNX model must have at least one output"

    policy = OnnxPolicy(onnx_path)

    single = policy.get_action(np.zeros(OBS_DIM, dtype=np.float32))
    assert single.shape == (ACTION_DIM,), f"expected action ({ACTION_DIM},), got {single.shape}"

    batch = np.random.randn(4, OBS_DIM).astype(np.float32)
    batched = policy.get_action(batch)
    assert batched.shape == (4, ACTION_DIM), f"expected (4, {ACTION_DIM}), got {batched.shape}"

    clipped = np.clip(single, 0.0, 1.0)
    assert clipped.shape == (ACTION_DIM,)
