"""ChaseTag house-style baseline: flat ONNX player (MyoSuite 3.1+ FBVs).

Place ``model.onnx`` next to this file (546 → 354). Prefer exporting with
MyoSuite's ``export_fbvs_onnx`` / tutorial 5.6 rather than shipping Torch.

Zip / Docker layout:
  policy.py
  model.onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "model.onnx"


class Policy:
    """Flat obs→action ONNX wrapper for one ChaseTag agent."""

    def __init__(self):
        if not ONNX_PATH.is_file():
            raise FileNotFoundError(f"ChaseTag ONNX model missing: {ONNX_PATH}")
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def reset(self):
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[np.newaxis, :]
        out = self._session.run(None, {self._input_name: obs_arr})[0]
        return np.asarray(out).squeeze().astype(np.float32)
