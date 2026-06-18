"""Isolated test: mjlab-style actor → ONNX export → inference (no MyoSuite env)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch as th

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from export_mjlab_onnx import (  # noqa: E402
    ElbowActor,
    export_actor_to_elbow_onnx,
    export_with_myosuite_helpers,
)
from onnx_contract import assert_elbow_onnx_contract  # noqa: E402


class _MockMyoSuiteEnv:
    """Minimal env stub for attach_myosuite_onnx_metadata."""

    num_envs = 1
    device = "cpu"
    single_action_space = type("S", (), {"shape": (6,), "low": [0.0] * 6, "high": [1.0] * 6})()
    single_observation_space = type("S", (), {"shape": (9,)})()
    spec = type("S", (), {"id": "Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0"})()


def test_mjlab_direct_actor_export_onnx(tmp_path: Path):
    """Direct torch export path (no mjlab install required)."""
    actor = ElbowActor()
    onnx_path = tmp_path / "elbow_mjlab_direct.onnx"
    export_actor_to_elbow_onnx(actor, onnx_path)
    assert_elbow_onnx_contract(onnx_path)


def test_mjlab_myosuite_helper_export_onnx(tmp_path: Path):
    """mjlab_myosuite helper path when mjlab + onnx stack is installed."""
    pytest.importorskip("mjlab_myosuite")
    try:
        from mjlab.utils.lab_api.rl.exporter import export_policy_as_onnx  # noqa: F401
    except ImportError:
        try:
            from mjlab.rl.exporter_utils import export_policy_as_onnx  # noqa: F401
        except ImportError:
            pytest.skip("mjlab ONNX export utilities not installed")

    actor = ElbowActor()
    onnx_path = export_with_myosuite_helpers(
        actor,
        tmp_path,
        filename="elbow_mjlab_helpers.onnx",
        env=_MockMyoSuiteEnv(),
        run_path="pytest",
    )
    assert onnx_path.is_file()
    assert_elbow_onnx_contract(onnx_path)


def test_mjlab_export_elbow_onnx_auto(tmp_path: Path):
    """export_elbow_onnx() picks helpers or direct fallback."""
    from export_mjlab_onnx import export_elbow_onnx

    onnx_path = tmp_path / "elbow_auto.onnx"
    export_elbow_onnx(onnx_path, use_myosuite_helpers=False)
    assert_elbow_onnx_contract(onnx_path)
