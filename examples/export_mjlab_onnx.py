#!/usr/bin/env python3
"""Export an RSL-RL / mjlab-style actor to elbow-flexion ONNX.

Uses mjlab_myosuite helpers when installed; otherwise falls back to a
direct PyTorch ONNX export with the same I/O contract.

Usage:
  python export_mjlab_onnx.py --output elbow_model.onnx
  evalmc submit-onnx elbow_model.onnx -t elbowflexion -p 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch as th

OBS_DIM = 9
ACTION_DIM = 6


class ElbowActor(th.nn.Module):
    """Minimal actor head matching RSL-RL / mjlab MLP layout."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.is_recurrent = False
        self.actor = th.nn.Sequential(
            th.nn.Linear(obs_dim, 64),
            th.nn.ELU(),
            th.nn.Linear(64, action_dim),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.actor(obs)


def export_actor_to_elbow_onnx(
    actor: th.nn.Module,
    output: Path,
    obs_dim: int = OBS_DIM,
    *,
    input_name: str = "obs",
    output_name: str = "action",
) -> None:
    """Direct PyTorch export — same contract as eval harness."""
    actor.eval()
    dummy = th.randn(1, obs_dim, dtype=th.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        actor,
        dummy,
        str(output),
        opset_version=17,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        dynamo=False,
    )


def export_with_myosuite_helpers(
    actor: th.nn.Module,
    output_dir: Path,
    filename: str = "elbow_model.onnx",
    env=None,
    run_path: str = "local_export",
) -> Path:
    """Export via mjlab_myosuite when available."""
    from mjlab_myosuite.rl.exporter import (
        attach_myosuite_onnx_metadata,
        export_myosuite_policy_as_onnx,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    export_myosuite_policy_as_onnx(
        actor_critic=actor,
        normalizer=None,
        path=str(output_dir),
        filename=filename,
        verbose=False,
    )
    if env is not None:
        attach_myosuite_onnx_metadata(
            env=env,
            run_path=run_path,
            path=str(output_dir),
            filename=filename,
        )
    return output_dir / filename


def export_elbow_onnx(
    output: Path,
    *,
    use_myosuite_helpers: bool = True,
    env=None,
) -> Path:
    """Export a fresh actor; prefer mjlab_myosuite helpers when installed."""
    actor = ElbowActor()

    if use_myosuite_helpers:
        try:
            return export_with_myosuite_helpers(
                actor, output.parent, filename=output.name, env=env
            )
        except ImportError:
            pass

    export_actor_to_elbow_onnx(actor, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("elbow_model.onnx"))
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Skip mjlab_myosuite helpers even if installed",
    )
    args = parser.parse_args()

    path = export_elbow_onnx(
        args.output,
        use_myosuite_helpers=not args.direct_only,
    )
    print(f"Wrote {path} (obs [batch, {OBS_DIM}] -> action [batch, {ACTION_DIM}])")


if __name__ == "__main__":
    main()
