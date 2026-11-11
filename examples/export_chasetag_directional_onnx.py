#!/usr/bin/env python3
"""Export a tutorial 5.4 directional walker to ChaseTag ONNX (546→354).

On MyoSuite 3.1+, prefer the library helpers used by tutorial 5.6:

    from myosuite.integrations.musclemimic.fbvs_player import export_fbvs_onnx
    export_fbvs_onnx(checkpoint, "chasetag_player.onnx")

This script is a thin CLI wrapper around that path when available, with a
local adapter fallback for older checkouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Tutorial 5.4 / chasetag_teacher directional walker (.pt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chasetag_player.onnx"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from myosuite.integrations.musclemimic.fbvs_player import export_fbvs_onnx
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "myosuite.integrations",
            "myosuite.integrations.musclemimic",
            "myosuite.integrations.musclemimic.fbvs_player",
        }:
            raise
    else:
        export_fbvs_onnx(args.checkpoint, args.output)
        print(f"Wrote {args.output.resolve()} via export_fbvs_onnx (546→354)")
        return

    # Fallback for checkouts without fbvs_player (pre-3.1 API).
    import torch
    from torch import nn

    from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

    class FbvsDirectionalAdapter(nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            qpos, qvel = obs[:, 0:89], obs[:, 89:177]
            w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = torch.cos(-yaw), torch.sin(-yaw)
            roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
            vx, vy, vz = qvel[:, 0], qvel[:, 1], qvel[:, 2]
            wx, wy, wz = qvel[:, 3], qvel[:, 4], qvel[:, 5]
            delta = obs[:, 537:539]
            delta = torch.where(obs[:, 545:546] > 0.5, -delta, delta)
            heading = delta / torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(
                1e-6
            )
            tail = torch.stack(
                [
                    c * vx - s * vy,
                    s * vx + c * vy,
                    heading[:, 0],
                    heading[:, 1],
                    roll,
                    pitch,
                    c * wx - s * wy,
                    s * wx + c * wy,
                    wz,
                    vz,
                ],
                dim=-1,
            )
            return torch.cat([qpos[:, 7:], qvel[:, 6:], obs[:, 177:531], tail], dim=-1)

    class _ActorMean(nn.Module):
        def __init__(self, ac: ActorCritic) -> None:
            super().__init__()
            self.trunk = ac.trunk
            self.actor_mean = ac.actor_mean

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.actor_mean(self.trunk(obs))

    ac = ActorCritic.load(str(args.checkpoint), obs_dim=528)
    model = nn.Sequential(FbvsDirectionalAdapter(), _ActorMean(ac)).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 546),
        str(args.output),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Wrote {args.output.resolve()} via local adapter (546→354)")


if __name__ == "__main__":
    main()
