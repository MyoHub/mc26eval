"""ChaseTag baseline: MuscleMimic directional BC policy (repo showcase).

Matches ``myosuite4/scripts/render_policies.py`` 1v1 protocol:
  - ``policy_bc_best.pt`` ActorCritic (528-dim obs, 354 act)
  - Motion-clip spawn so agents start upright
  - Heading command toward (chase) or away from (evade) the opponent

Zip layout:
  policy.py
  policy_bc_best.pt
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "policy_bc_best.pt"
FALL_Z = 0.6

_CLIP_FILES = [
    (
        "amathislab/musclemimic-retargeted",
        "MyoFullBody/gmr/KIT/4/WalkInClockwiseCircle01_poses.npz",
    ),
    (
        "amathislab/musclemimic-retargeted",
        "MyoFullBody/gmr/KIT/4/WalkInCounterClockwiseCircle04_poses.npz",
    ),
]


def _pelvis_yaw(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Policy:
    """Directional BC controller for one ChaseTag agent (agent_0 by default)."""

    AGENT_ID = "agent_0"

    def __init__(self):
        self._net = None
        self._clips = None
        self._action_dim = 354
        self._rng = np.random.default_rng(0)
        self._spawn_xy = {
            "agent_0": np.array([-0.025, 1.1], dtype=np.float64),
            "agent_1": np.array([0.025, -1.1], dtype=np.float64),
        }
        if CKPT.is_file():
            from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

            self._net = ActorCritic.load(str(CKPT), obs_dim=528)

    def reset(self):
        pass

    def on_episode_start(self, env, seed: int | None = None) -> None:
        """Clip-initialize both agents after env.reset (required for upright gait)."""
        if self._net is None:
            return
        raw = env.unwrapped
        model, data, meta = raw.model, raw.data, raw._meta
        if self._clips is None:
            self._clips = self._load_clips(model.nq // 2, model.nv // 2)
        rng = np.random.default_rng(seed)
        for aid in ("agent_0", "agent_1"):
            angle = float(rng.uniform(-math.pi, math.pi))
            clip, fi = self._best_frame(angle)
            self._init_agent(data, model, meta, aid, clip, fi, self._spawn_xy[aid])
        data.act[:] = 0.05
        import mujoco

        mujoco.mj_forward(model, data)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Fallback when harness only passes flat obs (no env) — low random."""
        if self._net is None:
            return self._rng.uniform(0.0, 0.15, size=self._action_dim).astype(np.float32)
        # Without env we cannot build the 528-dim directional obs correctly.
        return np.clip(self._net.act(np.zeros(528, np.float32)), 0.0, 1.0)

    def get_action_from_env(self, env, agent_id: str = "agent_0") -> np.ndarray:
        """Build directional obs + act, using live pursue/flee heading."""
        if self._net is None:
            return self.get_action(np.zeros(1))
        raw = env.unwrapped
        model, data, meta = raw.model, raw.data, raw._meta
        heading = self._steering_heading(data, meta, agent_id, raw)
        obs = self._agent_obs(data, model, meta, agent_id, heading)
        return np.clip(np.asarray(self._net.act(obs), dtype=np.float32), 0.0, 1.0)

    # ── helpers (from render_policies.py) ────────────────────────────────────

    def _load_clips(self, nq: int, nv: int) -> list:
        from huggingface_hub import hf_hub_download
        from myosuite.core.trajectory_io import load_motion_clip

        clips = []
        for repo, fname in _CLIP_FILES:
            p = Path(hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset"))
            clips.append(load_motion_clip(p, expected_nq=nq, expected_nv=nv))
        return clips

    def _best_frame(self, target_yaw: float):
        best_clip, best_fi, best_err = None, 0, float("inf")
        for clip in self._clips:
            for fi in range(len(clip.qpos)):
                err = abs(
                    math.atan2(
                        math.sin(_pelvis_yaw(clip.qpos[fi]) - target_yaw),
                        math.cos(_pelvis_yaw(clip.qpos[fi]) - target_yaw),
                    )
                )
                if err < best_err:
                    best_err, best_clip, best_fi = err, clip, fi
        return best_clip, best_fi

    @staticmethod
    def _init_agent(data, model, meta, aid: str, clip, fi: int, spawn_xy: np.ndarray):
        qp = clip.qpos[fi]
        qv = clip.qvel[fi]
        jnt_ids = meta.jnt_ids[aid]
        rqa = int(model.jnt_qposadr[jnt_ids[0]])
        rva = int(model.jnt_dofadr[jnt_ids[0]])
        lqs = int(model.jnt_qposadr[jnt_ids[1]])
        lqe = lqs + (model.nq // 2 - 7)
        lvs = int(model.jnt_dofadr[jnt_ids[1]])
        lve = lvs + (model.nv // 2 - 6)
        data.qpos[rqa : rqa + 2] = spawn_xy
        data.qpos[rqa + 2 : rqa + 7] = qp[2:7]
        data.qpos[lqs:lqe] = qp[7:]
        data.qvel[rva : rva + 6] = qv[:6]
        data.qvel[lvs:lve] = qv[6:]

    @staticmethod
    def _agent_obs(data, model, meta, aid: str, heading: np.ndarray) -> np.ndarray:
        jnt_ids = meta.jnt_ids[aid]
        rqa = int(model.jnt_qposadr[jnt_ids[0]])
        rva = int(model.jnt_dofadr[jnt_ids[0]])
        lqs = int(model.jnt_qposadr[jnt_ids[1]])
        lqe = lqs + (model.nq // 2 - 7)
        lvs = int(model.jnt_dofadr[jnt_ids[1]])
        lve = lvs + (model.nv // 2 - 6)

        rw, rx, ry, rz = data.qpos[rqa + 3 : rqa + 7]
        yaw = math.atan2(2 * (rw * rz + rx * ry), 1 - 2 * (ry * ry + rz * rz))
        roll = math.atan2(2 * (rw * rx + ry * rz), 1 - 2 * (rx * rx + ry * ry))
        pitch = math.asin(float(np.clip(2 * (rw * ry - rz * rx), -1, 1)))
        vx_w, vy_w = data.qvel[rva], data.qvel[rva + 1]
        vx_b = vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
        vy_b = -vx_w * math.sin(yaw) + vy_w * math.cos(yaw)
        vz = float(data.qvel[rva + 2])
        wx_b = float(data.qvel[rva + 3])
        wy_b = float(data.qvel[rva + 4])
        wz_w = float(data.qvel[rva + 5])

        return np.concatenate(
            [
                data.qpos[lqs:lqe].astype(np.float32),
                data.qvel[lvs:lve].astype(np.float32),
                data.act[meta.act_indices[aid]].astype(np.float32),
                np.array([vx_b, vy_b], np.float32),
                heading.astype(np.float32),
                np.array([roll, pitch, wx_b, wy_b, wz_w, vz], np.float32),
            ]
        )

    @staticmethod
    def _steering_heading(data, meta, agent_id: str, raw) -> np.ndarray:
        """Pursue if this agent is chaser this episode, else flee."""
        cfg = raw._config
        task = getattr(cfg, "current_task", "CHASE")
        is_chaser = (agent_id == "agent_0" and task == "CHASE") or (
            agent_id == "agent_1" and task == "EVADE"
        )
        opp = "agent_1" if agent_id == "agent_0" else "agent_0"
        own = data.site_xpos[meta.site_ids[agent_id]["pelvis_site"]][:2]
        other = data.site_xpos[meta.site_ids[opp]["pelvis_site"]][:2]
        rel = other - own
        if not is_chaser:
            rel = -rel
        n = float(np.linalg.norm(rel))
        if n < 1e-6:
            return np.array([1.0, 0.0], np.float32)
        return (rel / n).astype(np.float32)
