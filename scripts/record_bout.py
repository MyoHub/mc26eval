#!/usr/bin/env python3
"""Play a ChaseTag ONNX policy (both agents) and save the bouts for viewer/index.html."""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
from myosuite import make_env

ENV_ID = "myoChallengeChaseTagFBVs-v0"
FRAME_STRIDE = 3  # 100 Hz control -> ~33 fps


def _policy(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda obs: np.clip(
        session.run(None, {name: np.asarray(obs, np.float32).reshape(1, -1)})[0].reshape(-1), 0.0, 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="submission ONNX (agent_0)")
    parser.add_argument("--opponent", type=Path, help="ONNX for agent_1 (default: the same model)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("-o", "--out", type=Path, default=Path("bout.bin.gz"))
    args = parser.parse_args()

    act0 = _policy(args.model)
    act1 = _policy(args.opponent or args.model)
    env = make_env(ENV_ID)
    raw = env.unwrapped
    clips, chunks = [], []
    for seed in args.seeds:
        obs, info = env.reset(seed=seed)
        frames = [raw.data.qpos.astype(np.float32).copy()]
        step = 0
        while True:
            obs, _, terminated, truncated, info = env.step(
                {"agent_0": act0(obs["agent_0"]), "agent_1": act1(obs["agent_1"])}
            )
            step += 1
            if step % FRAME_STRIDE == 0:
                frames.append(raw.data.qpos.astype(np.float32).copy())
            if terminated["agent_0"] or truncated["agent_0"]:
                break
        clips.append({
            "seed": seed,
            "agent_0_role": str(info["task"]),
            "tagged": bool(info["tagged"]),
            "elapsed_s": round(float(info["elapsed_s"]), 2),
            "score": round(float(info["score"]), 3),
            "frames": len(frames),
        })
        chunks.append(np.stack(frames))
        print(clips[-1])
    env.close()

    header = json.dumps({"nq": int(chunks[0].shape[1]), "fps": 100 / FRAME_STRIDE, "clips": clips}).encode()
    payload = struct.pack("<I", len(header)) + header + b"".join(c.tobytes() for c in chunks)
    args.out.write_bytes(gzip.compress(payload))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
