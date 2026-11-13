#!/usr/bin/env python3
"""Write the web viewer's body model: a kinematics-only MuJoCo XML plus decimated STL meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np
import trimesh
from myosuite import make_env

ENV_ID = "myoChallengeChaseTagFBVs-v0"
TINT = {"a0": (1.0, 0.55, 0.25, 1.0), "a1": (0.3, 0.55, 1.0, 1.0)}


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _write_mesh(model: mujoco.MjModel, mesh_id: int, out: Path, keep: float) -> int:
    v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    verts = model.mesh_vert[v0 : v0 + nv].astype(np.float64)
    faces = model.mesh_face[f0 : f0 + nf].astype(np.int32)
    if nf > 200:
        verts, faces = fast_simplification.simplify(verts, faces, target_reduction=1 - keep)
    trimesh.Trimesh(verts, faces, process=False).export(out)
    return len(faces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=Path("viewer/scene"))
    parser.add_argument("--keep", type=float, default=0.08, help="fraction of faces kept")
    args = parser.parse_args()

    model = make_env(ENV_ID).unwrapped.model
    args.out.mkdir(parents=True, exist_ok=True)
    name = lambda kind, i: mujoco.mj_id2name(model, kind, i)

    mesh_ids = sorted({int(model.geom_dataid[g]) for g in range(model.ngeom)
                       if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 0})
    faces = sum(_write_mesh(model, i, args.out / f"m{i}.stl", args.keep) for i in mesh_ids)

    geoms: dict[int, list[str]] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 0:
            continue
        tint = TINT.get((name(mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")[:2], (0.7, 0.7, 0.7, 1.0))
        geoms.setdefault(int(model.geom_bodyid[g]), []).append(
            f'<geom type="mesh" mesh="m{model.geom_dataid[g]}" pos="{_fmt(model.geom_pos[g])}" '
            f'quat="{_fmt(model.geom_quat[g])}" rgba="{_fmt(tint)}" contype="0" conaffinity="0"/>'
        )

    joints: dict[int, list[str]] = {}
    for j in range(model.njnt):
        kind = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[int(model.jnt_type[j])]
        ref = f' ref="{model.qpos0[model.jnt_qposadr[j]]:.6g}"' if kind in ("slide", "hinge") else ""
        axis = f' axis="{_fmt(model.jnt_axis[j])}"' if kind in ("slide", "hinge") else ""
        joints.setdefault(int(model.jnt_bodyid[j]), []).append(
            f'<joint type="{kind}" pos="{_fmt(model.jnt_pos[j])}"{axis}{ref}/>'
        )

    children: dict[int, list[int]] = {}
    for b in range(1, model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)

    def body_xml(b: int, depth: int) -> str:
        pad = "  " * depth
        lines = [f'{pad}<body pos="{_fmt(model.body_pos[b])}" quat="{_fmt(model.body_quat[b])}">']
        lines.append(f'{pad}  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>')
        lines += [f"{pad}  {s}" for s in joints.get(b, []) + geoms.get(b, [])]
        lines += [body_xml(c, depth + 1) for c in children.get(b, [])]
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    assets = "\n    ".join(f'<mesh name="m{i}" file="m{i}.stl"/>' for i in mesh_ids)
    top = "\n".join(
        [f"    {s}" for s in geoms.get(0, [])] + [body_xml(b, 2) for b in children.get(0, [])]
    )
    xml = f'<mujoco>\n  <asset>\n    {assets}\n  </asset>\n  <worldbody>\n{top}\n  </worldbody>\n</mujoco>\n'
    (args.out / "scene.xml").write_text(xml)

    check = mujoco.MjModel.from_xml_path(str(args.out / "scene.xml"))
    assert check.nq == model.nq, (check.nq, model.nq)
    size = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"{len(mesh_ids)} meshes, {faces} faces, nq={check.nq}, {size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
