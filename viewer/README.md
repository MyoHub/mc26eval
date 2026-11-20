# ChaseTag bout viewer

A static page that replays recorded bouts in the browser: MuJoCo WASM computes the poses and three.js draws them. No server-side rendering or GPU.

```bash
python scripts/record_bout.py model.onnx -o viewer/bout.bin.gz   # records one bout per role
python -m http.server -d viewer 8000                              # open http://localhost:8000
```

- `bout.bin.gz`: joint positions of both agents at about 33 fps (about 150 kB per bout), loaded with `?bout=<url>`.
- `scene/`: kinematics-only body model with decimated meshes (2.5 MB), rebuilt with `scripts/export_viewer_scene.py` (needs `fast_simplification` and `trimesh`).
