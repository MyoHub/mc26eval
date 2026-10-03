# Train → Export ONNX → Submit (Elbow Flexion)

End-to-end guide for **MyoChallenge 2026 Phase 1 — Elbow Flexion**.

> First time submitting? Do the 15-minute baseline first:
> [NEWBIE_QUICKSTART.md](NEWBIE_QUICKSTART.md).  
> Why ONNX instead of shipping torch in a Docker agent?
> [SUBMISSION_DEPENDENCIES.md](SUBMISSION_DEPENDENCIES.md) — the elbow eval image
> does **not** include PyTorch; export ONNX after training (or install torch in your own image).

You will:

1. Train a policy (Stable-Baselines3 **or** mjlab)
2. Export a `.onnx` file that matches the eval contract
3. Submit with `evalmc submit-onnx`

| Item | Value |
|---|---|
| Task (CLI) | `elbowflexion` |
| Phase | `1` |
| MyoSuite env | `myoElbowPose1D6MRandom-v0` |
| Observation | `(9,)` float |
| Action | `(6,)` muscle activations in `[0, 1]` |

---

## Prerequisites

### MyoChallenge CLI

```bash
pip install evalmc
evalmc host --set-host https://api.mc.myosuite.org
evalmc login
evalmc status
```

### MyoSuite (`ms3` branch)

Required for local roll-outs and SB3 training:

```bash
git clone -b ms3 https://github.com/MyoHub/myosuite.git
cd myosuite
pip install -e .
```

Smoke-test the environment:

```bash
python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0
```

---

## ONNX contract (eval server)

The remote harness loads your file with **ONNX Runtime** and expects:

| Tensor | Name (recommended) | Shape | Dtype |
|---|---|---|---|
| Input | `obs` | `[batch, 9]` | `float32` |
| Output | `action` | `[batch, 6]` | `float32` |

After inference, activations are **clipped to `[0, 1]`**. Train and export so actions already lie in that range (MyoSuite's elbow env uses a `Box(0, 1)` action space).

Verify locally before submitting:

```bash
pip install -U onnxruntime numpy
python ../scripts/smoke_test_onnx.py elbow_model.onnx --task elbowflexion
# or:
python - <<'EOF'
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("elbow_model.onnx")
obs = np.random.randn(1, 9).astype(np.float32)
action = session.run(None, {"obs": obs})[0]
print("action shape:", action.shape)  # expect (1, 6)
EOF
```

---

## Path A — Stable-Baselines3 (simplest)

Best when you want a single-process CPU/GPU trainer without mjlab's vectorized stack.

### 1. Install training deps

```bash
pip install "stable-baselines3[extra]" gymnasium torch onnx
```

### 2. Train

From the `mc26eval/examples` directory:

```bash
python train_sb3_elbow.py --timesteps 200000 --save-dir runs/elbow_sb3
```

This writes `runs/elbow_sb3/elbow_ppo.zip`.

### 3. Export ONNX

SB3 does not ship MyoSuite-specific exporters; wrap the learned actor and use PyTorch's ONNX export. Name I/O tensors `obs` / `action` to match the eval contract:

```bash
python export_sb3_onnx.py \
  --checkpoint runs/elbow_sb3/elbow_ppo.zip \
  --output elbow_model.onnx
```

### 4. Optional — roll out in MyoSuite

```bash
python - <<'EOF'
import gymnasium as gym
import myosuite
import numpy as np
import onnxruntime as ort

myosuite.register_all_envs()
env = gym.make("myoElbowPose1D6MRandom-v0")
session = ort.InferenceSession("elbow_model.onnx")

obs, _ = env.reset()
for _ in range(500):
    action = session.run(None, {"obs": obs.astype(np.float32)[None, :]})[0].squeeze()
    action = np.clip(action, 0.0, 1.0)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
env.close()
print("Roll-out OK")
EOF
```

### 5. Submit

```bash
evalmc submit-onnx elbow_model.onnx -t elbowflexion -p 1
```

---

## Path B — mjlab + mjlab_myosuite (GPU / vectorized)

[mjlab_myosuite](https://github.com/MyoHub/mjlab_myosuite) wraps MyoSuite for [mjlab](https://github.com/mujocolab/mjlab)'s training stack (RSL-RL PPO, thousands of parallel envs on GPU). **mjlab does not use Stable-Baselines3** — it uses RSL-RL. Use this path when you need throughput; use Path A for a lighter SB3 setup.

### 1. Install

```bash
git clone https://github.com/MyoHub/mjlab_myosuite.git
cd mjlab_myosuite
uv venv && source .venv/bin/activate   # or: python -m venv .venv
uv pip install -e .
uv pip install "myosuite @ git+https://github.com/MyoHub/myosuite.git@ms3"
# mjlab itself — follow mjlab_myosuite README for CUDA group if using GPU
```

### 2. Train

```bash
uv run train Mjlab-MyoSuite-myoElbowPose1D6MRandom-v0 \
  --agent.max-iterations 500 \
  --agent.num-steps-per-env 512
```

Checkpoints land under `logs/rsl_rl/.../model_*.pt`.

### 3. Export ONNX with MyoSuite helpers

`mjlab_myosuite` provides export utilities that wrap mjlab's ONNX exporter and attach task metadata:

```python
from mjlab_myosuite.rl.exporter import (
    attach_myosuite_onnx_metadata,
    export_myosuite_policy_as_onnx,
)

# actor_critic = loaded RSL-RL policy module
# normalizer   = policy.actor_obs_normalizer if obs normalization was enabled, else None
# env          = training env (unwrapped MyoSuite wrapper)

export_myosuite_policy_as_onnx(
    actor_critic=policy,
    normalizer=normalizer,
    path="exports/",
    filename="elbow_model.onnx",
)

attach_myosuite_onnx_metadata(
    env=env.unwrapped,
    run_path="local_export",
    path="exports/",
    filename="elbow_model.onnx",
)
```

**Automatic export during training:** if you enable **Weights & Biases** logging, `MyoSuiteOnPolicyRunner.save()` exports ONNX alongside each checkpoint:

```python
from mjlab_myosuite.rl.runner import MyoSuiteOnPolicyRunner

runner = MyoSuiteOnPolicyRunner(env, agent_cfg, log_dir, device)
runner.learn(num_learning_iterations=1000)
# With logger_type="wandb", ONNX appears next to model_*.pt
```

### 4. Rename / verify I/O names

mjlab's default ONNX I/O names may differ from `obs` / `action`. The eval harness uses the **first** input and output regardless of name, but for local clarity run:

```bash
python -c "
import onnx
m = onnx.load('exports/elbow_model.onnx')
print('inputs :', [(i.name, i.type.tensor_type.shape) for i in m.graph.input])
print('outputs:', [(o.name, o.type.tensor_type.shape) for o in m.graph.output])
"
```

Confirm shapes are `[batch, 9]` → `[batch, 6]`.

### 5. Submit

```bash
evalmc submit-onnx exports/elbow_model.onnx -t elbowflexion -p 1
```

---

## Submission checklist

- [ ] Registered at [mc26.myosuite.org](https://mc26.myosuite.org) and ran `evalmc login`
- [ ] ONNX input rank-2: `[1, 9]` for a single step
- [ ] ONNX output rank-2: `[1, 6]`
- [ ] Actions in `[0, 1]` (or accept server-side clipping)
- [ ] Local smoke: `python scripts/smoke_test_onnx.py your_model.onnx --task elbowflexion`
- [ ] `evalmc submit-onnx ... -t elbowflexion -p 1` returns a submission ID
- [ ] Score appears on the leaderboard after async evaluation (usually a few minutes)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No credentials found` | `evalmc login` |
| Wrong output shape | Re-export with correct `obs_dim=9`, `action_dim=6` |
| Good local score, poor leaderboard | Eval uses **held-out seeds**; overfitting to training seeds is common with short runs |
| mjlab ONNX export ImportError | Install mjlab with ONNX extras; see `mjlab_myosuite` tests: `pytest tests/test_onnx_export.py -v` |
| SB3 export fails on MPS/CUDA | Export on CPU: `model = PPO.load(path, device="cpu")` |

---

## Related files in this repo

| File | Purpose |
|---|---|
| [`examples/train_sb3_elbow.py`](../examples/train_sb3_elbow.py) | Minimal SB3 PPO training script |
| [`examples/export_sb3_onnx.py`](../examples/export_sb3_onnx.py) | SB3 checkpoint → ONNX (`obs` / `action`) |
| [`examples/export_elbow_onnx.py`](../examples/export_elbow_onnx.py) | Constant 0.5 baseline ONNX (smoke test only) |
| [`examples/task_config.json`](../examples/task_config.json) | Local dims for `policy.py` / Docker agents |

For Docker submission (Python `policy.py` instead of ONNX), see [DOCKER_SUBMISSION.md](../DOCKER_SUBMISSION.md).
