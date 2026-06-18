# Examples

Runnable templates for MyoChallenge 2026 submissions. Start with the workflow that matches your path, then replace the baseline with your trained policy.

## Files

| File | Purpose |
|---|---|
| [`policy.py`](policy.py) | Documented `Policy` class template — copy and customize |
| [`random_policy.py`](random_policy.py) | Random muscle activations — fastest zip smoke test |
| [`elbow_policy.py`](elbow_policy.py) | Constant 0.5 activations — deterministic baseline |
| [`policy_utils.py`](policy_utils.py) | Resolves action dimension from `task_config.json` or eval harness metadata |
| [`task_config.json`](task_config.json) | Local dev config for elbow flexion (`obs_dim=9`, `action_dim=6`) |
| [`export_elbow_onnx.py`](export_elbow_onnx.py) | Writes a valid test `.onnx` for the elbow flexion contract |
| [`train_sb3_elbow.py`](train_sb3_elbow.py) | Train elbow flexion with Stable-Baselines3 PPO |
| [`export_mjlab_onnx.py`](export_mjlab_onnx.py) | Export mjlab/RSL-RL actor → ONNX (MyoSuite helpers when installed) |

Full walkthrough: [`../tutorials/TRAIN_EXPORT_SUBMIT.md`](../tutorials/TRAIN_EXPORT_SUBMIT.md) (SB3 + mjlab training, MyoSuite ONNX helpers, submission).

### Isolated pipeline tests

No MyoSuite or MuJoCo required. From the `mc26eval` root:

```bash
python -m venv .test-venv && source .test-venv/bin/activate
pip install -e ".[test]"
pytest tests/test_sb3_onnx_pipeline.py tests/test_mjlab_onnx_pipeline.py -v
```

- `test_sb3_onnx_pipeline.py` — dummy env → SB3 PPO → `export_sb3_onnx` → ONNX inference
- `test_mjlab_onnx_pipeline.py` — mjlab-style actor → ONNX export (helper test skips if `mjlab_myosuite` not installed)

### End-to-end: two users train and submit

[`tests/test_e2e_two_user_submit.py`](../tests/test_e2e_two_user_submit.py) registers **separate users** for each pipeline and POSTs ONNX to the API:

```bash
# Embedded API (no server)
./scripts/run_e2e_submit.sh offline

# Real HTTP against local mcEvalProd
./scripts/run_e2e_submit.sh local

# Production (requires working /register on the server)
LIVE_API_BASE=https://api.mc.myosuite.org ./scripts/run_e2e_submit.sh live
```

## Workflows

### Zip — random baseline (any task)

```bash
cp random_policy.py policy.py
zip my_baseline.zip policy.py policy_utils.py task_config.json
evalmc submit my_baseline.zip -t elbowflexion -p 1
```

Edit `task_config.json` when future tasks open (set `task`, `env_id`, and dimensions).

### Zip — from the template

```bash
# policy.py is already the template; add your weights beside it
zip my_submission.zip policy.py my_weights.pt policy_utils.py task_config.json
evalmc submit my_submission.zip -t elbowflexion -p 1
```

### ONNX — elbow flexion

```bash
pip install onnx numpy
python export_elbow_onnx.py -o elbow_model.onnx
evalmc submit-onnx elbow_model.onnx -t elbowflexion -p 1
```

**ONNX I/O contract:** `obs` → `[batch, 9]`, `action` → `[batch, 6]`.

### Local roll-out (before submitting)

```bash
cp random_policy.py policy.py   # or your own policy.py

python - <<'EOF'
import gymnasium as gym
import myosuite
from policy import Policy

myosuite.register_all_envs()
env = gym.make("myoElbowPose1D6MRandom-v0")
policy = Policy()
obs, _ = env.reset()
policy.reset()
terminated = truncated = False
while not (terminated or truncated):
    obs, _, terminated, truncated, _ = env.step(policy.get_action(obs))
env.close()
print("OK")
EOF
```

Requires MyoSuite [`ms3`](https://github.com/MyoHub/myosuite/tree/ms3). See the [main README](../README.md#local-development) for install steps.

## What goes in the zip?

```
my_submission.zip
├── policy.py          ← required, at root
├── policy_utils.py    ← recommended (action sizing)
├── task_config.json   ← required for local dev; optional on server
└── weights.pt         ← your trained parameters (optional)
```

The evaluation harness mounts your zip at `/policy` and runs `from policy import Policy`.
