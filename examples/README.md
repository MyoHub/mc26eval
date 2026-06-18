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
