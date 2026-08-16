# Examples

Runnable templates for MyoChallenge 2026 submissions. Start with the workflow that matches your path, then replace the baseline with your trained policy.

**New here?** → [`../tutorials/NEWBIE_QUICKSTART.md`](../tutorials/NEWBIE_QUICKSTART.md)  
**What can I import?** → [`../tutorials/SUBMISSION_DEPENDENCIES.md`](../tutorials/SUBMISSION_DEPENDENCIES.md)

## Files

| File | Purpose |
|---|---|
| [`policy.py`](policy.py) | Documented `Policy` class template — copy into a Docker agent image |
| [`random_policy.py`](random_policy.py) | Random muscle activations — fastest local / Docker smoke test |
| [`elbow_policy.py`](elbow_policy.py) | Constant 0.5 activations — deterministic baseline |
| [`chasetag_policy.py`](chasetag_policy.py) | SB3 chaser/runner wrapper for full-body ChaseTag |
| [`chasetag_task_config.json`](chasetag_task_config.json) | Local config for ChaseTag (`obs_dim=546`, `action_dim=354`) |
| [`policy_utils.py`](policy_utils.py) | Resolves action dimension from `task_config.json` or eval harness metadata |
| [`task_config.json`](task_config.json) | Local dev config for elbow flexion (`obs_dim=9`, `action_dim=6`) |
| [`export_elbow_onnx.py`](export_elbow_onnx.py) | Writes a valid test `.onnx` for the elbow flexion contract |
| [`train_sb3_elbow.py`](train_sb3_elbow.py) | Train elbow flexion with Stable-Baselines3 PPO |
| [`export_mjlab_onnx.py`](export_mjlab_onnx.py) | Export mjlab/RSL-RL actor → ONNX (MyoSuite helpers when installed) |

Full walkthroughs:

- [`../tutorials/NEWBIE_QUICKSTART.md`](../tutorials/NEWBIE_QUICKSTART.md) — first submit
- [`../tutorials/SUBMISSION_DEPENDENCIES.md`](../tutorials/SUBMISSION_DEPENDENCIES.md) — allowed imports
- [`../tutorials/TRAIN_EXPORT_SUBMIT.md`](../tutorials/TRAIN_EXPORT_SUBMIT.md) — SB3 + mjlab → ONNX
- [`../DOCKER_SUBMISSION.md`](../DOCKER_SUBMISSION.md) — Docker agent images with `policy.py`

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

### ONNX — elbow flexion *(primary)*

```bash
pip install onnx numpy onnxruntime
python export_elbow_onnx.py -o elbow_model.onnx
python ../scripts/smoke_test_onnx.py elbow_model.onnx --task elbowflexion
evalmc submit-onnx elbow_model.onnx -t elbowflexion -p 1
```

**ONNX I/O contract:** `obs` → `[batch, 9]`, `action` → `[batch, 6]`.

### Docker — random `policy.py` baseline

```bash
cp random_policy.py policy.py
python ../scripts/smoke_test_policy.py --task elbowflexion   # needs MyoSuite ms3
docker build -t myteam-elbowflexion-p1:latest -f- . <<'EOF'
FROM python:3.10-slim
WORKDIR /policy
COPY policy.py policy_utils.py task_config.json .
EOF
evalmc push myteam-elbowflexion-p1:latest -t elbowflexion -p 1
```

Edit `task_config.json` when future tasks open (set `task`, `env_id`, and dimensions).
See [DOCKER_SUBMISSION.md](../DOCKER_SUBMISSION.md) for production ECR details.

### Local roll-out (before Docker push)

```bash
cp random_policy.py policy.py   # or your own policy.py
python ../scripts/smoke_test_policy.py --task elbowflexion
```

Requires MyoSuite [`ms3`](https://github.com/MyoHub/myosuite/tree/ms3). See the [main README](../README.md#local-development) for install steps.

## What goes in a Docker agent image?

```
/policy/                 ← mounted / copied into the agent image
├── policy.py            ← required (or model.onnx)
├── policy_utils.py      ← recommended (action sizing)
├── task_config.json     ← useful for local smoke tests
└── weights.pt           ← your trained parameters (optional)
```

**Do not** expect a bare `requirements.txt` to be installed by the platform —
install packages in your Dockerfile, or stick to packages already in the
[task eval image](../tutorials/SUBMISSION_DEPENDENCIES.md).

The evaluation harness mounts `/policy` (or `/app`) and runs `from policy import Policy`
(or loads `model.onnx` via ONNX Runtime).
