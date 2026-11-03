# Submission dependencies — what a Docker agent image can (and cannot) include

A Docker agent submission is **not** “upload a venv.” The evaluation worker runs your
agent image (or mounts `/policy` from it) against a **task-specific eval image**.
Imports resolve from packages installed in those images — not from a hoped-for
`pip install` at score time.

For **ONNX** submissions (`evalmc submit-onnx`), you do not ship Python at all —
the harness loads the graph with ONNX Runtime.

```text
┌─────────────────────────────────────────────┐
│  Task eval image (preinstalled libraries)   │
│  ┌───────────────────────────────────────┐  │
│  │  /policy  ← agent mount (Docker)      │  │
│  │    policy.py / model.onnx / weights   │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## Decision guide

| Your situation | Recommended path |
|---|---|
| Neural net is obs → action only | **ONNX** — `evalmc submit-onnx` |
| Need Python around weights / multi-file checkpoints | **Docker** — [DOCKER_SUBMISSION.md](../DOCKER_SUBMISSION.md) |
| Need CUDA stacks, conda, or packages not in the image | **Docker** — install them in your agent Dockerfile |

Do **not** expect a `requirements.txt` next to `policy.py` to be installed by the
platform. Evaluation has **no network** and does not run `pip install` for you —
bake deps into the image (or use packages already listed below).

---

## What belongs in the agent image (`/policy` or `/app`)

| Include | Examples |
|---|---|
| Entry point | `policy.py` and/or `model.onnx` at `/policy` (or `/app`) |
| Your own Python modules | `policy_utils.py`, `models/actor.py`, … |
| Weights / checkpoints | `weights.pt`, `model.npz`, SB3 `.zip` checkpoints |
| Local-test config | `task_config.json` / `chasetag_task_config.json` |

| Do **not** rely on | Why |
|---|---|
| Host `site-packages/` / laptop venv | Not in the image |
| Bare `requirements.txt` hoping for install | Offline eval — never installed by the platform |
| Large unrelated datasets | Waste of image size; unused at score time |

Vendoring a **small pure-Python** helper you wrote (or a single-file utility) is fine
because `/policy` is on `sys.path`. Shipping full third-party distributions without
installing them properly in the Dockerfile is not.

---

## Preinstalled packages (eval images)

Exact pins can change between image rebuilds; treat this as the **allowed import set**
when you depend on the task eval image rather than installing everything yourself.
If you need something else → export to ONNX or install it in your Docker agent image.

### Elbow Flexion (`elbowflexion`)

| Package | Typical use |
|---|---|
| `numpy` | Arrays, actions |
| `onnxruntime` | ONNX inference (also used by the harness for bare `.onnx`) |
| `gymnasium` | Env API (harness-side; optional in your policy) |
| `mujoco` | Physics (harness-side) |
| MyoSuite (`ms3`) | Environment (harness-side) |

Safe imports for elbow `policy.py` agents that rely on the eval image: `numpy`,
stdlib, your own modules, and loading weights with **numpy** (`.npz`) without
PyTorch in the elbow image.

> The elbow eval image does **not** preinstall PyTorch or Stable-Baselines3.
> Train with them locally, then **export ONNX** (see [TRAIN_EXPORT_SUBMIT.md](TRAIN_EXPORT_SUBMIT.md))
> or install those packages in your own Docker agent image.

### ChaseTag (`chasetag`)

| Package | Typical use |
|---|---|
| `numpy` | Arrays |
| `torch` | Load `.pt` checkpoints, inference |
| `stable_baselines3` | Load SB3 zip checkpoints |
| `onnxruntime` | Bare ONNX submissions |
| `musclemimic_models==1.0.5` | Scored ChaseTag FBVs body (do not use 1.0.6 unless forced; see MyoSuite #579) |
| `huggingface_hub` | Fetching published clips/models **at image build time only** — your policy must not download at eval time |
| `gymnasium`, `mujoco`, MyoSuite | Harness-side |

Safe imports for ChaseTag agents using the eval image: the table above + your own
modules + checkpoint files next to `policy.py`.

---

## Policy contract (Docker `policy.py`)

```python
# policy.py at /policy (or /app)
class Policy:
    def __init__(self):
        # Load files from Path(__file__).parent — same directory as policy.py
        ...

    def reset(self):
        # Called at the start of every episode
        ...

    def get_action(self, obs):  # obs: 1-D np.ndarray
        # Return 1-D float action of the task length
        ...
```

Harness behaviour:

1. Mount agent `/policy` (or `/app`)
2. Prefer `policy.py` → `from policy import Policy`
3. Else load `model.onnx` with ONNX Runtime

---

## ONNX contract (no Python policy)

| Task | Input | Output | Notes |
|---|---|---|---|
| Elbow flexion | `obs` `[B, 9]` | `action` `[B, 6]` | Clipped to `[0, 1]` after inference |
| ChaseTag | `obs` `[B, 546]` | `action` `[B, 354]` | You control `agent_0` |

---

## Rules that affect dependencies

- **No network** during evaluation — download nothing in `__init__` / `get_action`.
- Bundle every weight file you need inside the Docker image (or use ONNX alone).
- Observation/action transforms must live inside `get_action` or the ONNX graph.

---

## Next steps

- First submission in ~10 minutes: [NEWBIE_QUICKSTART.md](NEWBIE_QUICKSTART.md)
- Train → ONNX → submit: [TRAIN_EXPORT_SUBMIT.md](TRAIN_EXPORT_SUBMIT.md)
- Custom environment: [DOCKER_SUBMISSION.md](../DOCKER_SUBMISSION.md)
