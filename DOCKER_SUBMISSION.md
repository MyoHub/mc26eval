# Docker submission (advanced)

Use Docker when your policy needs a custom base image, system packages, a Python
`policy.py` wrapper, or libraries beyond a standalone ONNX upload.

Prefer ONNX when possible — and check the
[allowed import list](tutorials/SUBMISSION_DEPENDENCIES.md) before choosing Docker:

```bash
evalmc submit-onnx model.onnx -t elbowflexion -p 1
evalmc submit-onnx model.onnx -t chasetag -p 1
```

New to submissions? Start with [tutorials/NEWBIE_QUICKSTART.md](tutorials/NEWBIE_QUICKSTART.md).

## Image contract

Your agent image must include **one** of these paths:

| Path | Contents |
|---|---|
| `/policy/policy.py` | Python policy class |
| `/policy/model.onnx` | ONNX model |
| `/app/policy.py` | Same as above (alternate layout) |
| `/app/model.onnx` | Same as above (alternate layout) |

`evalmc push` checks for these files before recording a submission.

## Minimal elbow flexion agent

Reference Dockerfile (from the eval harness):

```dockerfile
FROM python:3.10-slim
WORKDIR /policy
COPY policy_utils.py policy.py task_config.json .
```

Build and submit:

```bash
# From a directory with policy.py, policy_utils.py, task_config.json
docker build -t myteam-elbowflexion-p1:latest .
evalmc push myteam-elbowflexion-p1:latest -t elbowflexion -p 1
```

## Remote vs local API

| API host | What happens |
|---|---|
| **Production** (`api.mc.myosuite.org`) | CLI obtains ECR credentials; you `docker push` to AWS. The eval worker pulls from ECR. |
| **Local dev** (`localhost`) | Image tag is recorded directly; the eval worker on the same machine must have that tag. |

Pushing a **local-only** Docker tag to the production API will record a submission, but evaluation will fail because remote workers cannot see images on your laptop. The CLI warns about this — use ONNX instead, or complete the ECR push flow.

## Commands

```bash
evalmc host --set-host https://api.mc.myosuite.org
evalmc login
evalmc push my-image:tag -t elbowflexion -p 1
```

Skip the in-image policy check only if you know what you are doing:

```bash
evalmc push my-image:tag -t elbowflexion -p 1 --skip-policy-check
```

## When Docker makes sense

- Proprietary CUDA stacks or compiled dependencies
- Packages **not** listed in [SUBMISSION_DEPENDENCIES.md](tutorials/SUBMISSION_DEPENDENCIES.md)
- Large conda environments
- Non-Python inference runtimes
- Custom Python around weights (`policy.py` with multi-file checkpoints)

For standard obs→action graphs, **ONNX is simpler and faster** — do not use Docker
just to ship a bare inference model.
