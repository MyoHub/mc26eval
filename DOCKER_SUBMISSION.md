# ChaseTag Docker submission

Use Docker when you need a Python `policy.py`, multiple checkpoint files, system libraries, or packages beyond the evaluation image. For a single inference graph, submit ONNX instead.

## Required files

Include `policy.py` and/or `model.onnx` under `/policy` or `/app`. Bundle every model weight and helper file used at runtime. The policy must not download files or call external services during evaluation.

A Python policy implements:

```python
class Policy:
    def __init__(self):
        ...  # load bundled model files

    def reset(self):
        ...  # called at episode start

    def get_action(self, obs):
        ...  # obs: 546 values; return 354 actions
```

ChaseTag evaluates both chase and evade roles. Read the role flag in the final two observation values.

## Build and submit

From the directory containing your policy files and Dockerfile:

```bash
docker build -t myteam-chasetag:latest .
evalmc push myteam-chasetag:latest -t chasetag -p 1
```

For production, `evalmc push` uploads the image to ECR so the remote evaluator can pull it. A local-only image tag cannot be accessed by production workers. The CLI checks that the image contains a supported policy path before submitting.

## When evaluation cannot import a package

Check the [dependency guide](tutorials/SUBMISSION_DEPENDENCIES.md). Either install the package in your agent image or export your policy to ONNX. The evaluator does not install a standalone requirements file.
