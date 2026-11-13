# ChaseTag policy dependencies

Use a standalone ONNX model when possible. ONNX submissions contain only the model file, so you do not choose Python imports or build an agent image.

For Docker submissions, your policy can import packages already present in the ChaseTag evaluation image, or install additional packages in your own Dockerfile.

## Policy image contract

Put `policy.py` and/or `model.onnx` under `/policy` or `/app`. Include every helper module and checkpoint your policy loads. The evaluator does not install a bare `requirements.txt` and policy containers have no network access during evaluation.

| Include in image | Example |
|---|---|
| Policy entrypoint | `policy.py` or `model.onnx` |
| Your Python modules | `policy_utils.py`, `models/actor.py` |
| Model weights | `weights.pt`, `model.npz` |

Do not rely on your laptop’s virtual environment, host `site-packages`, or runtime downloads.

## Environment and contract

ChaseTag runs `myoChallengeChaseTagFBVs-v0` on MyoSuite 3.1+ / myo_sim `myofullbody`. Episodes are 10 seconds. The observation and action shapes are `(546,)` and `(354,)` per agent.

ONNX input/output:

| Tensor | Shape | Type |
|---|---|---|
| `obs` | `[batch, 546]` | `float32` |
| `action` | `[batch, 354]` | `float32` |

Your agent controls `agent_0`. The last two observation values encode its role: `[1, 0]` is chase and `[0, 1]` is evade. Both roles occur in evaluation.

## Docker `policy.py` interface

```python
class Policy:
    def __init__(self):
        # Load bundled files here.
        ...

    def reset(self):
        # Called at the start of each episode.
        ...

    def get_action(self, obs):
        # obs is a 1-D NumPy array; return 354 float actions.
        ...
```

If both files are present, the harness uses `policy.py`; otherwise it runs `model.onnx` with ONNX Runtime.

## Next

- [Quickstart](NEWBIE_QUICKSTART.md)
- [Training and export](TRAIN_EXPORT_SUBMIT.md)
- [Docker submission](../DOCKER_SUBMISSION.md)
