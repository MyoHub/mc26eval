# Train, export, and submit a ChaseTag policy

A ChaseTag policy controls one full-body agent in both chase and evade episodes. Use the observation’s final two values to identify the role.

## Contract

| Item | Value |
|---|---|
| Environment | `myoChallengeChaseTagFBVs-v0` |
| MyoSuite | 3.1+; myo_sim `myofullbody` |
| Episode limit | 10 seconds |
| Observation | `float32`, `[batch, 546]` for ONNX |
| Action | `float32`, `[batch, 354]` for ONNX |
| Role flag | `[1, 0]` chase; `[0, 1]` evade |

## Train

Use the [MyoSuite 3.1 ChaseTag Colab](https://colab.research.google.com/github/MyoHub/myosuite/blob/dev/tutorials/5.6_ChaseTag_MyoChallenge.ipynb) for checkpoint selection, ONNX export, and local play. Its default export is the tutorial 5.4 directional walker adapted to ChaseTag. The stronger `chasetag_supermimic_selfplay.onnx` is an optional self-play refinement of a ChaseTag DAgger student; select `USE_STUDENT` in the notebook to use it. These are distinct policies. The older `policy_bc_best.pt` / `policy_heading_table.pt` files are not valid inputs to the current export cell. For local environment checks, install:

```bash
python -m pip install -U 'myosuite>=3.1.0'
python -c "from myosuite import make_env; env=make_env('myoChallengeChaseTagFBVs-v0'); print(env.observation_space, env.action_space); env.close()"
```

## Export

The submitted model must accept `[batch, 546]` observations and return `[batch, 354]` actions. A tested starter model is included at `examples/artifacts/chasetag_walker.onnx`. For a model exported from Colab, use the exact file path printed by the export cell and download it to your computer.

Check the exported model against a real ChaseTag reset observation (this checks
ONNX input/output shapes and that runtime inference returns finite actions):

```bash
python -m pip install -U 'myosuite>=3.1.0' onnxruntime numpy
python scripts/smoke_test_onnx.py chasetag.onnx
```

Then test behavior in the MyoSuite environment and verify both chase and evade
roles before submitting. The repository's `scripts/smoke_test_policy.py` runs a
local `policy.py` wrapper (for example, one that calls your ONNX model) against
the ChaseTag environment for debugging only; it is not a substitute for
evaluating both roles or measuring competitive performance, and only the ONNX
file is submitted.

## Submit

```bash
evalmc submit-onnx examples/artifacts/chasetag_walker.onnx -t chasetag -p 1
```

Run from the repository root; `evalmc` rejects a missing path. Replace the model path if you want to submit your own export.

ONNX is the only accepted submission format. Any preprocessing your policy needs must be exported as part of the ONNX graph.
