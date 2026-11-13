# ChaseTag ONNX submission contract

**ONNX is the only accepted submission format.** A submission is a single `.onnx` file: no Python code, extra files, or packages are installed or run alongside it.

```bash
evalmc submit-onnx model.onnx -t chasetag -p 1
```

## Environment and contract

ChaseTag runs `myoChallengeChaseTagFBVs-v0` on MyoSuite 3.1+ / myo_sim `myofullbody`. Episodes are 10 seconds. The observation and action shapes are `(546,)` and `(354,)` per agent.

| Tensor | Shape | Type |
|---|---|---|
| `obs` | `[batch, 546]` | `float32` |
| `action` | `[batch, 354]` | `float32` |

Your agent controls `agent_0`. The last two observation values encode its role: `[1, 0]` is chase and `[0, 1]` is evade. Both roles occur in evaluation.

## Packaging checklist

- Export everything the policy needs into the graph: observation normalization, role handling, and action scaling.
- Keep weights embedded in the `.onnx` file (avoid external-data files).
- The evaluator runs the model with ONNX Runtime on CPU and has no network access.
- Check the exported file with [`scripts/smoke_test_onnx.py`](../scripts/smoke_test_onnx.py) before submitting.

## Next

- [Quickstart](NEWBIE_QUICKSTART.md)
- [Training and export](TRAIN_EXPORT_SUBMIT.md)
