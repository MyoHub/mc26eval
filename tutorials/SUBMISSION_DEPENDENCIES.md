# ChaseTag ONNX submission contract

**ONNX is the only accepted submission format.** A submission is a single `.onnx` file: no Python code, extra files, or packages are installed or run alongside it.

```bash
evalmc submit-onnx model.onnx -t chasetag -p 1
```

## Environment and contract

ChaseTag runs `myoChallengeChaseTagFBVs-v0` on MyoSuite 3.1+ / myo_sim `myofullbody`. Episodes are 10 seconds. The observation and action shapes are `(546,)` and `(354,)` per agent.

**Start positions are randomized.** In evaluation, both players' starting positions and headings vary slightly from the default MyoSuite reset in every episode. The change is mirrored, so both players get the same relative start, and each chase/evade pair of episodes shares it. The exact ranges are not published: train policies that are robust to small changes in the starting pose, not only to the default reset.

| Tensor | Shape | Type |
|---|---|---|
| `obs` | `[batch, 546]` | `float32` |
| `action` | `[batch, 354]` | `float32` |

Your agent controls `agent_0`. The last two observation values encode its role: `[1, 0]` is chase and `[0, 1]` is evade. Both roles occur in evaluation.

## Recurrent policies (GRU, LSTM)

Stateful policies are supported. Besides `obs` and `action`, the model may have
extra **state inputs** named `<name>_in` with a matching **state output**
`<name>_out` of the same shape:

| Tensor | Shape | Notes |
|---|---|---|
| input 0 `obs` | `[1, 546]` | unchanged |
| state inputs, e.g. `h_in`, `c_in` | fixed, e.g. `[num_layers, 1, hidden]` | `float32` |
| output 0 `action` | `[1, 354]` | unchanged |
| state outputs, e.g. `h_out`, `c_out` | same as the matching `_in` | |

- **Reset:** every state input is zeros at the start of each episode, including each
  chase/evade switch. There is no "is first step" input: zero state means a fresh episode.
- **Each step:** the evaluator feeds each `<name>_out` back as the next step's `<name>_in`.
- **Per agent:** each player has its own model instance and state; nothing is shared.
- **Batch is 1.** State shapes must be static apart from the batch dimension.
- GRU uses one pair (`h_in`/`h_out`); LSTM uses two (`h_in`/`h_out`, `c_in`/`c_out`).
  Any extra input not named `<name>_in`, or without a matching `_out`, is rejected.

Export example (PyTorch):

```python
import torch, torch.nn as nn

class Actor(nn.Module):
    def __init__(self, hidden=128, layers=1):
        super().__init__()
        self.rnn = nn.LSTM(546, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 354)

    def forward(self, obs, h_in, c_in):
        y, (h_out, c_out) = self.rnn(obs.unsqueeze(1), (h_in, c_in))
        return torch.sigmoid(self.head(y[:, 0])), h_out, c_out

actor = Actor().eval()
args = (torch.zeros(1, 546), torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
torch.onnx.export(actor, args, "model.onnx", opset_version=17,
                  input_names=["obs", "h_in", "c_in"],
                  output_names=["action", "h_out", "c_out"],
                  external_data=False)  # one self-contained .onnx file
```

Train with the same convention (state zeroed at episode start) so evaluation
matches training.

## Packaging checklist

- Export everything the policy needs into the graph: observation normalization, role handling, and action scaling.
- Keep weights embedded in the `.onnx` file (avoid external-data files).
- The evaluator runs the model with ONNX Runtime on CPU and has no network access.
- Check the exported file with [`scripts/smoke_test_onnx.py`](../scripts/smoke_test_onnx.py) before submitting.

## Next

- [Quickstart](NEWBIE_QUICKSTART.md)
- [Training and export](TRAIN_EXPORT_SUBMIT.md)
