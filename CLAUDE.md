# Repository notes

This repository contains participant-facing instructions and examples for the
2026 MyoChallenge ChaseTag task. Keep task guidance aligned with the current
MyoSuite ChaseTag environment and the public challenge website.

## Submission paths

- ONNX only: `evalmc submit-onnx model.onnx -t chasetag -p 1`

ONNX is the only accepted submission format; do not document Docker or other
paths here. Docker support is disabled for 2026; the removed guide and the plan
to restore it live in the backend repo at `mcEvalProd/docs/DOCKER_SUBMISSIONS_FUTURE.md`. The ChaseTag observation has 546 values (`[batch, 546]` float32 input)
and the action has 354 values (`[batch, 354]` float32 output). Check the README and tutorials for the
participant workflow; do not copy backend administration or deployment
instructions into this repository.
