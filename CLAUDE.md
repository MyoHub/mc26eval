# Repository notes

This repository contains participant-facing instructions and examples for the
2026 MyoChallenge ChaseTag task. Keep task guidance aligned with the current
MyoSuite ChaseTag environment and the public challenge website.

## Submission paths

- ONNX: `evalmc submit-onnx model.onnx -t chasetag -p 1`
- Docker/ECR: `evalmc push image:tag -t chasetag -p 1`

The ChaseTag observation has 546 values and the action has 354 values. The
Docker harness imports `Policy` from `policy.py`, calls `reset()` per episode,
and calls `get_action(obs)` each step. Check the README and tutorials for the
participant workflow; do not copy backend administration or deployment
instructions into this repository.
