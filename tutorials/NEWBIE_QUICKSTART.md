# ChaseTag quickstart

Go from team registration to your first submitted policy.

## 1. Register and configure the CLI

Register at [mc26.myosuite.org](https://mc26.myosuite.org). Save the CLI token shown for your team. ChaseTag is invitation-only, so ask an organizer to approve your team before submitting.

```bash
pip install evalmc
evalmc host --set-host https://api.mc.myosuite.org
evalmc login
evalmc status
```

`evalmc status` should show the expected API host and team.

## 2. Export one policy

**ONNX is the simplest route.** Your model receives a batch of 546-value observations and returns 354-value actions:

```text
obs [batch, 546] → model → action [batch, 354]
```

The last two observation values encode role: `[1, 0]` means chase; `[0, 1]` means evade. Your policy must handle both roles. Each bout balances the roles.

For local environment checks or training, install MyoSuite 3.1+:

```bash
python -m pip install -U 'myosuite>=3.1.0'
python -c "from myosuite import make_env; env=make_env('myoChallengeChaseTagFBVs-v0'); print(env.observation_space, env.action_space); env.close()"
```

For a first upload, use this repository's tested starter model: `examples/artifacts/chasetag_walker.onnx`. To train or choose another policy, use the [MyoSuite 3.1 ChaseTag Colab](https://colab.research.google.com/github/MyoHub/myosuite/blob/dev/tutorials/5.6_ChaseTag_MyoChallenge.ipynb), then download its exported ONNX file. The Colab's default `chasetag_player.onnx` and optional self-play model are distinct policies.

## 3. Submit ONNX

```bash
evalmc submit-onnx examples/artifacts/chasetag_walker.onnx -t chasetag -p 1
```

This command assumes you run it from the repository root. Check the model exists first:

```bash
ls -lh examples/artifacts/chasetag_walker.onnx
evalmc submit-onnx examples/artifacts/chasetag_walker.onnx -t chasetag -p 1
```

The CLI returns a submission ID. Evaluation is asynchronous; follow the status in the website’s **Submissions** tab.

## Or submit a Docker agent

Use Docker when your policy needs Python preprocessing, multiple weight files, or packages beyond the evaluation image. Include `policy.py` and/or `model.onnx` under `/policy` or `/app`, then:

```bash
evalmc push myteam-chasetag:latest -t chasetag -p 1
```

For production submissions, the CLI publishes the image through ECR so the remote evaluator can pull it. A local-only Docker tag is not available to production workers. See [Docker submission](../DOCKER_SUBMISSION.md).

## Before submitting

- Confirm the ONNX input/output contract: `[batch, 546]` → `[batch, 354]`.
- Make sure the policy reads the role flag and supports chase and evade.
- Bundle every weight and helper file needed by a Docker policy.
- Do not download files or call external services during evaluation.
- Confirm your team has invitation access and the CLI points to `https://api.mc.myosuite.org`.

Scores appear after the worker finishes evaluation. If a submission fails, open its row in the Submissions tab for the error.
