# MyoChallenge 2026 — Submission Guide

Welcome to **MyoChallenge 2026**, a competition for musculoskeletal motor control using [MyoSuite](https://github.com/MyoHub/myosuite).

Train a policy that controls a physiologically-realistic arm model to perform dexterous tasks. Submit your trained policy and see how it ranks on the leaderboard.

---

## Tasks

| Task | Description |
|---|---|
| **TableTennis** | Return a ball to a target zone using a racket |
| **Soccer** | Kick a ball to a goal with precise control |

Each task has multiple competition **phases** with increasing difficulty.

---

## Submission (recommended: zip upload)

No Docker required. Package your trained policy as a zip file and upload it in seconds.

### 1. Install the CLI

```bash
pip install evalmc
```

### 2. Configure the API host

```bash
evalmc host --set-host https://api.mc.myosuite.org
```

### 3. Log in with your team credentials

```bash
evalmc login
# prompts for email + auth token
```

Your auth token was emailed to you when you registered. You can also pass credentials inline with `--email` / `--auth-token` on every command.

### 4. Write your policy

Your zip **must** contain a file called `policy.py` at its root, defining a class `Policy` with this interface:

```python
class Policy:
    def __init__(self):
        # load weights here (files bundled in the zip are available
        # at the same path as policy.py)
        pass

    def reset(self):
        # called at the start of every episode
        pass

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # obs shape and action shape are task-specific (see below)
        ...
```

See [`examples/policy.py`](examples/policy.py) for a documented template, and [`examples/random_policy.py`](examples/random_policy.py) for a minimal baseline.

**Observation / action shapes by task:**

| Task | obs shape | action shape |
|---|---|---|
| TableTennis phase 1 | (78,) | (39,) |
| Soccer phase 1 | (78,) | (39,) |

### 5. Bundle and submit

```bash
# Create the zip (policy.py must be at the root, not inside a subfolder)
zip my_submission.zip policy.py weights.pt

# Submit
evalmc submit my_submission.zip -p 1 -t tabletennis
```

You will receive a **Submission ID** immediately. Evaluation runs asynchronously; results appear on the leaderboard once scoring is complete.

---

## Submission (advanced: Docker image)

If your policy requires custom system dependencies or a non-standard Python environment, you can submit a Docker image instead:

```bash
evalmc push my-image:tag -p 1 -t tabletennis
```

See the [Docker submission guide](DOCKER_SUBMISSION.md) for build instructions.

---

## Checking your submission status

```bash
evalmc status
```

Or visit the leaderboard at **https://myochallenge26.netlify.app**.

---

## Rules

- One active submission per team per task/phase at a time.
- Submissions are evaluated on held-out environment seeds not available to participants.
- Policies must be self-contained — no network access during evaluation.
- Any post-processing of observations or pre-processing of actions must be inside `get_action`.

---

## Local testing

Install MyoSuite and test your policy locally before submitting:

```bash
pip install myosuite

python - <<'EOF'
import myosuite
import gymnasium as gym
import numpy as np
from policy import Policy  # your policy.py

env = gym.make("myoTTBall-v0")  # example task
policy = Policy()

obs, _ = env.reset()
policy.reset()
done = False
while not done:
    action = policy.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print("Episode complete")
EOF
```

---

## Registration

Register your team at **https://myochallenge26.netlify.app** or via the API:

```bash
curl -X POST https://api.mc.myosuite.org/register \
  -H "Content-Type: application/json" \
  -d '{"email": "team@example.com", "team_name": "MyTeam"}'
```

Your auth token will be returned in the response — save it securely.

---

## Questions

Open an issue in this repository or contact the organizers at the [MyoHub discussions](https://github.com/MyoHub/myosuite/discussions).
