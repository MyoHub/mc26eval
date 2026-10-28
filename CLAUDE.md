# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MyoChallenge competition platform for musculoskeletal policy evaluation (built on [MyoSuite](https://github.com/MyoHub/myosuite)):

- **`mcEvalProd/`** — FastAPI backend server (registrations, submissions, scoring, leaderboard)
- **`mcEvalProd-CLI/`** — Python CLI (`evalmc`, published to PyPI) teams use to submit
- **`mcEval-Web/`** — Static HTML/JS frontend at [mc26.myosuite.org](https://mc26.myosuite.org)
- **`mc26eval/`** — Public participant-facing repo with submission instructions and policy examples

## Commands

### mcEvalProd (server)

```bash
cd mcEvalProd
pip install -r requirements.txt

# Start server only (EVAL_AUTO_START=false, default)
python start_server.py

# Start server + eval worker in one process (set EVAL_AUTO_START=true in config/.env)
python start_server.py

# Run eval worker as a separate process (recommended for production-like local dev)
python -m app.services.evaluate --watch
python -m app.services.evaluate --watch --task tabletennis --phase 1  # filtered
python -m app.services.evaluate --task tabletennis --phase 1           # one-shot

# Run API + worker as separate Docker containers (docker-compose)
docker compose up --build

# Run offline integration tests (SQLite in-memory, no AWS or Docker needed)
python -m pytest tests/integration/test_config_contract.py tests/integration/test_api_offline.py tests/integration/test_worker.py -v

# Run a single test
python -m pytest tests/integration/test_worker.py::test_worker_run_once_onnx_scores_submission -v
```

### mcEvalProd-CLI

```bash
cd mcEvalProd-CLI
pip install -e .

# Run CLI smoke tests
python -m pytest tests/test_cli_smoke.py -v
```

### All integration tests (from repo root)

```bash
./scripts/run_integration_tests.sh

# Also run online tests against a live API
LIVE_API_BASE=https://api.mc.myosuite.org ./scripts/run_integration_tests.sh
```

## Architecture

Competition format/access guidance is in `docs/COMPETITION_FORMATS.md`; review and
verification steps are in `docs/REVIEWER_GUIDE.md`. Edition invitations govern
registration; `TaskPhase.access_mode=invitational` governs per-team submission.
Title-cycle product rules (including the **2026 closed-door ChaseTag special**) are in
`docs/CHASETAG_TITLE_CYCLE_RULES.md`.

### 2026 ChaseTag closed-door special

Configured in `mcEvalProd/config/challenges/2026.yaml`:

- Only ChaseTag Phase 1 is visible; Elbow Flexion / Boxing / Fencing are hidden
- ChaseTag: `evaluation_mode: title_cycle`, `access_mode: invitational`
- Two scheduled windows (America/New_York), `auto_run_tournament: true`:
  - Cycle 1: Thu Oct 29 08:00 → Thu Oct 29 12:00 (freeze + finals)
  - Cycle 2: Thu Oct 29 12:00 → Thu Oct 29 20:00 (freeze + finals)
- Teams nominate a scored in-window submission for finals (else best score)
- After deploy: `python init/init_task_phases.py`, grant teams via
  `POST /admin/task-phase/chasetag/1/access`, keep the eval worker running

### Submission paths (two coexist)

```
Path A — ONNX upload (recommended for ChaseTag / elbow when open):
  evalmc submit-onnx model.onnx -p 1 -t chasetag
    → POST /submit/onnx  (multipart)
    → stored in S3/local under submissions/{task}/phase{N}/{user_id}/
    → Submission.submission_type = "onnx"
    → evaluate.py: onnx_runner.run_onnx(bytes) — writes model.onnx, mounts at /policy in ENV_IMAGE

Path B — Docker/ECR (advanced / custom Python policies):
  evalmc push image:tag -p 1 -t chasetag
    → GET  /ecr/upload-info/{team}/{task}/{phase}  → ECR credentials
    → POST /ecr/submit                             → records submission
    → Submission.submission_type = "docker"
    → evaluate.py: docker_runner.run_pair(agent_image) or docker_harness_runner
```

There is **no zip submission path** (`/submit/zip` and `evalmc submit` were removed).

Both paths converge at `evaluate_submission()` in `app/services/evaluate.py`, then share the same `record_score()` → leaderboard pipeline. Title-cycle / ladder tasks branch earlier inside `evaluate_submission()`.

### Backend (`mcEvalProd/`)

- **`app/api/admin_api.py`** — All HTTP endpoints. `POST /submit/onnx` is the ONNX path; `POST /ecr/submit` is the Docker path. Title-cycle nominate/freeze/tournament admin routes live here too.
- **`app/models/db.py`** — SQLAlchemy models. `Submission` has a `submission_type` column (`"docker"` | `"onnx"`, server default `"docker"`). Score tables created dynamically per task/phase as `score_{task}_{phase}`. `ensure_open_title_cycle()` supports rolling `duration_weeks` **or** explicit scheduled `windows`.
- **`app/services/zip_runner.py`** — Shared policy-mount helper (`run_zip_mount`): mounts a policy dir at `/policy` inside `ENV_IMAGE`, runs `/eval_harness.py`, reads `/shared/results.json`. Used by ONNX and Docker harness runners (not a zip submission path).
- **`app/services/docker_runner.py`** — Evaluates Docker submissions: runs ENV_IMAGE + agent container pair. Both Docker clients are lazy (`_client()`) so importing the module doesn't require a running daemon.
- **`app/services/title_cycle.py`** — Mid-cycle scoring vs fixed champion, freeze/seed, single-elim tournament, `advance_title_cycles()` for scheduled auto freeze+finals.
- **`config/title_cycle.py`** — Loads `title_cycle` knobs from year YAML (`duration_weeks` or `windows` + `timezone` + `auto_run_tournament`).
- **`app/services/evaluate.py`** — Evaluation worker. `run_once()` advances title cycles first, then processes pending submissions. Branches on `submission_type` / evaluation mode to dispatch to the right runner.
- **`start_server.py`** — Starts uvicorn. If `EVAL_AUTO_START=true`, also starts the worker as a background daemon thread (convenient locally; for production use a separate process/container).
- **`app/services/ecr_utils.py`** — AWS ECR operations (Docker path only).
- **`app/utils/s3_utils.py`** / **`local_storage.py`** — Storage abstraction toggled by `USE_LOCAL_STORAGE`.
- **`config/config.py`** — All config from `config/.env`. `CHALLENGE_YEAR` is the **bootstrap** year: it controls the DB table prefix (`MyoChallenge2026*`) for whatever edition this process is bound to. The DB's `MyoChallengeEditions` table (see below) is the runtime source of truth for which edition is "active"; `CHALLENGE_YEAR` only matters at process startup and when creating a brand-new edition's tables.
- **`config/challenges/{year}.yaml`** — Task/phase definitions seeded by `init/init_task_phases.py` (visibility, evaluation_mode, access_mode, title_cycle windows, env images).

### Policy contract (harness mount)

The base evaluation image runs `/eval_harness.py` which:
1. Adds `/policy` (materialized ONNX or extracted Docker policy) to `sys.path`
2. Imports `from policy import Policy` **or** loads `/policy/model.onnx`
3. Calls `policy.reset()` per episode, `policy.get_action(obs)` per step
4. Writes `{"score_name": value, ...}` to `/shared/results.json`

Docker agent images must ship `policy.py` and/or `model.onnx` under `/policy` (or `/app`). ONNX submissions are written as `/policy/model.onnx` by the worker.
### DB migration note

The `submission_type` column was added to the `Submission` table with `server_default='docker'`. For **existing PostgreSQL** databases, run:

```sql
ALTER TABLE "MyoChallenge2026Submissions"
  ADD COLUMN IF NOT EXISTS submission_type VARCHAR(20) NOT NULL DEFAULT 'docker';
```

SQLite (tests) auto-creates tables fresh, so no migration needed there.

### Multi-year editions (launching mc27, mc28, …)

Every table except `Account` (`MyoChallengeAccounts`) and the new `MyoChallengeEditions` /
`MyoChallengeInvites` tables is year-prefixed (`MyoChallenge{YEAR}Teams`, `...Participants`,
`...Submissions`, `...TaskPhases`, plus dynamic `score_{task}_{phase}` tables). Team
registration is per-edition; a participant's login identity (`Account`) is shared across
editions.

`app/models/db.py` has a `ChallengeEdition` model tracking every edition (year, title,
`is_active`, `registration_open`, `invitational_only`), seeded on first `init_db()` run from
`CHALLENGE_YEAR`. Admin endpoints manage it — **no repo clone, no code change**:

```bash
# 1. Register the new edition (row only — its tables are created lazily by init_db()
#    the next time a process starts up bound to that year)
curl -X POST $API/admin/editions -H "X-Admin-Token: $ADMIN_TOKEN" \
  -d '{"year": 2027, "title": "MyoChallenge 2027"}'

# 2. Mark it active (switches /challenge-info, registration policy, and new
#    score-table naming immediately)
curl -X POST $API/admin/editions/2027/activate -H "X-Admin-Token: $ADMIN_TOKEN"

# 3. Update config/.env: CHALLENGE_YEAR=2027, then restart the server process.
#    Team/User/Submission/TaskPhase are SQLAlchemy ORM classes bound to
#    CHALLENGE_YEAR's tables at import time, so a restart is still required
#    for the process to actually accept registrations/submissions against the
#    new year's tables — but this is a config edit + restart of the SAME
#    deployment, not a new deployment or a repo fork. mc26's tables, URL, and
#    process are untouched and stay fully browsable (leaderboard, dashboard,
#    admin views) via GET .../challenge-info?year=2026, .../leaderboard/...
#    etc. — those reads aren't restricted to the active edition.
```

Score tables avoid cross-year name collisions automatically: the original bootstrap year
keeps its legacy `score_{task}_{phase}` naming (untouched), any other active edition gets
`score_{year}_{task}_{phase}`.

**Invitational (whitelist-only) registration**: set per-edition via
`PATCH /admin/editions/{year} {"invitational_only": true}`. While enabled, `/register`
rejects any email not present in that edition's invite list with `403`. Manage the list with
`POST` / `GET` / `DELETE /admin/editions/{year}/invites`. This is a generic per-edition
toggle — not hardcoded to any specific year — enabled for 2026 via the admin API, not in code.

### CLI (`mcEvalProd-CLI/evalmc/`)

- **`main.py`** — Click entrypoint; registers `submit-onnx` and `push`
- **`submit_onnx.py`** — `evalmc submit-onnx` command: POSTs multipart to `/submit/onnx`
- **`submission.py`** — `evalmc push` command: Docker/ECR flow
- **`login.py`**, **`set_token.py`**, **`host.py`** — Auth/config; credentials in `~/.myochallenge/`

### Frontend (`mcEval-Web/`)

Pure HTML/JS; `js/config.js` + `js/env.js` set the API base URL. Served at **https://mc26.myosuite.org**.

### Public participant repo (`mc26eval/`)

- **`README.md`** — Full submission guide (ONNX and Docker paths, registration, local testing)
- **`examples/policy.py`** — Documented policy template with the full contract (for Docker agent images)
- **`examples/random_policy.py`** — Minimal random baseline

## Configuration

Copy `mcEvalProd/config/.env-template` to `mcEvalProd/config/.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `DB_URL` | Neon PostgreSQL | Database connection string |
| `USE_LOCAL_STORAGE` | `true` | `true` = local filesystem, `false` = S3 |
| `CHALLENGE_YEAR` | `2026` | Bootstrap edition year — controls DB table prefix (`MyoChallenge2026*`) this process is bound to. Change + restart to switch which edition's tables are live; see "Multi-year editions" above. |
| `ENV_IMAGE` | — | Base Docker image for policy harness eval and Docker env container |
| `EVAL_AUTO_START` | `false` | Start eval worker thread inside `start_server.py` |
| `EVAL_POLL_INTERVAL` | `10` | Seconds between DB polls in watch mode |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | Required for ECR/S3 |
| `ECR_REPOSITORY_PREFIX` | `myochallenge` | ECR repo prefix for Docker images |

See `docs/EVAL_WORKER.md` for the full deployment guide (local, Docker Compose, ECS Fargate, EC2 systemd).

For offline/test runs, set `DB_URL=sqlite:///:memory:` and `USE_LOCAL_STORAGE=true` — the integration test `conftest.py` does this automatically.


## Color Style
Use a dark scientific palette centered around deep graphite (#0D1117), slate (#161B22), and steel gray (#30363D) foundations, with off-white (#E6EDF3) and cool gray (#8B949E) typography. Primary interaction accents should use electric blue (#58A6FF) and deep technical blue (#2F81F7), while biomechanical or activation states use restrained muscle green (#3FB950). High-load or warning states should use signal orange (#DB6D28), and critical/error states should use alert red (#F85149). The overall tone should feel like a modern biomechanics simulation platform or robotics research interface: minimal, analytical, matte, high-contrast, and technically credible rather than glossy or consumer-oriented.