#!/usr/bin/env bash
# Train + submit as two separate users (SB3 and mjlab-style pipelines).
#
# Offline (embedded API, no server):  ./scripts/run_e2e_submit.sh offline
# Local HTTP server:                  ./scripts/run_e2e_submit.sh local
# Production API:                     ./scripts/run_e2e_submit.sh live
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-offline}"
VENV="${ROOT}/.test-venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Creating test venv (Python 3.10 recommended)..."
  python3.10 -m venv "${VENV}" 2>/dev/null || python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -q -U pip
  "${VENV}/bin/pip" install -q -e "${ROOT}[test]"
  "${VENV}/bin/pip" install -q -r "${ROOT}/../mcEvalProd/requirements.txt"
fi

run_pytest() {
  local extra=()
  if [[ -n "${1:-}" ]]; then
    extra=(-k "$1")
  fi
  "${VENV}/bin/python" -m pytest "${ROOT}/tests/test_e2e_two_user_submit.py" -v -s "${extra[@]}"
}

case "${MODE}" in
  offline)
    run_pytest offline
    ;;
  local)
    API_PID=""
    cleanup() {
      [[ -n "${API_PID}" ]] && kill "${API_PID}" 2>/dev/null || true
    }
    trap cleanup EXIT
    echo "Starting local API on http://127.0.0.1:8080 ..."
    (
      cd "${ROOT}/../mcEvalProd"
      DB_URL="sqlite:////tmp/mc26_e2e.db" USE_LOCAL_STORAGE=true EVAL_ON_SUBMIT=false \
        "${VENV}/bin/python" start_server.py
    ) &
    API_PID=$!
    for _ in $(seq 1 30); do
      curl -sf http://127.0.0.1:8080/challenge-info >/dev/null && break
      sleep 1
    done
    LIVE_API_BASE=http://127.0.0.1:8080 run_pytest live
    ;;
  live)
    LIVE_API_BASE="${LIVE_API_BASE:-https://api.mc.myosuite.org}" run_pytest live
    ;;
  *)
    echo "Usage: $0 [offline|local|live]" >&2
    exit 1
    ;;
esac
