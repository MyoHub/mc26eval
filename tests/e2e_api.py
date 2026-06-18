"""HTTP helpers for mc26eval end-to-end submission tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

TASK = "elbowflexion"
PHASE = 1


@dataclass
class RegisteredUser:
    email: str
    team_name: str
    auth_token: str
    user_id: int | None = None


class ApiClient:
    """Thin wrapper over requests or FastAPI TestClient."""

    def __init__(self, base_url: str | None = None, test_client: Any | None = None):
        if test_client is not None:
            self._client = test_client
            self._base = ""
        else:
            if not base_url:
                raise ValueError("base_url required when test_client is not provided")
            self._base = base_url.rstrip("/")
            self._client = None

    def _request(self, method: str, path: str, **kwargs) -> Any:
        if self._client is not None:
            return getattr(self._client, method)(path, **kwargs)
        return requests.request(method, f"{self._base}{path}", timeout=60, **kwargs)

    def ensure_task_phase(self) -> None:
        existing = self._request("get", f"/task-phase/{TASK}/{PHASE}")
        if existing.status_code == 200:
            return
        r = self._request(
            "post",
            "/task-phase",
            json={"task": TASK, "phase": PHASE, "score_names": ["Score", "Time", "Effort", "Pose Error"]},
        )
        assert r.status_code == 200, getattr(r, "text", r.status_code)

    def register(self, email: str, team_name: str, password: str = "testpass99") -> RegisteredUser:
        r = self._request(
            "post",
            "/register",
            json={"email": email, "team_name": team_name, "password": password},
        )
        assert r.status_code == 200, getattr(r, "text", r.body if hasattr(r, "body") else r)
        body = r.json()
        return RegisteredUser(
            email=email,
            team_name=team_name,
            auth_token=body["auth_token"],
            user_id=body.get("user_id"),
        )

    def submit_onnx(self, user: RegisteredUser, onnx_path: Path) -> dict:
        with open(onnx_path, "rb") as f:
            r = self._request(
                "post",
                "/submit/onnx",
                data={
                    "task": TASK,
                    "phase": PHASE,
                    "email": user.email,
                    "auth_token": user.auth_token,
                },
                files={"onnx_model": (onnx_path.name, f, "application/octet-stream")},
            )
        assert r.status_code == 200, getattr(r, "text", r)
        body = r.json()
        assert body["status"] == "submitted"
        assert body["submission_type"] == "onnx"
        assert isinstance(body["submission_id"], int)
        return body


def live_api_client() -> ApiClient | None:
    base = os.getenv("LIVE_API_BASE", "").strip().rstrip("/")
    if not base:
        return None
    return ApiClient(base_url=base)


def offline_api_client() -> ApiClient:
    import sys

    root = Path(__file__).resolve().parents[2]
    prod = root / "mcEvalProd"
    if str(prod) not in sys.path:
        sys.path.insert(0, str(prod))

    os.environ.setdefault("DB_URL", "sqlite:///:memory:")
    os.environ.setdefault("USE_LOCAL_STORAGE", "true")
    os.environ.setdefault("EVAL_ON_SUBMIT", "false")

    from app.api.admin_api import app as fastapi_app
    from app.models.db import Base, engine, init_db
    from fastapi.testclient import TestClient

    Base.metadata.drop_all(bind=engine)
    init_db()
    return ApiClient(test_client=TestClient(fastapi_app))
