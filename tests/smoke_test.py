import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app import app  # noqa: E402


def _client():
    return app.test_client()


def test_app_imports():
    assert app is not None


def test_health():
    client = _client()
    res = client.get("/health")
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json(silent=True) or {}
    assert data.get("ok") is True


def test_embed():
    client = _client()
    res = client.get("/embed.js")
    assert res.status_code == 200, f"embed status {res.status_code}"
    body = res.get_data(as_text=True)
    assert "Chat with us" in body, "embed content missing"


def test_protected_routes_redirect_when_unauthenticated():
    client = _client()
    res = client.get("/dashboard", follow_redirects=False)
    assert res.status_code in (302, 401)
    location = res.headers.get("Location", "")
    assert "/login" in location
    res = client.get("/admin/helpdesk", follow_redirects=False)
    assert res.status_code in (302, 401, 403)
