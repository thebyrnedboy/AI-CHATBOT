import json
import os
import uuid

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app import app  # noqa: E402


def _client():
    return app.test_client()


def assert_ok(resp, msg=""):
    assert resp.status_code == 200, f"{msg} status {resp.status_code}"
    data = json.loads(resp.data)
    assert data.get("ok") is True, f"{msg} data {data}"
    return data


def run_health(client):
    data = assert_ok(client.get("/health"), "health")
    print("Health OK:", data)


def run_embed(client):
    res = client.get("/embed.js")
    assert res.status_code == 200, f"embed status {res.status_code}"
    body = res.get_data(as_text=True)
    assert "Chat with us" in body, "embed content missing"
    print("Embed OK")


def run_domain_block(client):
    # Simulate missing API key
    res = client.post("/chat_stream", json={"message": "hi"})
    assert res.status_code in (401, 403, 500), "chat_stream should guard missing auth"
    print("Chat_stream auth guard OK")

# --------------------
# Pytest-compatible tests
# --------------------
def test_health():
    client = _client()
    assert_ok(client.get("/health"), "health")


def test_embed():
    client = _client()
    res = client.get("/embed.js")
    assert res.status_code == 200, f"embed status {res.status_code}"
    body = res.get_data(as_text=True)
    assert "Chat with us" in body, "embed content missing"


def test_chat_stream_guard():
    client = _client()
    res = client.post("/chat_stream", json={"message": "hi"})
    assert res.status_code in (401, 403, 500), "chat_stream should guard missing auth"


def main():
    client = _client()
    run_health(client)
    run_embed(client)
    run_domain_block(client)
    print("Smoke suite passed.")


if __name__ == "__main__":
    main()
