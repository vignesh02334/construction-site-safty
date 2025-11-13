import io
import base64
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_docs():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "paths" in r.json()


def test_predict_invalid_file():
    r = client.post("/predict", files={"file": ("x.txt", b"not-an-image", "text/plain")})
    assert r.status_code in (400, 500)


def test_events_list():
    r = client.get("/events")
    assert r.status_code == 200
    assert "events" in r.json()

