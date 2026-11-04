import os

os.environ["LUMEN_MOCK"] = "1"

from app.guidance import Box, cue_for
from app import models
from app.models import parse_boxes, parse_gemini_box, parse_mistral_box

SAMPLE = (
    "<0f>0<|box;x1=71;y1=111;x2=169;y2=238;w=640;h=400|>"
    "<|mask;x=0;y=0;data=128,99,~abc|>,"
    "1<|box;x1=300;y1=100;x2=400;y2=200;w=640;h=400|><|mask;x=0;y=0;data=1,2,~x|>"
)


def test_parse_boxes_reads_every_match():
    boxes = parse_boxes(SAMPLE)
    assert len(boxes) == 2
    assert (boxes[0].x1, boxes[0].x2, boxes[0].y2, boxes[0].w) == (
        71, 170, 239, 640
    )


def test_parse_boxes_empty_is_fine():
    assert parse_boxes("") == []


def test_parse_boxes_handles_reordered_fields_and_skips_malformed_boxes():
    text = (
        "<|box;w=640;h=400;y2=238;x2=169;y1=111;x1=71;c=0.93|>"
        "<|box;x1=oops;y1=1;x2=2;y2=3;w=640;h=400|>"
    )
    boxes = parse_boxes(text)
    assert len(boxes) == 1
    assert (boxes[0].x1, boxes[0].x2, boxes[0].y2) == (71, 170, 239)


def test_parse_mistral_box():
    boxes = parse_mistral_box('{"found": true, "box": [100, 200, 400, 600]}')
    assert len(boxes) == 1
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (
        100, 200, 400, 600
    )
    assert parse_mistral_box('{"found": false, "box": null}') == []


def test_mistral_find_object_uses_vision_result(monkeypatch):
    monkeypatch.setenv("LUMEN_MOCK", "0")
    monkeypatch.setenv("LUMEN_PROVIDER", "mistral")
    monkeypatch.setattr(
        models,
        "_mistral_complete",
        lambda prompt, image, mime: '{"found": true, "box": [100, 200, 400, 600]}',
    )

    boxes = models.find_object(b"image-bytes", "keys", "image/png")
    assert len(boxes) == 1
    assert boxes[0].w == 1000 and boxes[0].h == 1000


def test_gemini_find_object_uses_vision_result(monkeypatch):
    monkeypatch.setenv("LUMEN_MOCK", "0")
    monkeypatch.setenv("LUMEN_PROVIDER", "gemini")
    monkeypatch.setattr(
        models,
        "_gemini_complete",
        lambda prompt, image, mime: '{"found": true, "box": [100, 200, 400, 600]}',
    )

    boxes = models.find_object(b"image-bytes", "keys", "image/png")
    assert len(boxes) == 1
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (
        100, 200, 400, 600
    )
    assert parse_gemini_box('{"found": false, "box": null}') == []


def test_gemini_complete_sends_text_and_image(monkeypatch):
    from google.genai import types

    class Models:
        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"text": "  Gemini response  "})()

    class Client:
        models = Models()

    client = Client()
    monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    monkeypatch.setattr(models, "_gemini_client", lambda: client)

    assert models._gemini_complete("Locate keys", b"image-bytes", "image/png") == (
        "Gemini response"
    )
    request = client.models.kwargs
    assert request["model"] == "gemini-flash-lite-latest"
    assert request["contents"][0] == "Locate keys"
    assert isinstance(request["contents"][1], types.Part)


def test_cues():
    assert "left" in cue_for(Box(10, 100, 60, 150, 640, 400))["say"]
    assert "right" in cue_for(Box(560, 100, 620, 150, 640, 400))["say"]
    assert cue_for(None)["found"] is False
    far = cue_for(Box(300, 180, 340, 220, 640, 400))
    assert not far["done"] and "forward" in far["say"]
    near = cue_for(Box(180, 40, 460, 320, 640, 400))
    assert near["done"] and "Reach" in near["say"]


def test_approach_alert_and_tracker():
    from app.watch import Tracker, approach_alert

    near = Box(180, 40, 460, 320, 640, 400)
    alert = approach_alert("bicycle", 0.056, near, 2.0)
    assert alert and "Bicycle approaching" in alert["say"] and alert["close"]
    assert approach_alert("bicycle", 0.30, near, 2.0) is None
    assert approach_alert("bicycle", 0.056, near, 10.0) is None
    assert approach_alert("bicycle", 0.056, near, 0.0) is None

    tracker = Tracker()
    small = Box(250, 140, 350, 240, 640, 400)
    assert tracker.update("session", "car", small, 0.0) is None
    assert tracker.update("session", "car", near, 2.0) is not None
    assert tracker.update("session", "car", None, 3.0) is None


def test_mock_router():
    assert models.route("Remember my keys") == {"tool": "remember", "label": "keys"}
    assert models.route("Where are my keys?")["tool"] == "recall"
    assert models.route("Guide me to my keys") == {"tool": "guide", "label": "keys"}
    assert models.route("Read this label")["tool"] == "read_text"
    assert models.route("Hello there")["tool"] == "chat"


def test_gemini_is_default_provider(monkeypatch):
    monkeypatch.delenv("LUMEN_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert models.provider() == "gemini"
    assert models.gemini_model() == "gemini-flash-lite-latest"


def test_full_flow_in_mock_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_DB", str(tmp_path / "lumen.db"))
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    img = {"image": ("f.jpg", b"fake-bytes", "image/jpeg")}
    r = c.post("/remember", data={"label": "keys"}, files=img).json()
    assert r["saved"] and "fridge" in r["say"]
    r = c.post("/recall", json={"question": "where are my keys?"}).json()
    assert r["found"] and "fridge" in r["say"]
    r = c.post("/guide", data={"target": "keys"}, files=img).json()
    assert "say" in r
    assert c.post("/guide", data={"target": " "}, files=img).status_code == 400


def test_agent_watch_and_demo_page(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_DB", str(tmp_path / "lumen.db"))
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    image = {"image": ("frame.jpg", b"fake-bytes", "image/jpeg")}
    response = client.post(
        "/agent", data={"text": "Remember my keys"}, files=image
    ).json()
    assert response["tool"] == "remember" and "fridge" in response["say"]
    response = client.post(
        "/agent", data={"text": "Where are my keys?"}
    ).json()
    assert response["tool"] == "recall"
    response = client.post(
        "/agent", data={"text": "Guide me to my keys"}
    ).json()
    assert response["start_guide"] == "keys"
    assert client.post("/agent", data={"text": "Hello"}).json()["tool"] == "chat"
    response = None
    for _ in range(4):
        response = client.post(
            "/watch",
            data={"session_id": "test", "concepts": "bicycle"},
            files=image,
        ).json()
    assert response and response["alerts"]
    assert client.get("/").status_code == 200


def test_model_failures_return_safe_message(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_DB", str(tmp_path / "lumen.db"))
    from fastapi.testclient import TestClient
    from app import models
    from app.main import app
    from app.safety import FRIENDLY

    def fail(*args, **kwargs):
        raise RuntimeError("provider failure contains secret")

    monkeypatch.setattr(models, "find_object", fail)
    client = TestClient(app)
    image = {"image": ("frame.jpg", b"fake-bytes", "image/jpeg")}
    response = client.post(
        "/remember", data={"label": "keys"}, files=image
    )
    assert response.status_code == 200
    assert response.json()["say"] == FRIENDLY
    assert "secret" not in response.text
