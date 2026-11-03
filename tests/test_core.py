import os

os.environ["LUMEN_MOCK"] = "1"
os.environ["LUMEN_DB"] = "/tmp/lumen_test/lumen.db"

from app.guidance import Box, cue_for
from app.models import parse_boxes

SAMPLE = (
    "<0f>0<|box;x1=71;y1=111;x2=169;y2=238;w=640;h=400|>"
    "<|mask;x=0;y=0;data=128,99,~abc|>,"
    "1<|box;x1=300;y1=100;x2=400;y2=200;w=640;h=400|><|mask;x=0;y=0;data=1,2,~x|>"
)


def test_parse_boxes_reads_every_match():
    boxes = parse_boxes(SAMPLE)
    assert len(boxes) == 2
    assert (boxes[0].x1, boxes[0].y2, boxes[0].w) == (71, 238, 640)


def test_parse_boxes_empty_is_fine():
    assert parse_boxes("") == []


def test_cues():
    assert "left" in cue_for(Box(10, 100, 60, 150, 640, 400))["say"]
    assert "right" in cue_for(Box(560, 100, 620, 150, 640, 400))["say"]
    assert cue_for(None)["found"] is False
    far = cue_for(Box(300, 180, 340, 220, 640, 400))
    assert not far["done"] and "forward" in far["say"]
    near = cue_for(Box(180, 40, 460, 320, 640, 400))
    assert near["done"] and "Reach" in near["say"]


def test_full_flow_in_mock_mode():
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
