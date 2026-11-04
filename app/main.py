import os
import time
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from . import models, store  # noqa: E402
from .guidance import cue_for  # noqa: E402
from .safety import safely  # noqa: E402
from .watch import Tracker  # noqa: E402

app = FastAPI(title="Lumen")
tracker = Tracker()


class RecallBody(BaseModel):
    question: str


def _need(value: str, name: str) -> str:
    if not value or not value.strip():
        raise HTTPException(status_code=400, detail=f"{name} is required")
    return value.strip()


async def _read_image(image: UploadFile) -> Tuple[bytes, str]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image is required")
    return data, image.content_type or "image/jpeg"


def do_remember(label: str, data: bytes, mime: str) -> dict:
    if not models.find_object(data, label, mime):
        return {
            "saved": False,
            "say": f"I can't see your {label}. Hold it in view and try again.",
        }
    place = models.describe_place(data, label, mime)
    item_id = store.add_item(label, place, data)
    return {
        "saved": True,
        "id": item_id,
        "place": place,
        "say": f"Got it. You left your {label} {place}.",
    }


def do_recall(question: str) -> dict:
    items = store.list_items()
    if not items:
        return {"found": False, "say": "I don't remember where you put anything yet."}
    result = models.pick_memory(question, items)
    chosen = next((item for item in items if item["id"] == result["id"]), None)
    return {"found": chosen is not None, "item": chosen, "say": result["say"]}


def do_guide(target: str, data: bytes, mime: str) -> dict:
    boxes = models.find_object(data, target, mime)
    best = max(boxes, key=lambda box: box.area, default=None)
    return cue_for(best)


def do_watch(session: str, concepts: List[str], data: bytes, mime: str) -> dict:
    now = time.time()
    alerts = []
    for concept in concepts:
        boxes = models.find_object(data, concept, mime)
        best = max(boxes, key=lambda box: box.area, default=None)
        alert = tracker.update(session, concept, best, now)
        if alert:
            alerts.append(alert)
    return {"alerts": alerts, "say": alerts[0]["say"] if alerts else ""}


@app.get("/health")
def health():
    return {
        "ok": True,
        "mock": models.is_mock(),
        "provider": "mock" if models.is_mock() else models.provider(),
    }


@app.post("/remember")
async def remember(label: str = Form(...), image: UploadFile = File(...)):
    label = _need(label, "label")
    data, mime = await _read_image(image)
    return await run_in_threadpool(safely, do_remember, label, data, mime)


@app.post("/recall")
async def recall(body: RecallBody):
    question = _need(body.question, "question")
    return await run_in_threadpool(safely, do_recall, question)


@app.post("/guide")
async def guide(target: str = Form(...), image: UploadFile = File(...)):
    target = _need(target, "target")
    data, mime = await _read_image(image)
    return await run_in_threadpool(safely, do_guide, target, data, mime)


@app.post("/watch")
async def watch(
    session_id: str = Form(...),
    concepts: str = Form("person,bicycle,car"),
    image: UploadFile = File(...),
):
    session_id = _need(session_id, "session_id")
    names = [name.strip().lower() for name in concepts.split(",") if name.strip()][:4]
    data, mime = await _read_image(image)
    return await run_in_threadpool(safely, do_watch, session_id, names, data, mime)


@app.post("/agent")
async def agent(text: str = Form(...), image: Optional[UploadFile] = File(None)):
    text = _need(text, "text")
    data, mime = (await _read_image(image)) if image else (None, "image/jpeg")
    intent = await run_in_threadpool(safely, models.route, text)
    if "error" in intent:
        return {"tool": "error", "label": None, **intent}

    tool, label = intent["tool"], intent.get("label")
    if tool == "remember":
        if not label:
            result = {"say": "What should I remember? Say, remember my keys."}
        elif not data:
            result = {"say": "I need the camera on to remember that."}
        else:
            result = await run_in_threadpool(safely, do_remember, label, data, mime)
    elif tool == "recall":
        result = await run_in_threadpool(safely, do_recall, text)
    elif tool == "guide":
        if not label:
            result = {"say": "What should I find? Say, guide me to my keys."}
        else:
            place = next(
                (
                    item["place"]
                    for item in store.list_items()
                    if label in item["label"].lower()
                    or item["label"].lower() in label
                ),
                None,
            )
            lead = f"You left your {label} {place}. " if place else ""
            result = {
                "say": f"{lead}I'll guide you. Slowly point the camera ahead.",
                "start_guide": label,
            }
    elif tool == "read_text":
        if not data:
            result = {"say": "I need the camera on to read that."}
        else:
            result = await run_in_threadpool(safely, models.read_text, data, mime)
            if not isinstance(result, dict):
                result = {"say": result}
    else:
        result = {
            "say": "I can remember where you put things, find them, guide you to them, "
            "or read text. Try: remember my keys."
        }
    return {"tool": tool, "label": label, **result}


@app.get("/items")
def items():
    return store.list_items()


STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
