from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import models, store
from .guidance import cue_for

app = FastAPI(title="Lumen")


class RecallBody(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True, "mock": models.is_mock()}


@app.post("/remember")
async def remember(label: str = Form(...), image: UploadFile = File(...)):
    """SAM confirms the object is in view, then Spark describes where it is."""
    if not label or not label.strip():
        raise HTTPException(status_code=400, detail="label is required")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image is required")

    boxes = models.find_object(data, label)
    if not boxes:
        return {
            "saved": False,
            "say": f"I can't see your {label}. Hold it in view and try again.",
        }

    place = models.describe_place(data, label)
    item_id = store.add_item(label, place, data)

    return {
        "saved": True,
        "id": item_id,
        "place": place,
        "say": f"Got it. You left your {label} {place}.",
    }


@app.post("/recall")
def recall(body: RecallBody):
    """Use remembered items to answer a spoken question."""
    items = store.list_items()
    if not items:
        return {"found": False, "say": "I don't remember where you put anything yet."}

    result = models.pick_memory(body.question, items)
    chosen = next((i for i in items if i["id"] == result["id"]), None)

    return {
        "found": chosen is not None,
        "item": chosen,
        "say": result["say"],
    }


@app.post("/guide")
async def guide(target: str = Form(...), image: UploadFile = File(...)):
    """Call this in a loop with a fresh frame from the user's camera."""
    if not target or not target.strip():
        raise HTTPException(status_code=400, detail="target is required")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="image is required")

    boxes = models.find_object(data, target)
    best = max(boxes, key=lambda b: b.area, default=None)
    return cue_for(best)


@app.get("/items")
def items():
    return store.list_items()
