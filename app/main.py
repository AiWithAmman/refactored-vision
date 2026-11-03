from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import models, store  # noqa: E402
from .guidance import cue_for  # noqa: E402

app = FastAPI(title="Lumen")


class RecallBody(BaseModel):
    question: str


# api endpoints
@app.get("/health")
def health():
    return {"ok": True, "mock": models.is_mock()}


@app.post("/remember")
async def remember(label: str = Form(...), image: UploadFile = File(...)):
    """SAM 3.1 confirms the object is in view; Muse Spark describes where it is."""
    data = await image.read()
    if not models.find_object(data, label):
        return {"saved": False,
                "say": f"I can't see your {label}. Hold it in view and try again."}
    place = models.describe_place(data, label)
    item_id = store.add_item(label, place, data)
    return {"saved": True, "id": item_id, "place": place,
            "say": f"Got it. You left your {label} {place}."}


@app.post("/recall")
def recall(body: RecallBody):
    """Muse Spark reasons over everything remembered and answers in speech-ready text."""
    items = store.list_items()
    if not items:
        return {"found": False, "say": "I don't remember where you put anything yet."}
    result = models.pick_memory(body.question, items)
    chosen = next((i for i in items if i["id"] == result["id"]), None)
    return {"found": chosen is not None, "item": chosen, "say": result["say"]}


@app.post("/guide")
async def guide(target: str = Form(...), image: UploadFile = File(...)):
    """Call in a loop with fresh frames. Returns a left/right/closer cue each time."""
    data = await image.read()
    boxes = models.find_object(data, target)
    best = max(boxes, key=lambda b: b.area, default=None)
    return cue_for(best)


@app.get("/items")
def items():
    return store.list_items()
