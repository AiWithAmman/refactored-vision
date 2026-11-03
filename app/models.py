"""All calls to Meta models live here, so the rest of the app never touches the API.

LUMEN_MOCK=1 returns canned answers so you can build the app without credits.
"""

import base64
import json
import os
import re
from itertools import cycle
from typing import List, Optional

from .guidance import Box

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"

# SAM 3.1 responses look like:
# <|box;x1=71;y1=111;x2=169;y2=238;w=640;h=400|>
BOX_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>"
)


def is_mock() -> bool:
    return os.getenv("LUMEN_MOCK") == "1"


def spark_model() -> str:
    return os.getenv("SPARK_MODEL", "muse-spark-1.3")


def _client():
    if is_mock():
        return None

    api_key = os.getenv("MODEL_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise RuntimeError("MODEL_API_KEY is missing or still set to the placeholder value.")

    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=api_key)


def _data_url(image: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def _image_input(prompt: str, image: Optional[bytes]) -> list:
    content = [{"type": "input_text", "text": prompt}]
    if image:
        content.append({"type": "input_image", "image_url": _data_url(image)})
    return [{"role": "user", "content": content}]


def parse_boxes(text: str) -> List[Box]:
    return [Box(*map(int, m)) for m in BOX_RE.findall(text or "")]


_MOCK_BOXES = cycle([
    Box(20, 150, 90, 220, 640, 400),
    Box(150, 140, 260, 250, 640, 400),
    Box(260, 120, 380, 240, 640, 400),
    Box(180, 40, 460, 320, 640, 400),
])


def find_object(image: bytes, concept: str) -> List[Box]:
    """One concrete noun phrase per call, e.g. 'keys' or 'yellow school bus'."""
    if is_mock():
        return [next(_MOCK_BOXES)]

    client = _client()
    resp = client.responses.create(
        model=SAM_MODEL,
        input=_image_input(concept, image),
        store=False,
    )
    return parse_boxes(resp.output_text)


def ask_spark(prompt: str, image: Optional[bytes] = None) -> str:
    if is_mock():
        if "where it is" in prompt.lower():
            return "on top of the fridge in the kitchen"
        if "remember" in prompt.lower() and "json" in prompt.lower():
            return '{"id": null, "say": "I do not remember where that is."}'
        return "I understand."

    client = _client()
    resp = client.responses.create(
        model=spark_model(),
        input=_image_input(prompt, image),
        store=False,
    )
    return resp.output_text.strip()


def describe_place(image: bytes, label: str) -> str:
    """Short place description, e.g. 'on top of the fridge in the kitchen'."""
    if is_mock():
        return "on top of the fridge in the kitchen"

    return ask_spark(
        f"A blind user just put down their {label}. Say where it is in ONE short "
        "phrase that starts with a position word (on, in, next to, under...), "
        "naming the surface and the room if you can tell. "
        "Example: 'on top of the fridge in the kitchen'. Reply with the phrase only.",
        image,
    ).rstrip(".")


def pick_memory(question: str, items: List[dict]) -> dict:
    """Pick which remembered item the user means and write the spoken answer."""
    if is_mock():
        q = question.lower()
        for it in items:
            if it["label"].lower() in q:
                return {
                    "id": it["id"],
                    "say": f"You left your {it['label']} {it['place']}.",
                }
        return {"id": None, "say": "I don't remember where that is."}

    prompt = (
        "You are Lumen, a voice assistant for a blind user. These are the things "
        "you remember (JSON):\n" + json.dumps(items) + "\n\n"
        f'The user asked: "{question}"\n'
        'Reply with JSON only: {"id": <item id or null>, "say": "<one or two short '
        'spoken sentences>"}. Pick the item the user most likely means; if nothing '
        "matches, use null and say you don't remember. Never guess a location."
    )

    raw = ask_spark(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found")
        data = json.loads(raw[start:end])
        return {"id": data.get("id"), "say": str(data["say"])}
    except (ValueError, TypeError, KeyError):
        return {
            "id": None,
            "say": "Sorry, I couldn't work that out. Please ask again.",
        }
