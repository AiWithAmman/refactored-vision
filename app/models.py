"""All Meta Model API calls live here, so the rest of the app never touches the API.

SAM 3.1 finds objects in a frame; Muse Spark understands, routes and reasons.
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

# SAM box fields may be reordered and may include confidence or other metadata.
BOX_TOKEN_RE = re.compile(r"<\|box;([^|]*)\|>")


def is_mock() -> bool:
    return os.getenv("LUMEN_MOCK") == "1"


def provider() -> str:
    return "meta"


def spark_model() -> str:
    return os.getenv("SPARK_MODEL", "muse-spark-1.3")


def _openai_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _meta_client():
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key or api_key in {"your-key-here", "paste-your-key-here"}:
        raise RuntimeError("MODEL_API_KEY is missing or still set to a placeholder.")
    return _openai_client(BASE_URL, api_key)


def _data_url(image: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def _meta_image_input(prompt: str, image: Optional[bytes], mime: str) -> list:
    content = [{"type": "input_text", "text": prompt}]
    if image:
        content.append({"type": "input_image", "image_url": _data_url(image, mime)})
    return [{"role": "user", "content": content}]


def parse_boxes(text: str) -> List[Box]:
    boxes = []
    for body in BOX_TOKEN_RE.findall(text or ""):
        fields = {
            key.strip(): value.strip()
            for key, value in (
                part.split("=", 1) for part in body.split(";") if "=" in part
            )
        }
        try:
            x1, y1, x2, y2, width, height = (
                int(fields[key]) for key in ("x1", "y1", "x2", "y2", "w", "h")
            )
        except (KeyError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2 + 1, width), min(y2 + 1, height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(x1, y1, x2, y2, width, height))
    return boxes


_MOCK_PATH = [
    Box(20, 150, 90, 220, 640, 400),
    Box(150, 140, 260, 250, 640, 400),
    Box(260, 120, 380, 240, 640, 400),
    Box(180, 40, 460, 320, 640, 400),
]
_mock_cycles: dict = {}


def _mock_box(concept: str) -> Box:
    return next(_mock_cycles.setdefault(concept, cycle(_MOCK_PATH)))


def find_object(image: bytes, concept: str, mime: str = "image/jpeg") -> List[Box]:
    """One concrete noun phrase per call, e.g. 'keys' or 'yellow school bus'."""
    if is_mock():
        return [_mock_box(concept)]

    resp = _meta_client().responses.create(
        model=SAM_MODEL,
        input=_meta_image_input(concept, image, mime),
        store=False,
    )
    return parse_boxes(resp.output_text)


def ask_spark(prompt: str, image: Optional[bytes] = None,
              mime: str = "image/jpeg") -> str:
    if is_mock():
        if "where it is" in prompt.lower():
            return "on top of the fridge in the kitchen"
        if "remember" in prompt.lower() and "json" in prompt.lower():
            return '{"id": null, "say": "I do not remember where that is."}'
        return "I understand."

    resp = _meta_client().responses.create(
        model=spark_model(),
        input=_meta_image_input(prompt, image, mime),
        store=False,
    )
    return resp.output_text.strip()


def describe_place(image: bytes, label: str, mime: str = "image/jpeg") -> str:
    """Short place description, e.g. 'on top of the fridge in the kitchen'."""
    if is_mock():
        return "on top of the fridge in the kitchen"

    return ask_spark(
        f"A blind user just put down their {label}. Say where it is in ONE short "
        "phrase that starts with a position word (on, in, next to, under...), "
        "naming the surface and the room if you can tell. "
        "Example: 'on top of the fridge in the kitchen'. Reply with the phrase only.",
        image,
        mime,
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
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return {"id": data.get("id"), "say": str(data["say"])}
    except (ValueError, TypeError, KeyError):
        return {
            "id": None,
            "say": "Sorry, I couldn't work that out. Please ask again.",
        }


TOOLS = ("remember", "recall", "guide", "read_text", "chat")


def _mock_route(text: str) -> dict:
    lowered = text.lower()
    match = re.search(r"\bmy ([a-z][a-z ]*?)\s*[?.!]*$", lowered)
    label = match.group(1).strip() if match else None
    if re.search(r"\bread\b", lowered):
        tool = "read_text"
    elif any(phrase in lowered for phrase in ("guide", "take me", "navigate", "lead me")):
        tool = "guide"
    elif any(word in lowered for word in ("where", "find", "lost")):
        tool = "recall"
    elif any(word in lowered for word in ("remember", "put my", "left my", "placed")):
        tool = "remember"
    else:
        tool = "chat"
    return {"tool": tool, "label": label}


def route(text: str) -> dict:
    """Choose the tool for a request and extract its object label."""
    if is_mock():
        return _mock_route(text)

    prompt = (
        "You route requests for Lumen, a voice assistant for blind users. Tools:\n"
        "- remember: the user is putting an object down and wants it remembered\n"
        "- recall: the user asks where something is\n"
        "- guide: the user wants to be led to an object right now\n"
        "- read_text: the user wants text in front of them read aloud\n"
        "- chat: anything else\n"
        f'Request: "{text}"\n'
        'Reply with JSON only: {"tool": "<tool>", "label": "<the object as a short noun, e.g. keys, or null>"}'
    )
    raw = ask_spark(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON object found")
        data = json.loads(raw[start:end])
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        tool = data.get("tool") if data.get("tool") in TOOLS else "chat"
        label = data.get("label")
        return {"tool": tool, "label": str(label).strip().lower() if label else None}
    except (ValueError, TypeError):
        return {"tool": "chat", "label": None}


def read_text(image: bytes, mime: str = "image/jpeg") -> str:
    if is_mock():
        return "Mock mode. I would read any text I can see here."
    return ask_spark(
        "Read aloud all the text visible in this image, in natural reading order. "
        "Be brief. If there is no readable text, say: I don't see any text.",
        image,
        mime,
    )