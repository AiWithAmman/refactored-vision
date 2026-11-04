"""Model API calls live here, so the rest of the app never touches the API.

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
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
GEMINI_MODEL_DEFAULT = "gemini-flash-lite-latest"

# SAM box fields may be reordered and may include confidence or other metadata.
BOX_TOKEN_RE = re.compile(r"<\|box;([^|]*)\|>")


def is_mock() -> bool:
    return os.getenv("LUMEN_MOCK") == "1"


def provider() -> str:
    selected = os.getenv("LUMEN_PROVIDER", "gemini").strip().lower()
    if selected not in {"meta", "mistral", "gemini"}:
        raise ValueError("LUMEN_PROVIDER must be 'meta', 'mistral', or 'gemini'.")
    return selected


def spark_model() -> str:
    return os.getenv("SPARK_MODEL", "muse-spark-1.3")


def mistral_model() -> str:
    return os.getenv("MISTRAL_VISION_MODEL", "mistral-small-2506")


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT)


def _openai_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _meta_client():
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key or api_key in {"your-key-here", "paste-your-key-here"}:
        raise RuntimeError("MODEL_API_KEY is missing or still set to a placeholder.")
    return _openai_client(BASE_URL, api_key)


def _mistral_client():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key in {"your-key-here", "paste-your-key-here"}:
        raise RuntimeError("MISTRAL_API_KEY is missing or still set to a placeholder.")
    return _openai_client(MISTRAL_BASE_URL, api_key)


def _gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key in {"your-key-here", "paste-your-key-here"}:
        raise RuntimeError("GEMINI_API_KEY is missing or still set to a placeholder.")
    from google import genai

    return genai.Client(api_key=api_key)


def _data_url(image: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def _meta_image_input(prompt: str, image: Optional[bytes], mime: str) -> list:
    content = [{"type": "input_text", "text": prompt}]
    if image:
        content.append({"type": "input_image", "image_url": _data_url(image, mime)})
    return [{"role": "user", "content": content}]


def _mistral_messages(prompt: str, image: Optional[bytes], mime: str) -> list:
    content = [{"type": "text", "text": prompt}]
    if image:
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(image, mime)}}
        )
    return [{"role": "user", "content": content}]


def _mistral_complete(prompt: str, image: Optional[bytes] = None,
                      mime: str = "image/jpeg") -> str:
    response = _mistral_client().chat.completions.create(
        model=mistral_model(),
        messages=_mistral_messages(prompt, image, mime),
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if content:
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            elif getattr(part, "type", None) == "text":
                text_parts.append(part.text)
        if text_parts:
            return "".join(text_parts).strip()
    raise RuntimeError("Mistral returned an empty response.")


def _gemini_complete(prompt: str, image: Optional[bytes] = None,
                     mime: str = "image/jpeg") -> str:
    from google.genai import types

    if image:
        content = [prompt, types.Part.from_bytes(data=image, mime_type=mime)]
    else:
        content = prompt
    response = _gemini_client().models.generate_content(
        model=gemini_model(),
        contents=content,
    )
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


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


def _parse_vision_box(text: str, provider_name: str) -> List[Box]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"{provider_name} localization response did not contain JSON.")

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{provider_name} localization response contained invalid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"{provider_name} localization response must be a JSON object.")
    if data.get("found") is False:
        return []
    coords = data.get("box")
    if data.get("found") is not True or not isinstance(coords, list) or len(coords) != 4:
        raise ValueError(f"{provider_name} localization response did not contain a valid box.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 1000
        for value in coords
    ):
        raise ValueError(f"{provider_name} returned box coordinates outside the 0-1000 range.")

    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{provider_name} returned an invalid bounding box.")
    return [Box(x1, y1, x2, y2, 1000, 1000)]


def parse_mistral_box(text: str) -> List[Box]:
    return _parse_vision_box(text, "Mistral")


def parse_gemini_box(text: str) -> List[Box]:
    return _parse_vision_box(text, "Gemini")


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

    if provider() in {"mistral", "gemini"}:
        complete = _mistral_complete if provider() == "mistral" else _gemini_complete
        parse_box = parse_mistral_box if provider() == "mistral" else parse_gemini_box
        prompt = (
            "Locate the most visible instance of this object: "
            f"{concept!r}. Return JSON only with this schema: "
            '{"found": true, "box": [x1, y1, x2, y2]} or '
            '{"found": false, "box": null}. Coordinates must be integers '
            "from 0 to 1000, measured from the top-left, with x scaled to "
            "image width and y scaled to image height. Give your best visual "
            "estimate; do not claim the object is present if uncertain."
        )
        return parse_box(complete(prompt, image, mime))

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

    if provider() == "mistral":
        return _mistral_complete(prompt, image, mime)
    if provider() == "gemini":
        return _gemini_complete(prompt, image, mime)

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
