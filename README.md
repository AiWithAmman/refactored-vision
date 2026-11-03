# Lumen (backend)

Voice-first assistant for blind users: remember where things were put, recall them later, and guide the last ten feet to the object. Built for the Meta Global AI Developer Hackathon.

## Meta models and their jobs

| Job | Model |
|---|---|
| Confirm the object is in view and locate it in each frame | SAM 3.1 (`sam-3.1`) |
| Describe where an object is and reason over memory | Muse Spark (`SPARK_MODEL`) |
| Speech-to-text for next step | Muse Voice Transcribe |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0
```

## Run it

```bash
pytest
```

Open http://localhost:8000/docs to try every endpoint in the browser.

## Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /remember` | form: `label`, `image` | `saved`, `place`, `say` |
| `POST /recall` | JSON: `{"question": "where are my keys?"}` | `found`, `item`, `say` |
| `POST /guide` | form: `target`, `image` (call in a loop) | `say`, `done`, `found` |
| `GET /items` | | everything remembered |

```bash
curl -F label=keys -F image=@keys.jpg localhost:8000/remember
curl -H 'content-type: application/json' -d '{"question":"where are my keys?"}' localhost:8000/recall
curl -F target=keys -F image=@frame.jpg localhost:8000/guide
```

## Going live (set `LUMEN_MOCK=0`)

Three things were NOT verified against official docs. Check them first with a real key:

1. **SAM request shape** in `find_object`: confirm against https://dev.meta.ai/docs/media-segmentation.
   Output parsing (`<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|>`) matches a documented live response.
2. **Image input for Muse Spark** (`input_image` with a data URL via the Responses API).
3. **Model name**: default `muse-spark-1.3`; if it errors, try `muse-spark-1.2`.

Privacy: use the standard tier, not `*-contributor` models (those may train on prompts/images).
Requests send `store=False`.

## Not built yet

- `/watch` (approaching-hazard alert), the tool router, voice input.
- Speech out: the docs show speech-to-text only, so use the phone's built-in text-to-speech.
- Muse Voice is a realtime WebSocket (`wss://api.meta.ai/v1/asr/realtime`), so plan time for it.

Lumen assists and does not replace a cane or guide dog.
