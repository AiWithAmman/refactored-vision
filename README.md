# Lumen

Lumen is an experimental browser app for blind and low-vision users. It can remember where an object was placed, answer questions about saved items, and provide visual cues for locating an object. The browser demo supports camera capture, typed or spoken requests, and spoken responses.

## Meta configuration

Lumen's Meta integration sends requests to `https://api.meta.ai/v1` using the `openai` Python package. It uses SAM 3.1 for object localization and Muse Spark for descriptions, recall, and text reading.

The integration requires a valid `MODEL_API_KEY` with access to the configured endpoint and models. Access is not guaranteed. The linked [Strvm/meta-ai-api](https://github.com/Strvm/meta-ai-api) project is an unofficial wrapper for Meta AI's website; Lumen does not use it, and its README does not describe the image-analysis API Lumen needs.

## Setup on Windows

Open a PowerShell terminal in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` and copy the config with `cp .env.example .env`.

For a no-credential smoke test, the example config starts in mock mode. Mock mode returns canned responses and **does not analyze uploaded images**.

## Run it

```bash
python -m pytest
```

Open http://localhost:8000 for the browser demo, or http://localhost:8000/docs to try the API. In `/health`, confirm `"mock": false` to ensure requests use the configured live provider. The demo can capture camera frames, accept typed or spoken requests, and speak responses; speech recognition is supported by browsers such as Chrome and Edge.

## Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /remember` | form: `label`, `image` | `saved`, `place`, `say` |
| `POST /recall` | JSON: `{"question": "where are my keys?"}` | `found`, `item`, `say` |
| `POST /guide` | form: `target`, `image` (call in a loop) | `say`, `done`, `found` |
| `POST /watch` | form: `session_id`, optional `concepts`, `image` | `alerts`, `say` |
| `POST /agent` | form: `text`, optional `image` | selected `tool`, `say`, optional `start_guide` |
| `GET /items` | | everything remembered |
| `GET /health` | | mock mode and selected provider |

Call `/watch` with fresh frames about every two seconds; it can make one model call per concept (up to four). Model-call failures return a speakable message and an error class, while details are written to the server log.

```bash
curl -F label=keys -F image=@keys.jpg localhost:8000/remember
curl -H 'content-type: application/json' -d '{"question":"where are my keys?"}' localhost:8000/recall
curl -F target=keys -F image=@frame.jpg localhost:8000/guide
```

## Other providers

To use Mistral, set `LUMEN_PROVIDER=mistral`, `MISTRAL_API_KEY=<your-key>`, and `LUMEN_MOCK=0`.

To use Meta, set `LUMEN_PROVIDER=meta`, `MODEL_API_KEY=<your-key>`, and `LUMEN_MOCK=0`.

## Current limitations

- The browser demo depends on browser camera and speech APIs; the API itself accepts uploaded images and text rather than a continuous video or audio stream.
- `/watch` only compares estimated 2D object-box sizes between frames; it is not hazard detection, depth sensing, or reliable distance estimation. False or missed alerts are possible. Test it only while stationary and do not rely on it for safety.
- Model output can be wrong or uncertain. Use the app only as an experimental assistive aid, never as a replacement for a cane, guide dog, or other established mobility aid.

Gemini requests use the [Google GenAI Python SDK](https://ai.google.dev/gemini-api/docs/libraries) and its [image input API](https://ai.google.dev/gemini-api/docs/image-understanding). Mistral vision follows its [official vision API documentation](https://docs.mistral.ai/studio/conversations/vision/); Meta requests use the standard tier with `store=False`.
