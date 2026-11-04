# Lumen

Lumen helps blind and low-vision people remember where they placed objects, recall those locations, and try visual cues to find them. It includes a browser demo with camera capture, voice input, and spoken responses.

## Model providers

Choose the provider with `LUMEN_PROVIDER`:

| Provider | Object localization | Place description and recall |
|---|---|---|
| `gemini` | Gemini (`GEMINI_MODEL`) estimates a bounding box from the uploaded image | Same vision-capable model |
| `mistral` | Mistral vision model (`MISTRAL_VISION_MODEL`) estimates a bounding box from the uploaded image | Same vision-capable model |
| `meta` | SAM 3.1 (`sam-3.1`) | Muse Spark (`SPARK_MODEL`) |

Gemini's default model is `gemini-flash-lite-latest`; override it with `GEMINI_MODEL` if needed. Gemini free-tier quotas and model access are subject to Google's limits and can change. Google says free-tier API content may be used to improve its products, so do not send sensitive images using a free-tier key. Mistral/Gemini vision-model boxes are approximate, not a substitute for dedicated object detection, depth sensing, or safe mobility guidance. Lumen does not estimate distance. The optional approach alert is an experimental image-size heuristic and can miss or falsely report objects.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On macOS/Linux, activate with `source .venv/bin/activate` and copy the config with `cp .env.example .env`.

For a no-credential smoke test, the example config starts in mock mode. Mock mode returns canned responses and **does not analyze uploaded images**.

## Run it

```bash
python -m pytest
```

Open http://localhost:8000 for the browser demo, or http://localhost:8000/docs to try the API. In `/health`, confirm `"mock": false` to ensure requests use the configured live provider. The demo can capture camera frames, accept typed or spoken requests, and speak responses; speech recognition is supported by browsers such as Chrome and Edge.

### Use Gemini Flash Lite

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/api-keys). Set `LUMEN_PROVIDER=gemini`, `GEMINI_API_KEY=<your-key>`, and `LUMEN_MOCK=0` in `.env`. The model defaults to `gemini-flash-lite-latest`; optionally set `GEMINI_MODEL` to another model available to your account.
2. Restart the server and check `/health` for `"provider": "gemini"` and `"mock": false`.
3. Try `POST /remember` with an image showing an object, then `/recall` and `/guide`.

The free tier is quota-limited; switching providers does not guarantee unlimited requests. See [Google's Gemini API pricing and free-tier terms](https://ai.google.dev/gemini-api/docs/pricing).

### Try live Mistral vision

1. Create a Mistral API key, then set `LUMEN_PROVIDER=mistral`, `MISTRAL_API_KEY=<your-key>`, and `LUMEN_MOCK=0` in `.env`. Keep the key private and do not commit `.env`.
2. Start the server and open the demo or `/docs`.
3. Call `POST /remember` with an image showing the object and a label (for example `keys`). Check that it returns `"saved": true` and a plausible place.
4. Call `POST /recall` with a question about that item.
5. Call `POST /guide` with a fresh image and the target label. It returns an approximate left/right/forward/reach cue based on the model-estimated box.

If Mistral rejects the configured model, select a vision model enabled for your account using `MISTRAL_VISION_MODEL`. The default is `mistral-small-2506`.

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
