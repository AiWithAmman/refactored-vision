# Lumen

An assistive prototype for remembering where objects were placed, recalling those locations, and offering visual directional cues. Built for the Meta Global AI Developer Hackathon.

## Model providers

Choose the provider with `LUMEN_PROVIDER`:

| Provider | Object localization | Place description and recall |
|---|---|---|
| `mistral` | Mistral vision model (`MISTRAL_VISION_MODEL`) estimates a bounding box from the uploaded image | Same vision-capable model |
| `meta` | SAM 3.1 (`sam-3.1`) | Muse Spark (`SPARK_MODEL`) |

Mistral vision-model bounding boxes are approximate, not a substitute for dedicated object detection, depth sensing, or safe mobility guidance. Lumen does not estimate distance or detect hazards.

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

Open http://localhost:8000/docs to try the endpoints. In `/health`, confirm `"mock": false` to ensure requests use the configured live provider.

### Try live Mistral vision

1. Create a Mistral API key, then set `LUMEN_PROVIDER=mistral`, `MISTRAL_API_KEY=<your-key>`, and `LUMEN_MOCK=0` in `.env`. Keep the key private and do not commit `.env`.
2. Start the server and open `/docs`.
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
| `GET /items` | | everything remembered |

```bash
curl -F label=keys -F image=@keys.jpg localhost:8000/remember
curl -H 'content-type: application/json' -d '{"question":"where are my keys?"}' localhost:8000/recall
curl -F target=keys -F image=@frame.jpg localhost:8000/guide
```

## Meta provider

To use Meta instead, set `LUMEN_PROVIDER=meta`, `MODEL_API_KEY=<your-key>`, and `LUMEN_MOCK=0`. The Meta request formats and model availability still need to be verified with a real account and key.

## Current limitations

- No live camera, speech input, or text-to-speech integration; endpoints accept uploaded image files and text.
- No `/watch` hazard alert, depth sensing, or reliable distance estimation.
- Model output can be wrong or uncertain. Use the app only as an experimental assistive aid, never as a replacement for a cane, guide dog, or other established mobility aid.

Mistral vision follows its [official vision API documentation](https://docs.mistral.ai/studio/conversations/vision/); Meta requests use the standard tier with `store=False`.
