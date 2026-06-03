"""Google generativelanguage client — direct port of main.js generation logic."""
import asyncio
import base64
import time

import httpx

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

IMAGE_MODEL_ALIASES = {
    "imagen-4": "imagen-4.0-generate-001",
    "imagen-4-fast": "imagen-4.0-fast-generate-001",
    "imagen-4-ultra": "imagen-4.0-ultra-generate-001",
}
VIDEO_MODEL_ALIASES = {
    "veo-3": "veo-3.0-generate-001",
    "veo-3-fast": "veo-3.0-fast-generate-001",
    "veo-2": "veo-2.0-generate-001",
}
IMAGEN_MAX_BATCH = {"imagen-4.0-ultra-generate-001": 1}  # Ultra returns one sample per call

IMAGE_ASPECTS = ("1:1", "3:4", "4:3", "9:16", "16:9")
VIDEO_ASPECTS = ("16:9", "9:16")


class GenerationError(Exception):
    """Readable, user-facing generation failure."""


def resolve_image_model(alias: str) -> str:
    if alias not in IMAGE_MODEL_ALIASES:
        raise GenerationError(f"Unknown image model '{alias}'. Use one of: {', '.join(IMAGE_MODEL_ALIASES)}")
    return IMAGE_MODEL_ALIASES[alias]


def resolve_video_model(alias: str) -> str:
    if alias not in VIDEO_MODEL_ALIASES:
        raise GenerationError(f"Unknown video model '{alias}'. Use one of: {', '.join(VIDEO_MODEL_ALIASES)}")
    return VIDEO_MODEL_ALIASES[alias]


def imagen_max_batch(model: str) -> int:
    return IMAGEN_MAX_BATCH.get(model, 4)


async def fetch_json(client: httpx.AsyncClient, url: str, *, method="POST", headers=None,
                     json_body=None, max_retries=3, retry_delay=1.0) -> dict:
    """Port of main.js fetchJson: retry transport errors + 429/5xx with linear backoff;
    other 4xx raise immediately with the response body (Google's error message)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.request(method, url, headers=headers, json=json_body)
        except httpx.HTTPError as e:
            last_err = GenerationError(f"Request failed: {e}")
            await asyncio.sleep(retry_delay * (attempt + 1))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = GenerationError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            await asyncio.sleep(retry_delay * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise GenerationError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            return resp.json()
        except ValueError as e:
            raise GenerationError(f"Invalid JSON from {url}: {e}") from e
    raise last_err or GenerationError("Request failed after retries")


# ---- Imagen ----

def parse_imagen_predictions(json_resp: dict) -> list:
    preds = json_resp.get("predictions") or json_resp.get("generatedImages") or []
    out = []
    for p in preds:
        b64 = (p.get("bytesBase64Encoded")
               or (p.get("image") or {}).get("imageBytes")
               or (p.get("image") or {}).get("bytesBase64Encoded")
               or p.get("imageBytes"))
        if b64:
            out.append(base64.b64decode(b64))
    return out


async def call_imagen(client, api_key, model, prompt, sample_count=1, aspect_ratio="1:1") -> list:
    url = f"{GEMINI_API}/models/{model}:predict"
    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": max(1, min(4, sample_count)),
            "aspectRatio": aspect_ratio,
            "personGeneration": "allow_adult",
        },
    }
    json_resp = await fetch_json(client, url, headers={"x-goog-api-key": api_key}, json_body=body)
    return parse_imagen_predictions(json_resp)


# ---- Gemini image (edit / reference images) ----

def parse_gemini_parts(json_resp: dict) -> list:
    candidates = json_resp.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    return [base64.b64decode(p["inlineData"]["data"]) for p in parts
            if isinstance(p.get("inlineData"), dict) and p["inlineData"].get("data")]


async def call_gemini_image(client, api_key, prompt, reference_images=(), model=DEFAULT_GEMINI_IMAGE_MODEL) -> list:
    """reference_images: list of (mime, raw_bytes). Refs go before the text part (main.js order)."""
    url = f"{GEMINI_API}/models/{model}:generateContent"
    parts = [{"inlineData": {"mimeType": m, "data": base64.b64encode(b).decode()}} for (m, b) in reference_images]
    parts.append({"text": prompt})
    body = {"contents": [{"parts": parts}]}
    json_resp = await fetch_json(client, url, headers={"x-goog-api-key": api_key}, json_body=body)
    images = parse_gemini_parts(json_resp)
    if not images:
        raise GenerationError("Gemini returned no images (prompt may have been refused — try rephrasing)")
    return images


# ---- Veo ----

def build_veo_body(model, prompt, start_image, aspect_ratio, duration_seconds, generate_audio) -> dict:
    instance = {"prompt": prompt}
    if start_image:  # (mime, raw_bytes)
        mime, raw = start_image
        instance["image"] = {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode()}}
    parameters = {
        "aspectRatio": aspect_ratio,
        "durationSeconds": str(duration_seconds),
        "sampleCount": 1,
        "personGeneration": "allow_adult",
    }
    # generateAudio is a Veo 3 affordance; Veo 2 ignores it (main.js parity)
    if generate_audio is not None and model.startswith("veo-3."):
        parameters["generateAudio"] = bool(generate_audio)
    return {"instances": [instance], "parameters": parameters}


async def start_veo(client, api_key, model, prompt, start_image=None,
                    aspect_ratio="16:9", duration_seconds=8, generate_audio=True) -> str:
    url = f"{GEMINI_API}/models/{model}:predictLongRunning"
    body = build_veo_body(model, prompt, start_image, aspect_ratio, duration_seconds, generate_audio)
    json_resp = await fetch_json(client, url, headers={"x-goog-api-key": api_key}, json_body=body)
    if not json_resp.get("name"):
        raise GenerationError("Veo: no operation name returned")
    return json_resp["name"]


def parse_veo_done(json_resp: dict) -> dict:
    resp = json_resp.get("response") or {}
    samples = ((resp.get("generateVideoResponse") or {}).get("generatedSamples")
               or resp.get("generatedSamples") or [])
    if not samples:
        raise GenerationError("Veo finished with no video samples (likely safety-filtered)")
    s = samples[0]
    video = s.get("video") or {}
    return {
        "uri": video.get("uri") or s.get("uri") or video.get("url"),
        "inline_b64": video.get("bytesBase64Encoded") or s.get("bytesBase64Encoded"),
    }


async def poll_veo(client, api_key, operation_name, on_progress=None) -> dict:
    """Poll until done. 4s first wait then 8s interval (main.js cadence). Transient errors skipped."""
    url = f"{GEMINI_API}/{operation_name}"
    start = time.monotonic()
    delay = 4.0
    while True:
        await asyncio.sleep(delay)
        delay = 8.0
        try:
            resp = await client.get(url, headers={"x-goog-api-key": api_key})
            json_resp = resp.json()
        except (httpx.HTTPError, ValueError):
            continue  # transient — keep polling
        if json_resp.get("error"):
            err = json_resp["error"]
            raise GenerationError(f"Veo error: {err.get('message') or err}")
        elapsed = int(time.monotonic() - start)
        if json_resp.get("done"):
            return parse_veo_done(json_resp)
        if on_progress:
            on_progress(f"Generating video… {elapsed}s elapsed")


async def download_veo_video(client, sample: dict, api_key: str) -> bytes:
    if sample.get("inline_b64"):
        return base64.b64decode(sample["inline_b64"])
    uri = sample.get("uri")
    if not uri:
        raise GenerationError("Veo response missing both video URI and inline bytes")
    sep = "&" if "?" in uri else "?"
    attempts = [
        (uri, {"x-goog-api-key": api_key}),
        (f"{uri}{sep}key={api_key}", {}),
        (uri, {}),
    ]
    last_err = None
    for url, headers in attempts:
        try:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            if resp.status_code != 200:
                last_err = GenerationError(f"Download HTTP {resp.status_code}")
                continue
            if resp.content:
                return resp.content
            last_err = GenerationError("Empty MP4 response")
        except httpx.HTTPError as e:
            last_err = GenerationError(str(e))
    raise last_err or GenerationError("Veo MP4 download failed")
