"""Google generativelanguage client — direct port of main.js generation logic."""
import asyncio
import base64
import random
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


# ---- Local diffusion via ComfyUI (uncensored; runs on the 4090 box over the LAN) ----
# Workflows below are validated against the live ComfyUI (gemini-flash/Imagen still available
# as the cloud fallback when the box is unreachable). Each returns raw PNG bytes — same
# contract as call_imagen / call_gemini_image.

LOCAL_MODELS = {
    # alias -> (checkpoint filename, family)
    "pony": ("ponyDiffusionV6XL.safetensors", "sdxl"),   # fast, maximally uncensored
    "flux": ("flux1-dev-fp8.safetensors", "flux"),        # best quality/coherence
}
# Pony responds to score tags; we prepend/append sensible defaults so callers needn't know them.
PONY_POS_PREFIX = "score_9, score_8_up, score_7_up, "
PONY_NEG_DEFAULT = ("score_6, score_5, score_4, worst quality, low quality, blurry, "
                    "jpeg artifacts, text, watermark, signature, deformed, extra limbs")


def _sdxl_workflow(ckpt, pos, neg, width, height, steps, cfg, seed):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": "sdxl_vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m_sde",
            "scheduler": "karras", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["10", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "forge", "images": ["8", 0]}},
    }


def _flux_workflow(ckpt, pos, width, height, steps, seed, guidance=3.5):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "12": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": guidance}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0,
            "model": ["4", 0], "positive": ["12", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "forge", "images": ["8", 0]}},
    }


# --- Local VIDEO (Wan 2.2 14B T2V, native ComfyUI nodes — workflow verified live) ---
LOCAL_VIDEO_MODELS = {
    # alias -> spec. `requires` lists the model files a box must have to serve this (used for
    # capability auto-detection in the gen-pool, so a box only gets video jobs it can run).
    "wan": {
        "capability": "video-wan",
        "high": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "low": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "shift": 8.0, "cfg": 3.5, "steps": 20, "fps": 16,
    },
}
WAN_NEG_DEFAULT = "blurry, low quality, static, distorted, watermark, text, jpeg artifacts"
# aspect -> (w,h) at ~480p base (Wan upscales well; keep base modest for speed)
VIDEO_AR = {"1:1": (512, 512), "16:9": (832, 480), "9:16": (480, 832), "4:3": (640, 480), "3:4": (480, 640)}


def _wan_t2v_workflow(spec, pos, neg, width, height, length, steps, seed, fps):
    """Wan 2.2 two-expert (high→low noise) T2V graph. Mirrors the live-validated workflow."""
    half = max(1, steps // 2)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": spec["high"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": spec["shift"]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": spec["low"], "weight_dtype": "default"}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": spec["shift"]}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": spec["clip"], "type": "wan"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["5", 0]}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": spec["vae"]}},
        "9": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["8", 0], "width": width, "height": height, "length": length, "batch_size": 1}},
        "10": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["2", 0], "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": spec["cfg"], "sampler_name": "euler", "scheduler": "simple", "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["9", 0], "start_at_step": 0, "end_at_step": half, "return_with_leftover_noise": "enable"}},
        "11": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["4", 0], "add_noise": "disable", "noise_seed": seed, "steps": steps, "cfg": spec["cfg"], "sampler_name": "euler", "scheduler": "simple", "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["10", 0], "start_at_step": half, "end_at_step": steps, "return_with_leftover_noise": "disable"}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["8", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": float(fps)}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": "forge_video", "format": "mp4", "codec": "h264"}},
    }


async def comfy_has_models(client, comfy_url, unet_names, timeout=5.0) -> bool:
    """Capability auto-detection: does this box's ComfyUI have the given diffusion models?
    Queries the single-node schema (lightweight) and checks the unet_name enum. Lets the pool
    route a video job only to boxes that actually have the model — and a new box auto-qualifies
    the moment its models are synced, no config change."""
    try:
        d = (await client.get(f"{comfy_url.rstrip('/')}/object_info/UNETLoader", timeout=timeout)).json()
        opts = d.get("UNETLoader", {}).get("input", {}).get("required", {}).get("unet_name", [[]])[0]
        have = set(opts if isinstance(opts, list) else [])
        return all(n in have for n in unet_names)
    except (httpx.HTTPError, ValueError, KeyError):
        return False


async def comfy_reachable(client, comfy_url, timeout=4.0) -> bool:
    try:
        r = await client.get(f"{comfy_url.rstrip('/')}/system_stats", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def comfy_busy(client, comfy_url, timeout=4.0) -> bool:
    """True if ComfyUI already has a job running/queued — so a new gen should overflow to
    another box instead of waiting behind it. Unknown (probe fails) → False (don't block)."""
    try:
        d = (await client.get(f"{comfy_url.rstrip('/')}/queue", timeout=timeout)).json()
        return bool(d.get("queue_running") or d.get("queue_pending"))
    except (httpx.HTTPError, ValueError):
        return False


async def box_gaming(client, presence_url, timeout=3.0) -> bool:
    """True if the box hosting this ComfyUI is being used by a human (gaming/desktop) per its
    gpu-presence agent — so we must NOT send it work. Fail-OPEN: if presence is unset or the
    probe fails, return False (the box's own local guard still protects it)."""
    if not presence_url:
        return False
    try:
        d = (await client.get(presence_url.rstrip("/"), timeout=timeout)).json()
        return bool(d.get("present"))
    except (httpx.HTTPError, ValueError):
        return False


async def select_comfy(client, backends, require_unets=None) -> tuple[str | None, str]:
    """Pick a ComfyUI backend for a new local gen, honoring the routing policy:
    primary first, overflow only when primary is busy; never a box being gamed on.
    `backends` = ordered list of {url, presence_url, label} (highest priority first).
    `require_unets` = optional list of diffusion-model filenames the box must have (capability
    gate — e.g. the Wan video models, so video only routes to boxes that can run it; a box
    auto-qualifies once its models are synced). Returns (url, label) or (None, reason)."""
    reachable = []
    for b in backends:
        if not b.get("url"):
            continue
        if not await comfy_reachable(client, b["url"]):
            continue
        if await box_gaming(client, b.get("presence_url", "")):
            continue  # skip a box someone is gaming on
        if require_unets and not await comfy_has_models(client, b["url"], require_unets):
            continue  # box lacks the required model (e.g. no Wan video models yet)
        reachable.append(b)
    for b in reachable:
        if not await comfy_busy(client, b["url"]):
            return b["url"], b["label"]
    if reachable:
        return reachable[0]["url"], reachable[0]["label"]
    return None, "no reachable ComfyUI backend (all unreachable or gaming)"


async def call_comfy(client, comfy_url, prompt, *, model="pony", negative_prompt=None,
                     width=832, height=1216, steps=None, cfg=None, seed=None,
                     guidance=3.5, poll_seconds=240) -> list:
    """Generate one image on the local ComfyUI box. Returns [png_bytes]. Raises GenerationError
    (so the tool can fall back to cloud) on any failure."""
    if model not in LOCAL_MODELS:
        raise GenerationError(f"Unknown local model '{model}'. Use one of: {', '.join(LOCAL_MODELS)}")
    ckpt, family = LOCAL_MODELS[model]
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    base = comfy_url.rstrip("/")

    if family == "sdxl":
        pos = prompt if prompt.lower().startswith("score_") else PONY_POS_PREFIX + prompt
        wf = _sdxl_workflow(ckpt, pos, negative_prompt or PONY_NEG_DEFAULT,
                            width, height, steps or 28, cfg or 6.5, seed)
    else:  # flux
        wf = _flux_workflow(ckpt, prompt, width, height, steps or 20, seed, guidance)

    q = await fetch_json(client, f"{base}/prompt", json_body={"prompt": wf, "client_id": "forge"})
    pid = q.get("prompt_id")
    if not pid:
        raise GenerationError(f"ComfyUI rejected the workflow: {str(q)[:400]}")

    out = None
    for _ in range(poll_seconds):
        await asyncio.sleep(1.0)
        try:
            rec = (await client.get(f"{base}/history/{pid}")).json().get(pid)
        except (httpx.HTTPError, ValueError):
            continue
        if not rec:
            continue
        for node in (rec.get("outputs") or {}).values():
            if node.get("images"):
                out = node["images"][0]
                break
        if out:
            break
        st = rec.get("status") or {}
        if st.get("status_str") == "error":
            raise GenerationError(f"ComfyUI generation error: {str(st.get('messages') or st)[:400]}")
    if not out:
        raise GenerationError("ComfyUI generation timed out")

    r = await client.get(f"{base}/view", params={
        "filename": out["filename"], "subfolder": out.get("subfolder", ""), "type": out.get("type", "output")})
    if r.status_code != 200 or not r.content:
        raise GenerationError(f"ComfyUI image fetch failed: HTTP {r.status_code}")
    return [r.content]


async def call_comfy_video(client, comfy_url, prompt, *, model="wan", negative_prompt=None,
                           width=512, height=512, length=49, steps=None, seed=None, fps=None,
                           poll_seconds=900) -> bytes:
    """Generate a video on a local ComfyUI box (Wan 2.2 T2V). Returns mp4 bytes. Raises
    GenerationError on failure (so the tool can fall back to cloud Veo)."""
    spec = LOCAL_VIDEO_MODELS.get(model)
    if not spec:
        raise GenerationError(f"Unknown local video model '{model}'. Use: {', '.join(LOCAL_VIDEO_MODELS)}")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    base = comfy_url.rstrip("/")
    wf = _wan_t2v_workflow(spec, prompt, negative_prompt or WAN_NEG_DEFAULT,
                           width, height, length, steps or spec["steps"], seed, fps or spec["fps"])
    q = await fetch_json(client, f"{base}/prompt", json_body={"prompt": wf, "client_id": "forge"})
    pid = q.get("prompt_id")
    if not pid:
        raise GenerationError(f"ComfyUI rejected the video workflow: {str(q)[:400]}")
    out = None
    for _ in range(poll_seconds):
        await asyncio.sleep(1.0)
        try:
            rec = (await client.get(f"{base}/history/{pid}")).json().get(pid)
        except (httpx.HTTPError, ValueError):
            continue
        if not rec:
            continue
        for node in (rec.get("outputs") or {}).values():
            if node.get("images"):
                out = node["images"][0]; break
        if out:
            break
        st = rec.get("status") or {}
        if st.get("status_str") == "error":
            raise GenerationError(f"ComfyUI video error: {str(st.get('messages') or st)[:400]}")
    if not out:
        raise GenerationError("ComfyUI video generation timed out")
    r = await client.get(f"{base}/view", params={
        "filename": out["filename"], "subfolder": out.get("subfolder", ""), "type": out.get("type", "output")})
    if r.status_code != 200 or not r.content:
        raise GenerationError(f"ComfyUI video fetch failed: HTTP {r.status_code}")
    return r.content
