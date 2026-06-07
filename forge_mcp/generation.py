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


# ---- ElevenLabs TTS (cloud) ----
# Expressive text-to-speech + a big prebuilt voice library + instant cloning. Returns mp3 bytes
# (same raw-bytes contract as call_imagen). Needs ELEVENLABS_API_KEY in the service .env.

ELEVENLABS_API = "https://api.elevenlabs.io/v1"
DEFAULT_ELEVEN_MODEL = "eleven_multilingual_v2"      # high quality; eleven_turbo_v2_5 = low latency
DEFAULT_ELEVEN_VOICE = "21m00Tcm4TlvDq8ikWAM"        # "Rachel" — ElevenLabs' public default voice


_default_voice_cache = None


async def _default_eleven_voice(client, api_key) -> str:
    """Free-tier accounts can't use Voice-Library voices via the API, so default to the
    account's OWN first voice (cached) rather than a hardcoded library id like Rachel."""
    global _default_voice_cache
    if _default_voice_cache is None:
        try:
            voices = await list_elevenlabs_voices(client, api_key)
            _default_voice_cache = voices[0]["voice_id"] if voices else DEFAULT_ELEVEN_VOICE
        except GenerationError:
            return DEFAULT_ELEVEN_VOICE
    return _default_voice_cache


async def call_elevenlabs(client, api_key, text, *, voice_id=None, model_id=None,
                          stability=0.5, similarity_boost=0.75, style=0.0,
                          output_format="mp3_44100_128", max_retries=3, retry_delay=1.0) -> bytes:
    """Synthesize speech via ElevenLabs. Returns mp3 bytes. Raises GenerationError (readable)."""
    if not api_key:
        raise GenerationError("ELEVENLABS_API_KEY is not configured on the forge service")
    if not (text or "").strip():
        raise GenerationError("text is required")
    voice_id = voice_id or await _default_eleven_voice(client, api_key)
    model_id = model_id or DEFAULT_ELEVEN_MODEL
    url = f"{ELEVENLABS_API}/text-to-speech/{voice_id}"
    body = {"text": text, "model_id": model_id,
            "voice_settings": {"stability": stability, "similarity_boost": similarity_boost, "style": style}}
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, headers=headers, params={"output_format": output_format},
                                     json=body, timeout=120)
        except httpx.HTTPError as e:
            last_err = GenerationError(f"ElevenLabs request failed: {e}")
            await asyncio.sleep(retry_delay * (attempt + 1)); continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = GenerationError(f"ElevenLabs HTTP {resp.status_code}: {resp.text[:300]}")
            await asyncio.sleep(retry_delay * (attempt + 1)); continue
        if resp.status_code >= 400:
            raise GenerationError(f"ElevenLabs HTTP {resp.status_code}: {resp.text[:300]}")
        if not resp.content:
            raise GenerationError("ElevenLabs returned empty audio")
        return resp.content
    raise last_err or GenerationError("ElevenLabs failed after retries")


async def list_elevenlabs_voices(client, api_key) -> list:
    """List the account's ElevenLabs voices: [{voice_id, name, category}]."""
    if not api_key:
        raise GenerationError("ELEVENLABS_API_KEY is not configured on the forge service")
    try:
        r = await client.get(f"{ELEVENLABS_API}/voices", headers={"xi-api-key": api_key}, timeout=30)
    except httpx.HTTPError as e:
        raise GenerationError(f"ElevenLabs request failed: {e}") from e
    if r.status_code >= 400:
        raise GenerationError(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
    voices = (r.json() or {}).get("voices", [])
    return [{"voice_id": v.get("voice_id"), "name": v.get("name"), "category": v.get("category")}
            for v in voices]


# ---- Local Chatterbox TTS (isolated GPU container; see chatterbox_service/) ----
# Same primary→overflow, presence-yield routing as ComfyUI, but over the Chatterbox
# /tts HTTP endpoints. box_gaming() (defined below) is resolved at call time.

async def chatterbox_reachable(client, url, timeout=4.0) -> bool:
    if not url:
        return False
    try:
        r = await client.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def select_chatterbox(client, cfg) -> tuple[str | None, str]:
    """Pick a Chatterbox backend: primary unless its box is gamed-on/unreachable, then
    overflow. Reuses the ComfyUI presence endpoints (same boxes). (url, label) or (None, reason)."""
    candidates = [
        (cfg.chatterbox_url, cfg.comfy_presence_url, "laybackrig"),
        (cfg.chatterbox_overflow_url, cfg.comfy_overflow_presence_url, "maingamingrig"),
    ]
    reachable = []
    for url, presence, label in candidates:
        if not await chatterbox_reachable(client, url):
            continue
        if await box_gaming(client, presence):
            continue
        reachable.append((url, label))
    if reachable:
        return reachable[0]
    return None, "no reachable Chatterbox backend (unreachable, or the box is gaming)"


async def call_chatterbox(client, url, text, *, exaggeration=0.5, cfg_weight=0.5,
                          audio_prompt_bytes=None, timeout=300) -> bytes:
    """Synthesize speech on a local Chatterbox container. Returns wav bytes. Raises
    GenerationError. audio_prompt_bytes = a reference clip for zero-shot voice cloning."""
    body = {"text": text, "exaggeration": exaggeration, "cfg_weight": cfg_weight}
    if audio_prompt_bytes:
        body["audio_prompt_b64"] = base64.b64encode(audio_prompt_bytes).decode()
    try:
        r = await client.post(f"{url.rstrip('/')}/tts", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise GenerationError(f"Chatterbox request failed: {e}") from e
    if r.status_code >= 400:
        raise GenerationError(f"Chatterbox HTTP {r.status_code}: {r.text[:300]}")
    if not r.content:
        raise GenerationError("Chatterbox returned empty audio")
    return r.content


# ---- Local diffusion via ComfyUI (uncensored; runs on the 4090 box over the LAN) ----
# Workflows below are validated against the live ComfyUI (gemini-flash/Imagen still available
# as the cloud fallback when the box is unreachable). Each returns raw PNG bytes — same
# contract as call_imagen / call_gemini_image.

LOCAL_MODELS = {
    # Curated aliases -> (checkpoint filename, family, prompt_style). Friendly name + correct family +
    # the quality-tag dialect the checkpoint was trained on (drives build_sdxl_prompts so callers needn't
    # know it). `model` may ALSO be any checkpoint filename installed in ComfyUI (auto-discovered) — see
    # resolve_model; raw filenames default to the 'pony' dialect (most uncensored SDXL is Pony-derived).
    "pony":           ("ponyDiffusionV6XL.safetensors",      "sdxl", "pony"),         # fast, maximally uncensored (anime-lean)
    "flux":           ("flux1-dev-fp8.safetensors",          "flux", "plain"),        # best quality/coherence
    "illustrious":    ("Illustrious-XL-v0.1.safetensors",    "sdxl", "illustrious"),  # anime/illustration, strong characters
    "juggernaut":     ("Juggernaut-XL-v9.safetensors",       "sdxl", "plain"),        # general photoreal (people, products)
    "cyberrealistic": ("cyberrealisticPony_v18.safetensors", "sdxl", "pony"),         # PHOTOREAL NSFW — Pony-based, score-tag native
    "bigasp":         ("bigaspV2.safetensors",               "sdxl", "plain"),        # PHOTOREAL NSFW — from-scratch SDXL, natural-language/booru
}


def model_family(name: str) -> str:
    """Heuristic family for a raw checkpoint filename (flux vs sdxl). Used when a caller passes a
    checkpoint that isn't a curated alias, so any installed model works without a code change."""
    return "flux" if "flux" in name.lower() else "sdxl"


def resolve_model(model: str) -> tuple[str, str]:
    """(checkpoint_filename, family) for either a curated alias OR a raw installed checkpoint name.
    A bare alias (no extension) that isn't curated is rejected; pass the actual .safetensors filename
    (see list_models for what each box has) to use a non-aliased model."""
    if model in LOCAL_MODELS:
        ckpt, family, _style = LOCAL_MODELS[model]
        return ckpt, family
    if model.endswith((".safetensors", ".ckpt", ".sft")):
        return model, model_family(model)
    raise GenerationError(
        f"Unknown model '{model}'. Use a curated alias ({', '.join(LOCAL_MODELS)}) or an installed "
        f"checkpoint filename (e.g. Foo_v1.safetensors — see list_models).")


def resolve_style(model: str) -> str:
    """The quality-tag dialect to auto-apply for a model: 'pony' (score_9…), 'illustrious'
    (masterpiece/booru quality tags), or 'plain' (no auto prefix — for from-scratch realistic models
    like bigasp/juggernaut, where score tags only hurt). Raw (non-aliased) checkpoints default to
    'pony', since most uncensored community SDXL checkpoints are Pony-derived and expect score tags."""
    if model in LOCAL_MODELS:
        return LOCAL_MODELS[model][2]
    return "pony"


# Each SDXL checkpoint family expects a different quality-tag dialect. We auto-apply the right one per
# model (see resolve_style / build_sdxl_prompts) so callers needn't know it. An explicit negative_prompt
# always wins, and a positive prompt that already opens with the dialect's signature tag is left as-is.
PONY_POS_PREFIX = "score_9, score_8_up, score_7_up, "
PONY_NEG_DEFAULT = ("score_6, score_5, score_4, worst quality, low quality, blurry, "
                    "jpeg artifacts, text, watermark, signature, deformed, extra limbs")
# Illustrious / NoobAI (anime) use booru quality tags, NOT pony score tags.
ILLUSTRIOUS_POS_PREFIX = "masterpiece, best quality, newest, absurdres, highres, "
ILLUSTRIOUS_NEG_DEFAULT = ("worst quality, low quality, lowres, bad anatomy, bad hands, missing fingers, "
                           "extra digit, jpeg artifacts, signature, watermark, text, blurry")
# From-scratch realistic models (bigasp, juggernaut): no magic prefix — they respond to plain
# natural-language / booru prompts; score tags only pollute them.
GENERIC_NEG_DEFAULT = ("worst quality, low quality, blurry, jpeg artifacts, text, watermark, signature, "
                       "deformed, bad anatomy, bad hands, extra limbs, mutated")


def build_sdxl_prompts(model: str, prompt: str, negative_prompt: str | None) -> tuple[str, str]:
    """Apply the model's quality-tag dialect to (prompt, negative_prompt) and return (positive, negative).
    A caller-supplied negative_prompt is used verbatim; otherwise the dialect's default applies. If the
    positive prompt already starts with the dialect's signature tag, it's left as-is (no double prefix)."""
    style = resolve_style(model)
    p = prompt.strip()
    low = p.lower()
    if style == "illustrious":
        pos = p if low.startswith(("masterpiece", "best quality", "score_")) else ILLUSTRIOUS_POS_PREFIX + p
        return pos, (negative_prompt or ILLUSTRIOUS_NEG_DEFAULT)
    if style == "plain":
        return p, (negative_prompt or GENERIC_NEG_DEFAULT)
    # default dialect: pony (score tags)
    pos = p if low.startswith("score_") else PONY_POS_PREFIX + p
    return pos, (negative_prompt or PONY_NEG_DEFAULT)


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
    "wan-i2v": {  # image-to-video (animate a still) — separate I2V-trained experts
        "capability": "video-wan-i2v",
        "high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
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


async def comfy_checkpoints(client, comfy_url, timeout=5.0) -> list:
    """Live list of checkpoint files installed on a box's ComfyUI (CheckpointLoaderSimple enum).
    This is what makes the model registry self-updating: drop a .safetensors in models/checkpoints,
    sync it, and it shows up here — no code change."""
    try:
        d = (await client.get(f"{comfy_url.rstrip('/')}/object_info/CheckpointLoaderSimple", timeout=timeout)).json()
        opts = d.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
        return list(opts) if isinstance(opts, list) else []
    except (httpx.HTTPError, ValueError, KeyError):
        return []


async def comfy_has_checkpoints(client, comfy_url, ckpt_names, timeout=5.0) -> bool:
    have = set(await comfy_checkpoints(client, comfy_url, timeout))
    return all(n in have for n in ckpt_names)


async def comfy_free(client, comfy_url, timeout=8.0) -> None:
    """Unload models + free VRAM on a box's ComfyUI after a gen, so chim/games get the card back
    immediately (ComfyUI keeps the last model resident otherwise). Best-effort — never raises."""
    try:
        await client.post(f"{comfy_url.rstrip('/')}/free",
                          json={"unload_models": True, "free_memory": True}, timeout=timeout)
    except httpx.HTTPError:
        pass


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


# GPU util (from a box's :11435 status/presence endpoint) above this = something is generating
# (e.g. chim's NPC dialogue on laybackrig) → treat the box as busy so Forge yields it.
GAMING_UTIL_PCT = 50


async def box_gaming(client, presence_url, timeout=3.0) -> bool:
    """True if the box hosting this ComfyUI is in use and must be yielded — either its presence
    agent says a human is on it (`present`, e.g. gaming on maingamingrig) OR its GPU util is high
    (something like chim is generating on laybackrig, whose :11435 reports util but no `present`).
    Fail-OPEN: unset/unreachable → False (the box's own local guard still protects it)."""
    if not presence_url:
        return False
    try:
        d = (await client.get(presence_url.rstrip("/"), timeout=timeout)).json()
        if bool(d.get("present")):
            return True
        u = d.get("util")
        return isinstance(u, (int, float)) and u >= GAMING_UTIL_PCT
    except (httpx.HTTPError, ValueError):
        return False


async def select_comfy(client, backends, require_unets=None, require_checkpoints=None) -> tuple[str | None, str]:
    """Pick a ComfyUI backend for a new local gen, honoring the routing policy:
    primary first, overflow only when primary is busy; never a box being gamed on.
    `backends` = ordered list of {url, presence_url, label} (highest priority first).
    `require_unets` / `require_checkpoints` = optional model filenames the box must have (capability
    gate — video unets, or a specific checkpoint — so a job only routes to a box that can run it; a
    box auto-qualifies once its models are synced). Returns (url, label) or (None, reason)."""
    elig = await eligible_comfy_backends(client, backends, require_unets, require_checkpoints)
    for b in elig:
        if not b["busy"]:
            return b["url"], b["label"]
    if elig:
        return elig[0]["url"], elig[0]["label"]
    return None, "no reachable ComfyUI backend (all unreachable, gaming, or lacking the model)"


async def eligible_comfy_backends(client, backends, require_unets=None, require_checkpoints=None):
    """All usable ComfyUI backends in priority order (reachable, not gamed-on/chim-busy, and
    having the required models), each annotated with `busy` (a job already running). Shared by
    select_comfy (picks the best one) and the batch fan-out (spreads work across several)."""
    out = []
    for b in backends:
        if not b.get("url"):
            continue
        if not await comfy_reachable(client, b["url"]):
            continue
        if await box_gaming(client, b.get("presence_url", "")):
            continue
        if require_unets and not await comfy_has_models(client, b["url"], require_unets):
            continue
        if require_checkpoints and not await comfy_has_checkpoints(client, b["url"], require_checkpoints):
            continue
        out.append({**b, "busy": await comfy_busy(client, b["url"])})
    return out


async def call_comfy(client, comfy_url, prompt, *, model="pony", negative_prompt=None,
                     width=832, height=1216, steps=None, cfg=None, seed=None,
                     guidance=3.5, poll_seconds=240, free_after=True) -> list:
    """Generate one image on the local ComfyUI box. Returns [png_bytes]. Raises GenerationError
    (so the tool can fall back to cloud) on any failure."""
    ckpt, family = resolve_model(model)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    base = comfy_url.rstrip("/")

    if family == "sdxl":
        pos, neg = build_sdxl_prompts(model, prompt, negative_prompt)
        wf = _sdxl_workflow(ckpt, pos, neg, width, height, steps or 28, cfg or 6.5, seed)
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
    if free_after:
        await comfy_free(client, base)  # hand the card back to chim/games after the gen
    return [r.content]


def _sdxl_ipadapter_workflow(ckpt, pos, neg, width, height, steps, cfg, seed, ref_names, preset, weight, weight_type):
    """SDXL graph with IPAdapter injected between the checkpoint MODEL and the KSampler, so the gen
    carries the character/face/style of the reference(s). Reuses the plain SDXL graph + repoints
    node 3's model input through IPAdapterUnifiedLoader → IPAdapterAdvanced. Multiple references are
    batched (ImageBatch) and averaged (combine_embeds='average') for a stronger, more robust likeness."""
    wf = _sdxl_workflow(ckpt, pos, neg, width, height, steps, cfg, seed)
    load_ids = []
    for i, name in enumerate(ref_names):
        nid = str(30 + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        load_ids.append(nid)
    image_ref = [load_ids[0], 0]
    for i in range(1, len(load_ids)):  # chain ImageBatch(prev, next) into one batched IMAGE
        bid = str(40 + i)
        wf[bid] = {"class_type": "ImageBatch", "inputs": {"image1": image_ref, "image2": [load_ids[i], 0]}}
        image_ref = [bid, 0]
    combine = "average" if len(load_ids) > 1 else "concat"
    wf["21"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["4", 0], "preset": preset}}
    wf["22"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["21", 0], "ipadapter": ["21", 1], "image": image_ref, "weight": weight,
        "weight_type": weight_type, "combine_embeds": combine, "start_at": 0.0, "end_at": 1.0,
        "embeds_scaling": "V only"}}
    wf["3"]["inputs"]["model"] = ["22", 0]
    return wf


async def call_comfy_ref(client, comfy_url, ref_images, prompt, *, model="pony",
                         preset="PLUS (high strength)", weight=0.8, weight_type="linear",
                         negative_prompt=None, width=832, height=1216, steps=None, cfg=None,
                         seed=None, poll_seconds=240, free_after=True) -> list:
    """Generate an image conditioned on REFERENCE image(s) via IPAdapter (character/face/style
    consistency). ref_images: bytes or a list of bytes (multiple angles → averaged likeness).
    SDXL/pony family only. Returns [png_bytes]. Raises GenerationError on failure."""
    ckpt, family = resolve_model(model)
    if family != "sdxl":
        raise GenerationError("Reference (IPAdapter) generation supports the SDXL family (pony/cyberrealistic/illustrious/juggernaut/bigasp) only")
    if isinstance(ref_images, (bytes, bytearray)):
        ref_images = [ref_images]
    if not ref_images:
        raise GenerationError("call_comfy_ref needs at least one reference image")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    base = comfy_url.rstrip("/")
    pos, neg = build_sdxl_prompts(model, prompt, negative_prompt)
    names = [await comfy_upload_image(client, base, b, f"forge_ref_{i}.png") for i, b in enumerate(ref_images)]
    wf = _sdxl_ipadapter_workflow(ckpt, pos, neg, width, height,
                                  steps or 28, cfg or 6.5, seed, names, preset, weight, weight_type)
    q = await fetch_json(client, f"{base}/prompt", json_body={"prompt": wf, "client_id": "forge"})
    pid = q.get("prompt_id")
    if not pid:
        raise GenerationError(f"ComfyUI rejected the IPAdapter workflow: {str(q)[:400]}")
    data = await _comfy_poll_image(client, base, pid, poll_seconds, "reference-gen")
    if free_after:
        await comfy_free(client, base)
    return [data]


async def comfy_upload_image(client, comfy_url, data, filename="forge_input.png") -> str:
    """Upload image bytes into a ComfyUI box's input dir; returns the server-side name for LoadImage."""
    r = await client.post(f"{comfy_url.rstrip('/')}/upload/image",
                          files={"image": (filename, data, "image/png")},
                          data={"type": "input", "overwrite": "true"}, timeout=30)
    if r.status_code != 200:
        raise GenerationError(f"ComfyUI image upload failed: HTTP {r.status_code}")
    return r.json().get("name", filename)


def _upscale_workflow(image_name, model_name="RealESRGAN_x4plus.pth"):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "forge_upscale", "images": ["3", 0]}},
    }


async def _comfy_poll_image(client, base, pid, poll_seconds, what) -> bytes:
    """Shared submit-result poll: wait for prompt `pid`'s first image output, fetch its bytes."""
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
            raise GenerationError(f"ComfyUI {what} error: {str(st.get('messages') or st)[:300]}")
    if not out:
        raise GenerationError(f"ComfyUI {what} timed out")
    r = await client.get(f"{base}/view", params={
        "filename": out["filename"], "subfolder": out.get("subfolder", ""), "type": out.get("type", "output")})
    if r.status_code != 200 or not r.content:
        raise GenerationError(f"ComfyUI {what} fetch failed: HTTP {r.status_code}")
    return r.content


async def call_comfy_upscale(client, comfy_url, image_bytes, *, model_name="RealESRGAN_x4plus.pth",
                             poll_seconds=180, free_after=True) -> bytes:
    """Upscale an image ~4x on a local ComfyUI box (ESRGAN). Returns png bytes. Raises GenerationError."""
    base = comfy_url.rstrip("/")
    name = await comfy_upload_image(client, base, image_bytes)
    q = await fetch_json(client, f"{base}/prompt", json_body={"prompt": _upscale_workflow(name, model_name), "client_id": "forge"})
    pid = q.get("prompt_id")
    if not pid:
        raise GenerationError(f"ComfyUI rejected upscale workflow: {str(q)[:300]}")
    data = await _comfy_poll_image(client, base, pid, poll_seconds, "upscale")
    if free_after:
        await comfy_free(client, base)
    return data


async def call_comfy_video(client, comfy_url, prompt, *, model="wan", negative_prompt=None,
                           width=512, height=512, length=49, steps=None, seed=None, fps=None,
                           poll_seconds=900, free_after=True) -> bytes:
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
    if free_after:
        await comfy_free(client, base)  # hand the card back to chim/games after the gen
    return r.content


def _wan_i2v_workflow(spec, pos, neg, width, height, length, steps, seed, fps, image_name):
    """Wan 2.2 two-expert I2V-A14B graph. Unlike the 5B Wan22ImageToVideoLatent path, the A14B I2V
    models condition on a CLIP-Vision encode of the start frame: CLIPVisionEncode → WanImageToVideo,
    which rewrites positive/negative conditioning AND builds the (wan_2.1 VAE, /8) latent at the right
    scale — so the KSamplers take their conditioning + latent from node 9's three outputs."""
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
        "15": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "17": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": spec["clip_vision"]}},
        "18": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["17", 0], "image": ["15", 0], "crop": "center"}},
        "9": {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0], "clip_vision_output": ["18", 0],
            "start_image": ["15", 0], "width": width, "height": height, "length": length, "batch_size": 1}},
        "10": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["2", 0], "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": spec["cfg"], "sampler_name": "euler", "scheduler": "simple", "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["9", 2], "start_at_step": 0, "end_at_step": half, "return_with_leftover_noise": "enable"}},
        "11": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["4", 0], "add_noise": "disable", "noise_seed": seed, "steps": steps, "cfg": spec["cfg"], "sampler_name": "euler", "scheduler": "simple", "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["10", 0], "start_at_step": half, "end_at_step": steps, "return_with_leftover_noise": "disable"}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["8", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": float(fps)}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": "forge_i2v", "format": "mp4", "codec": "h264"}},
    }


async def call_comfy_i2v(client, comfy_url, image_bytes, prompt, *, negative_prompt=None,
                         width=832, height=480, length=49, steps=None, seed=None, fps=None,
                         poll_seconds=900, free_after=True) -> bytes:
    """Animate a still into a video on a local ComfyUI box (Wan 2.2 I2V). image_bytes = the start
    frame. Returns mp4 bytes. Raises GenerationError on failure."""
    spec = LOCAL_VIDEO_MODELS["wan-i2v"]
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    base = comfy_url.rstrip("/")
    name = await comfy_upload_image(client, base, image_bytes, "forge_i2v_start.png")
    wf = _wan_i2v_workflow(spec, prompt, negative_prompt or WAN_NEG_DEFAULT,
                           width, height, length, steps or spec["steps"], seed, fps or spec["fps"], name)
    q = await fetch_json(client, f"{base}/prompt", json_body={"prompt": wf, "client_id": "forge"})
    pid = q.get("prompt_id")
    if not pid:
        raise GenerationError(f"ComfyUI rejected the I2V workflow: {str(q)[:400]}")
    data = await _comfy_poll_image(client, base, pid, poll_seconds, "i2v")
    if free_after:
        await comfy_free(client, base)
    return data


# --- Local instruction EDITING (Flux Kontext dev, native ComfyUI) ---
FLUX_KONTEXT = {
    "unet": "flux1-dev-kontext_fp8_scaled.safetensors",   # diffusion_models/
    "clip_l": "clip_l.safetensors",                        # text_encoders/
    "t5": "t5xxl_fp8_e4m3fn.safetensors",                  # text_encoders/
    "vae": "ae.safetensors",                               # vae/ (flux autoencoder)
    "guidance": 2.5, "steps": 20,
}


def _flux_kontext_workflow(spec, instruction, image_name, steps, seed, guidance):
    """Flux Kontext edit graph: VAE-encode the input image, attach it as a ReferenceLatent to the
    instruction conditioning, and denoise — so the model edits the image per the text instruction.
    Mirrors the native ComfyUI Kontext template (DualCLIPLoader flux + FluxKontextImageScale +
    ReferenceLatent + FluxGuidance)."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": spec["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": spec["clip_l"], "clip_name2": spec["t5"], "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": spec["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": instruction, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["6", 0]}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["4", 0], "latent": ["8", 0]}},
        "10": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["9", 0], "guidance": guidance}},
        "11": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["8", 0],
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "forge_kontext", "images": ["12", 0]}},
    }


async def call_comfy_kontext(client, comfy_url, image_bytes, instruction, *, steps=None, seed=None,
                             guidance=None, poll_seconds=300, free_after=True) -> bytes:
    """Edit an image by text instruction on a local ComfyUI box (Flux Kontext). Returns png bytes.
    Raises GenerationError on failure."""
    spec = FLUX_KONTEXT
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    base = comfy_url.rstrip("/")
    name = await comfy_upload_image(client, base, image_bytes, "forge_kontext_src.png")
    wf = _flux_kontext_workflow(spec, instruction, name, steps or spec["steps"], seed, guidance or spec["guidance"])
    q = await fetch_json(client, f"{base}/prompt", json_body={"prompt": wf, "client_id": "forge"})
    pid = q.get("prompt_id")
    if not pid:
        raise GenerationError(f"ComfyUI rejected the Kontext workflow: {str(q)[:400]}")
    data = await _comfy_poll_image(client, base, pid, poll_seconds, "kontext")
    if free_after:
        await comfy_free(client, base)
    return data
