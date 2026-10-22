"""VoxCPM2 HTTP adapter. The orchestrator never imports torch or holds GPU memory."""
import base64
import math

import httpx

from forge_mcp.generation import GenerationError

MODEL = "openbmb/VoxCPM2"


def validate(text, model=None, voice_description=None, prompt_text=None, voice=None,
             cfg_value=2.0, inference_timesteps=10, seed=None):
    if not text.strip() or len(text) > 2000:
        raise GenerationError("VoxCPM text must contain 1–2000 characters")
    if model not in (None, "voxcpm", "voxcpm2", MODEL):
        raise GenerationError(f"VoxCPM model must be {MODEL}")
    if voice_description and (len(voice_description) > 500 or any(c in voice_description for c in "()\n\r")):
        raise GenerationError("Voice description must be at most 500 characters without parentheses or newlines")
    if prompt_text and (not voice or not prompt_text.strip() or len(prompt_text) > 2000):
        raise GenerationError("Reference transcript needs a reference clip and at most 2000 characters")
    if not math.isfinite(cfg_value) or not 0.5 <= cfg_value <= 5:
        raise GenerationError("VoxCPM guidance must be between 0.5 and 5")
    if isinstance(inference_timesteps, bool) or not isinstance(inference_timesteps, int) or not 1 <= inference_timesteps <= 50:
        raise GenerationError("VoxCPM inference steps must be an integer between 1 and 50")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647):
        raise GenerationError("VoxCPM seed must be an integer between 0 and 2147483647")


async def status(client, cfg):
    if not cfg.voxcpm_url:
        return {"available": False, "reason": "VoxCPM worker is not configured"}
    try:
        r = await client.get(f"{cfg.voxcpm_url}/health", timeout=4)
        r.raise_for_status()
        data = r.json()
        return {"available": data.get("ready") is True, "reason": data.get("reason"),
                "model": data.get("model"), "sample_rate": 48000}
    except (httpx.HTTPError, ValueError):
        return {"available": False, "reason": "VoxCPM worker is not reachable"}


async def synthesize(client, cfg, text, *, reference=None, voice_description=None,
                     prompt_text=None, cfg_value=2.0, inference_timesteps=10, seed=None):
    if not cfg.voxcpm_url or not cfg.voxcpm_token:
        raise GenerationError("VoxCPM worker URL/token is not configured on Forge")
    body = {"text": text, "voice_description": voice_description, "prompt_text": prompt_text,
            "cfg_value": cfg_value, "inference_timesteps": inference_timesteps, "seed": seed}
    if reference:
        if len(reference) > 10 * 1024 * 1024:
            raise GenerationError("VoxCPM reference clip must be under 10 MB")
        body["audio_prompt_b64"] = base64.b64encode(reference).decode("ascii")
    try:
        r = await client.post(f"{cfg.voxcpm_url}/tts", json=body,
                              headers={"Authorization": f"Bearer {cfg.voxcpm_token}"}, timeout=100)
    except httpx.HTTPError as e:
        raise GenerationError("VoxCPM request failed; no automatic retry was made") from e
    if r.status_code >= 400:
        raise GenerationError(f"VoxCPM HTTP {r.status_code}: {r.text[:300]}")
    if not (r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"):
        raise GenerationError("VoxCPM returned invalid WAV audio")
    return r.content
