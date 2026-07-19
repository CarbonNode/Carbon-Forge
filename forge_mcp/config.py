import os
from dataclasses import dataclass


def _gemini_keys() -> tuple:
    """Ordered, de-duplicated non-empty Gemini keys: GEMINI_API_KEY (may be a
    comma-separated list) then GEMINI_API_KEY_BACKUP. call_imagen / call_gemini_image
    rotate to the next key ONLY on a quota/429 depletion, so a spent key fails over."""
    raw = [p.strip() for p in (os.environ.get("GEMINI_API_KEY", "") or "").split(",")]
    raw.append((os.environ.get("GEMINI_API_KEY_BACKUP", "") or "").strip())
    seen, out = set(), []
    for k in raw:
        if k and k not in seen:
            seen.add(k); out.append(k)
    return tuple(out)


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    token: str
    gemini_api_key: str
    elevenlabs_api_key: str            # cloud TTS (generate_speech provider='elevenlabs')
    comfy_url: str
    comfy_presence_url: str            # primary box (laybackrig) gpu-status (:11435) — yield it to chim (high GPU util)
    comfy_overflow_url: str            # 2nd ComfyUI (e.g. maingamingrig) for overflow when primary is busy/gaming
    comfy_overflow_presence_url: str   # its gpu-presence endpoint (:11435) — skip overflow if that box is being gamed on
    chatterbox_url: str                # local Chatterbox TTS container (this box, :5126)
    chatterbox_overflow_url: str       # 2nd box's Chatterbox (maingamingrig) for overflow
    workspace_root: str
    workspace_display_root: str
    results_root: str
    public_url: str
    cache_ttl_days: int
    max_image_mb: int
    max_video_mb: int
    gemini_api_keys: tuple = ()


def load_config() -> Config:
    gkeys = _gemini_keys()
    return Config(
        host=os.environ.get("FORGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("FORGE_PORT", "5125")),
        token=os.environ.get("FORGE_TOKEN", ""),
        gemini_api_key=gkeys[0] if gkeys else "",
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        comfy_url=os.environ.get("FORGE_COMFY_URL", "").rstrip("/"),
        comfy_presence_url=os.environ.get("FORGE_COMFY_PRESENCE_URL", "http://host.docker.internal:11435").rstrip("/"),
        comfy_overflow_url=os.environ.get("FORGE_COMFY_OVERFLOW_URL", "").rstrip("/"),
        comfy_overflow_presence_url=os.environ.get("FORGE_COMFY_OVERFLOW_PRESENCE_URL", "").rstrip("/"),
        chatterbox_url=os.environ.get("FORGE_CHATTERBOX_URL", "http://host.docker.internal:5126").rstrip("/"),
        chatterbox_overflow_url=os.environ.get("FORGE_CHATTERBOX_OVERFLOW_URL", "").rstrip("/"),
        workspace_root=os.environ.get("FORGE_WORKSPACE_ROOT", "/workspace"),
        workspace_display_root=os.environ.get("FORGE_WORKSPACE_DISPLAY_ROOT", "C:\\Workspace"),
        results_root=os.environ.get("FORGE_RESULTS_ROOT", "/results"),
        public_url=os.environ.get("FORGE_PUBLIC_URL", "https://forge.carbonrouting.dev").rstrip("/"),
        cache_ttl_days=int(os.environ.get("FORGE_CACHE_TTL_DAYS", "30")),
        max_image_mb=int(os.environ.get("FORGE_MAX_IMAGE_MB", "50")),
        max_video_mb=int(os.environ.get("FORGE_MAX_VIDEO_MB", "500")),
        gemini_api_keys=gkeys,
    )
