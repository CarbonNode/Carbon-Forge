import os
from dataclasses import dataclass


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
    workspace_root: str
    workspace_display_root: str
    results_root: str
    public_url: str
    cache_ttl_days: int
    max_image_mb: int
    max_video_mb: int


def load_config() -> Config:
    return Config(
        host=os.environ.get("FORGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("FORGE_PORT", "5125")),
        token=os.environ.get("FORGE_TOKEN", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        comfy_url=os.environ.get("FORGE_COMFY_URL", "").rstrip("/"),
        comfy_presence_url=os.environ.get("FORGE_COMFY_PRESENCE_URL", "http://host.docker.internal:11435").rstrip("/"),
        comfy_overflow_url=os.environ.get("FORGE_COMFY_OVERFLOW_URL", "").rstrip("/"),
        comfy_overflow_presence_url=os.environ.get("FORGE_COMFY_OVERFLOW_PRESENCE_URL", "").rstrip("/"),
        workspace_root=os.environ.get("FORGE_WORKSPACE_ROOT", "/workspace"),
        workspace_display_root=os.environ.get("FORGE_WORKSPACE_DISPLAY_ROOT", "C:\\Workspace"),
        results_root=os.environ.get("FORGE_RESULTS_ROOT", "/results"),
        public_url=os.environ.get("FORGE_PUBLIC_URL", "https://forge.carbonrouting.dev").rstrip("/"),
        cache_ttl_days=int(os.environ.get("FORGE_CACHE_TTL_DAYS", "30")),
        max_image_mb=int(os.environ.get("FORGE_MAX_IMAGE_MB", "50")),
        max_video_mb=int(os.environ.get("FORGE_MAX_VIDEO_MB", "500")),
    )
