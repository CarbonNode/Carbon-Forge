import os
from forge_mcp.config import load_config


def test_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FORGE_") or k == "GEMINI_API_KEY":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FORGE_TOKEN", "tok")
    cfg = load_config()
    assert cfg.port == 5125
    assert cfg.host == "0.0.0.0"
    assert cfg.workspace_root == "/workspace"
    assert cfg.results_root == "/results"
    assert cfg.public_url == "https://forge.carbonrouting.dev"
    assert cfg.cache_ttl_days == 30
    assert cfg.max_image_mb == 50
    assert cfg.max_video_mb == 500
    assert cfg.workspace_display_root == "C:\\Workspace"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("FORGE_TOKEN", "tok")
    monkeypatch.setenv("FORGE_PORT", "6000")
    monkeypatch.setenv("FORGE_WORKSPACE_ROOT", "C:/tmp/ws")
    monkeypatch.setenv("FORGE_PUBLIC_URL", "http://localhost:5125/")
    cfg = load_config()
    assert cfg.port == 6000
    assert cfg.workspace_root == "C:/tmp/ws"
    assert cfg.public_url == "http://localhost:5125"  # trailing slash stripped
