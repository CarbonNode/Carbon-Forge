# Carbon Forge MCP Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Host Carbon Forge as a Docker MCP server on laybackrig so any Claude session (via the Carbon Cortex gateway) can generate/edit images, generate video, and run the full processing suite, with results written into Conduit project folders on carbonserver.

**Architecture:** One Python container (FastMCP streamable-HTTP, port 5125) on laybackrig containing the rembg/LaMa engine (shared with the desktop app via a `backend/processing.py` refactor), a port of `main.js`'s Imagen/Gemini/Veo client, and ffmpeg video ops. Outputs go to a CIFS mount of `\\192.168.0.35\Workspace` + a local results cache served at `/files/<id>/<name>` (public via `forge.carbonrouting.dev` ingress on the existing cloudflared-gateway tunnel). The gateway proxies it as an `mcp-proxy` connector named **`forge`**.

**Tech Stack:** Python 3.12, `mcp` SDK (FastMCP, v1.x), uvicorn, httpx, rembg, onnxruntime (CPU), Pillow, scipy, ffmpeg, pytest. Gateway side: existing `McpProxyConnector`, `register-*.mjs` pattern, React console.

**Spec:** `docs/superpowers/specs/2026-06-03-carbon-forge-mcp-design.md`

**Deliberate deviations from spec (cosmetic/safety):**
1. Service package is `forge_mcp/`, NOT `mcp/` — a top-level `mcp/` directory would shadow the pip `mcp` package via namespace-package resolution.
2. Connector is named `forge` (tools surface as `forge__generate_image`), not `carbon-forge` — matches the short-name convention of `memory`/`tunebox`/`questbook` and keeps tool names ergonomic.
3. New `media` tool category is **created with permission grants** in the register script (root cause of the documented "media had 0 tools" failure was missing grants, not the category name).

**Execution notes:**
- Repos: service code in `C:\Programming\Carbon Forge` (CarbonNode/Carbon-Forge); gateway integration in `C:\Programming\Carbon Cortex` (CarbonNode/Carbon-Cortex).
- All multi-step remote work: ONE self-contained script per the shell-discipline rules. Docker builds on laybackrig/carbonserver MUST use the one-shot scheduled-task pattern (credential-helper gotcha).
- Commit after every task (both repos follow commit-per-change discipline).

---

## Task 1: Extract `backend/processing.py` (shared engine, desktop unchanged)

**Files:**
- Create: `backend/processing.py`
- Modify: `backend/server.py`

- [ ] **Step 1.1: Create `backend/processing.py`**

Move these from `backend/server.py` **verbatim** (current line refs):
- imports needed by the moved code: `io`, `json` (not needed here), `os`, `sys`, `numpy as np`, `PIL.Image`, `scipy.ndimage` (`gaussian_filter`, `binary_dilation`, `binary_erosion`, `label`), `rembg` (`remove`, `new_session`), the `onnxruntime` try/import (lines 19–23)
- `sessions` dict, `DEFAULT_MODEL`, `AVAILABLE_MODELS` (lines 27–35)
- LaMa constants block (lines 37–45) — **change**: compute `LAMA_MODEL_DIR` with an env override first:

```python
_env_lama = os.environ.get("FORGE_LAMA_DIR")
if _env_lama:
    LAMA_MODEL_DIR = _env_lama
elif getattr(sys, 'frozen', False):
    LAMA_MODEL_DIR = os.path.join(os.path.dirname(sys.executable), "models")
else:
    LAMA_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
```

- `get_session` (56–61), `download_lama_model` (66–85), `get_lama_session` (88–103), `remove_watermark` (106–205), `remove_colors` (210–246), `smooth_edges` (249–286), `trim_transparent` (289–315), `detect_bg_color` (318–336), `split_sprites` (339–392)

Then ADD the shared pipeline (new code — this is the `/remove-bg` route body, lines 485–519, parameterized):

```python
from dataclasses import dataclass, field

@dataclass
class PipelineOptions:
    model: str = DEFAULT_MODEL
    skip_bg: bool = False
    alpha_matting: bool = False
    fg_threshold: int = 240
    bg_threshold: int = 10
    erode_size: int = 10
    color_remove: bool = False
    colors: list = field(default_factory=list)   # [(r,g,b), ...]
    color_auto_detect: bool = False
    color_tolerance: int = 20
    edge_smooth: bool = False
    edge_strength: int = 50
    edge_trim: int = 0
    auto_trim: bool = False
    watermark_remove: bool = False
    watermark_position: str = "bottom-right"
    watermark_size_pct: int = 15


def parse_colors(raw_colors):
    """Hex strings or RGB triples -> [(r,g,b)]. Invalid entries are skipped."""
    colors = []
    for c in raw_colors or []:
        try:
            if isinstance(c, str) and c.startswith("#"):
                h = c.lstrip("#")
                colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
            elif isinstance(c, (list, tuple)) and len(c) >= 3:
                colors.append((int(c[0]), int(c[1]), int(c[2])))
        except (TypeError, ValueError):
            continue
    return colors


def run_pipeline(data: bytes, opts: PipelineOptions) -> bytes:
    """The desktop /remove-bg pipeline: watermark -> bg removal -> colors -> edges -> trim."""
    colors = list(opts.colors)
    if opts.color_auto_detect or (opts.color_remove and not colors):
        colors = [detect_bg_color(data)]

    if opts.watermark_remove:
        ratio = opts.watermark_size_pct / 100
        data = remove_watermark(data, opts.watermark_position, ratio, ratio)

    if not opts.skip_bg:
        session = get_session(opts.model)
        kwargs = dict(session=session)
        if opts.alpha_matting:
            kwargs["alpha_matting"] = True
            kwargs["alpha_matting_foreground_threshold"] = opts.fg_threshold
            kwargs["alpha_matting_background_threshold"] = opts.bg_threshold
            kwargs["alpha_matting_erode_size"] = opts.erode_size
        result = remove(data, **kwargs)
    else:
        result = data

    if opts.color_remove and colors:
        result = remove_colors(result, colors, opts.color_tolerance)

    if opts.edge_smooth:
        result = smooth_edges(result, opts.edge_strength, opts.edge_trim)

    if opts.auto_trim:
        result = trim_transparent(result)

    return result


def run_split_pipeline(data: bytes, opts: PipelineOptions, min_area: int = 400) -> list:
    """The desktop /split-sprites pipeline. Returns list of PNG byte buffers."""
    bg_color = detect_bg_color(data)

    if opts.watermark_remove:
        ratio = opts.watermark_size_pct / 100
        data = remove_watermark(data, opts.watermark_position, ratio, ratio)

    session = get_session(opts.model)
    kwargs = dict(session=session)
    if opts.alpha_matting:
        kwargs["alpha_matting"] = True
        kwargs["alpha_matting_foreground_threshold"] = opts.fg_threshold
        kwargs["alpha_matting_background_threshold"] = opts.bg_threshold
        kwargs["alpha_matting_erode_size"] = opts.erode_size
    result = remove(data, **kwargs)

    all_colors = [bg_color] + list(opts.colors)
    result = remove_colors(result, all_colors, max(opts.color_tolerance, 25))

    smooth_str = opts.edge_strength if opts.edge_smooth else 60
    smooth_trim = opts.edge_trim if opts.edge_smooth else 2
    result = smooth_edges(result, smooth_str, smooth_trim)

    return split_sprites(result, min_area)
```

- [ ] **Step 1.2: Slim `backend/server.py`**

Delete the moved code; add `from processing import (...)` importing everything the routes use (`PipelineOptions`, `run_pipeline`, `run_split_pipeline`, `parse_colors`, `split_sprites`, `get_session`, `sessions`, `DEFAULT_MODEL`, `AVAILABLE_MODELS`, `LAMA_MODEL_PATH`, `lama_session` — note `lama_session` is module state; for `/watermark-status` import the module: `import processing` and read `processing.lama_session`). Keep: Flask app, CORS hook, startup preload (`sessions[DEFAULT_MODEL] = new_session(...)` becomes `get_session(DEFAULT_MODEL)` + `print("MODEL_READY", flush=True)`), error handler, all routes. Routes now parse headers into `PipelineOptions` and call the shared functions:

```python
def _opts_from_headers(req) -> PipelineOptions:
    return PipelineOptions(
        model=req.headers.get("X-Model", DEFAULT_MODEL),
        skip_bg=req.headers.get("X-Skip-Bg", "false") == "true",
        alpha_matting=req.headers.get("X-Alpha-Matting", "false") == "true",
        fg_threshold=int(req.headers.get("X-FG-Threshold", "240")),
        bg_threshold=int(req.headers.get("X-BG-Threshold", "10")),
        erode_size=int(req.headers.get("X-Erode-Size", "10")),
        color_remove=req.headers.get("X-Color-Remove", "false") == "true",
        colors=parse_colors(_safe_json(req.headers.get("X-Colors", "[]"))),
        color_auto_detect=req.headers.get("X-Color-Auto-Detect", "false") == "true",
        color_tolerance=int(req.headers.get("X-Color-Tolerance", "20")),
        edge_smooth=req.headers.get("X-Edge-Smooth", "false") == "true",
        edge_strength=int(req.headers.get("X-Edge-Strength", "50")),
        edge_trim=int(req.headers.get("X-Edge-Trim", "0")),
        auto_trim=req.headers.get("X-Auto-Trim", "false") == "true",
        watermark_remove=req.headers.get("X-Watermark-Remove", "false") == "true",
        watermark_position=req.headers.get("X-Watermark-Position", "bottom-right"),
        watermark_size_pct=int(req.headers.get("X-Watermark-Size", "15")),
    )

def _safe_json(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
```

`/remove-bg` body becomes: read data → `opts = _opts_from_headers(request)` → **preserve the existing color-auto-detect quirk**: the old route auto-detected when `color_auto_detect or not colors` regardless of `color_remove`; `run_pipeline` reproduces this only when `color_remove` is on. To keep byte-identical desktop behavior set `opts.color_auto_detect = opts.color_auto_detect or not opts.colors` before calling → `return Response(run_pipeline(data, opts), mimetype="image/png")`.
`/split-sprites` becomes: `sprites = run_split_pipeline(data, opts, min_area)` → same base64 JSON response. `/split-only` calls `split_sprites(data, min_area)` directly. `/models`, `/health`, `/watermark-status` unchanged (reading via `processing.` module attrs).

- [ ] **Step 1.3: Verify desktop backend unchanged (concrete evidence)**

```powershell
cd "C:\Programming\Carbon Forge"; python backend/server.py 5124
```
In a second call once `MODEL_READY` prints (use run_in_background for the server):
```powershell
Invoke-WebRequest http://127.0.0.1:5124/health   # expect {"status":"ok"}
Invoke-WebRequest http://127.0.0.1:5124/models   # expect 5-model JSON
# real processing call on a real PNG (the repo icon):
Invoke-WebRequest -Method POST -InFile "icon.ico" -ContentType application/octet-stream -Headers @{"X-Auto-Trim"="true"} http://127.0.0.1:5124/remove-bg -OutFile $env:TEMP\forge-test.png
# expect: HTTP 200, file is a PNG (check first bytes 89 50 4E 47)
```
Kill the server after.

- [ ] **Step 1.4: Commit**

```bash
git add backend/processing.py backend/server.py
git commit -m "refactor(backend): extract pure processing engine into processing.py (shared with MCP service)"
git push
```

---

## Task 2: `forge_mcp` skeleton — config, auth, health, FastMCP spike

**Files:**
- Create: `forge_mcp/__init__.py` (empty), `forge_mcp/config.py`, `forge_mcp/server.py`, `requirements-mcp.txt`, `tests/__init__.py` (empty), `tests/test_config.py`

- [ ] **Step 2.1: `requirements-mcp.txt`**

```
mcp>=1.12,<2
uvicorn>=0.30
httpx>=0.27
rembg==2.0.*
pillow>=10
numpy
scipy
onnxruntime>=1.17.0
```

Install locally for dev: `cd "C:\Programming\Carbon Forge"; pip install -r requirements-mcp.txt pytest`

- [ ] **Step 2.2: Write failing test `tests/test_config.py`**

```python
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
```

Run: `python -m pytest tests/test_config.py -v` → expect FAIL (module missing).

- [ ] **Step 2.3: `forge_mcp/config.py`**

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    token: str
    gemini_api_key: str
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
        workspace_root=os.environ.get("FORGE_WORKSPACE_ROOT", "/workspace"),
        workspace_display_root=os.environ.get("FORGE_WORKSPACE_DISPLAY_ROOT", "C:\\Workspace"),
        results_root=os.environ.get("FORGE_RESULTS_ROOT", "/results"),
        public_url=os.environ.get("FORGE_PUBLIC_URL", "https://forge.carbonrouting.dev").rstrip("/"),
        cache_ttl_days=int(os.environ.get("FORGE_CACHE_TTL_DAYS", "30")),
        max_image_mb=int(os.environ.get("FORGE_MAX_IMAGE_MB", "50")),
        max_video_mb=int(os.environ.get("FORGE_MAX_VIDEO_MB", "500")),
    )
```

Run: `python -m pytest tests/test_config.py -v` → expect PASS.

- [ ] **Step 2.4: `forge_mcp/server.py` — minimal app (no tools yet)**

```python
"""Carbon Forge MCP service — assembly + entry point."""
import hmac
import json
import os

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from forge_mcp.config import load_config

cfg = load_config()

mcp = FastMCP(
    "carbon-forge",
    host=cfg.host,
    port=cfg.port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


class BearerAuthMiddleware:
    """401 anything under /mcp without the right bearer token. /health and /files stay open."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            expected = f"Bearer {self.token}"
            supplied = headers.get("authorization", "")
            if not self.token or not hmac.compare_digest(supplied, expected):
                resp = JSONResponse({"error": "unauthorized"}, status_code=401)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def health(_request):
    return JSONResponse({"status": "ok", "service": "carbon-forge"})


async def serve_file(request):
    file_id = request.path_params["file_id"]
    name = request.path_params["name"]
    # ids are token_urlsafe — reject anything that could traverse
    if "/" in file_id or "\\" in file_id or ".." in file_id or ".." in name or "/" in name or "\\" in name:
        return JSONResponse({"error": "bad path"}, status_code=400)
    path = os.path.join(cfg.results_root, "files", file_id, name)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


def build_app():
    app = mcp.streamable_http_app()
    app.router.routes.append(Route("/health", health, methods=["GET"]))
    app.router.routes.append(Route("/files/{file_id}/{name}", serve_file, methods=["GET"]))
    return BearerAuthMiddleware(app, cfg.token)


def main():
    import uvicorn
    uvicorn.run(build_app(), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.5: Spike-verify the MCP plumbing locally (this step de-risks SDK API drift)**

```powershell
cd "C:\Programming\Carbon Forge"
$env:FORGE_TOKEN="testtok"; $env:FORGE_RESULTS_ROOT="$env:TEMP\forge-results"; python -m forge_mcp.server
```
(run_in_background). Then verify:
```powershell
Invoke-WebRequest http://127.0.0.1:5125/health                  # 200 {"status":"ok",...}
try { Invoke-WebRequest http://127.0.0.1:5125/mcp -Method POST } catch { $_.Exception.Response.StatusCode }  # expect 401
# MCP initialize with auth (expect 200 + JSON-RPC result naming "carbon-forge"):
$body = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}'
Invoke-WebRequest http://127.0.0.1:5125/mcp -Method POST -Body $body -ContentType "application/json" -Headers @{Authorization="Bearer testtok"; Accept="application/json, text/event-stream"}
# Also verify a non-local Host header isn't rejected (DNS-rebinding check):
curl.exe -s -X POST http://127.0.0.1:5125/mcp -H "Host: forge.carbonrouting.dev" -H "Authorization: Bearer testtok" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d $body
```
**If the Host-header request returns 421/400**: the installed SDK version enables DNS-rebinding protection — fix by constructing `TransportSecuritySettings(enable_dns_rebinding_protection=False)` from `mcp.server.transport_security` and passing `transport_security=...` to the `FastMCP(...)` constructor (1.x Settings field). Re-verify. Kill the server after.

- [ ] **Step 2.6: Commit**

```bash
git add forge_mcp/ requirements-mcp.txt tests/
git commit -m "feat(mcp): FastMCP service skeleton — config, bearer auth, /health, /files"
git push
```

---

## Task 3: `forge_mcp/storage.py` — input resolution, output writing, URL minting, janitor

**Files:**
- Create: `forge_mcp/storage.py`, `tests/test_storage.py`

- [ ] **Step 3.1: Write failing tests `tests/test_storage.py`**

```python
import asyncio
import os
import time
import pytest
from forge_mcp import storage
from forge_mcp.config import Config


def make_cfg(tmp_path):
    return Config(
        host="0.0.0.0", port=5125, token="t", gemini_api_key="",
        workspace_root=str(tmp_path / "ws"),
        workspace_display_root="C:\\Workspace",
        results_root=str(tmp_path / "results"),
        public_url="https://forge.example.com",
        cache_ttl_days=30, max_image_mb=50, max_video_mb=500,
    )


PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 64
JPG = bytes.fromhex("ffd8ffe0") + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def test_sniff():
    assert storage.sniff_mime(PNG) == "image/png"
    assert storage.sniff_mime(JPG) == "image/jpeg"
    assert storage.sniff_mime(MP4) == "video/mp4"
    assert storage.sniff_mime(b"plain text here") is None


def test_workspace_input_traversal_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    with pytest.raises(storage.StorageError, match="traversal|invalid"):
        asyncio.run(storage.resolve_input("Proj/../../etc/passwd", cfg=cfg))


def test_workspace_input_reads_file(tmp_path):
    cfg = make_cfg(tmp_path)
    d = os.path.join(cfg.workspace_root, "Proj", "img")
    os.makedirs(d)
    with open(os.path.join(d, "a.png"), "wb") as f:
        f.write(PNG)
    r = asyncio.run(storage.resolve_input("Proj/img/a.png", cfg=cfg))
    assert r.mime == "image/png" and r.data == PNG


def test_validate_project_close_match(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "PerennaMono"))
    with pytest.raises(storage.StorageError, match="PerennaMono"):
        storage.validate_project("perenamono", cfg=cfg)


def test_save_result_writes_both_and_mints_url(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    res = asyncio.run(storage.save_result(
        PNG, project="Proj", subpath=None, filename="hero", ext="png", cfg=cfg))
    assert res["workspace_path"] == "C:\\Workspace\\Proj\\assets\\forge\\hero.png"
    assert os.path.isfile(os.path.join(cfg.workspace_root, "Proj", "assets", "forge", "hero.png"))
    assert res["url"].startswith("https://forge.example.com/files/")
    fid = res["url"].split("/files/")[1].split("/")[0]
    assert os.path.isfile(os.path.join(cfg.results_root, "files", fid, "hero.png"))
    assert "workspace_write_error" not in res


def test_save_result_collision_suffix(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    r1 = asyncio.run(storage.save_result(PNG, project="Proj", subpath=None, filename="x", ext="png", cfg=cfg))
    r2 = asyncio.run(storage.save_result(PNG, project="Proj", subpath=None, filename="x", ext="png", cfg=cfg))
    assert r1["workspace_path"].endswith("x.png")
    assert r2["workspace_path"].endswith("x-2.png")


def test_save_result_workspace_down_degrades(tmp_path):
    cfg = make_cfg(tmp_path)  # workspace_root never created -> write fails
    res = asyncio.run(storage.save_result(PNG, project="Proj", subpath=None, filename="y", ext="png", cfg=cfg))
    assert res["url"].startswith("https://")
    assert "workspace_write_error" in res


def test_janitor_prunes_old(tmp_path):
    cfg = make_cfg(tmp_path)
    old_dir = os.path.join(cfg.results_root, "files", "oldid123")
    os.makedirs(old_dir)
    with open(os.path.join(old_dir, "a.png"), "wb") as f:
        f.write(PNG)
    past = time.time() - 90 * 86400
    os.utime(old_dir, (past, past))
    new_res = asyncio.run(storage.save_result(PNG, project=None, subpath=None, filename="keep", ext="png", cfg=cfg))
    storage.prune_cache_once(cfg)
    assert not os.path.isdir(old_dir)
    kept_id = new_res["url"].split("/files/")[1].split("/")[0]
    assert os.path.isdir(os.path.join(cfg.results_root, "files", kept_id))
```

Run: `python -m pytest tests/test_storage.py -v` → expect FAIL (module missing).

- [ ] **Step 3.2: Implement `forge_mcp/storage.py`**

```python
"""Input resolution (URL / workspace path), result writing (workspace + cache), URL minting, janitor."""
import asyncio
import difflib
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from io import BytesIO

import httpx

from forge_mcp.config import Config

URL_FETCH_TIMEOUT_S = 60


class StorageError(Exception):
    """Readable, user-facing storage failure."""


@dataclass
class ResolvedInput:
    data: bytes
    mime: str
    source: str  # 'url' | 'workspace'


_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),       # checked further below
    (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"),
    (b"\x1a\x45\xdf\xa3", "video/webm"),
]


def sniff_mime(data: bytes):
    if len(data) < 12:
        return None
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    if data[4:8] == b"ftyp":  # mp4 / mov family
        brand = data[8:12]
        return "video/quicktime" if brand in (b"qt  ",) else "video/mp4"
    return None


def is_image_mime(mime):
    return bool(mime) and mime.startswith("image/")


def is_video_mime(mime):
    return bool(mime) and mime.startswith("video/")


def _workspace_abs(rel: str, cfg: Config) -> str:
    """Resolve '<Project>/<relpath>' under workspace_root, rejecting traversal."""
    norm = os.path.normpath(rel.replace("\\", "/")).replace("\\", "/")
    if norm.startswith("..") or "/../" in norm or norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise StorageError(f"Invalid workspace path (traversal not allowed): {rel}")
    abs_path = os.path.normpath(os.path.join(cfg.workspace_root, norm))
    root = os.path.normpath(cfg.workspace_root)
    if not abs_path.startswith(root + os.sep) and abs_path != root:
        raise StorageError(f"Invalid workspace path (escapes workspace): {rel}")
    return abs_path


async def resolve_input(ref: str, *, cfg: Config, kind: str = "image") -> ResolvedInput:
    """ref is an https URL or '<Project>/<relpath>' inside the workspace."""
    max_bytes = (cfg.max_image_mb if kind == "image" else cfg.max_video_mb) * 1024 * 1024
    if ref.startswith("http://") or ref.startswith("https://"):
        async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT_S, follow_redirects=True) as client:
            try:
                resp = await client.get(ref)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise StorageError(f"Could not fetch {ref}: {e}") from e
            data = resp.content
        source = "url"
    else:
        path = _workspace_abs(ref, cfg)
        if not os.path.isfile(path):
            raise StorageError(
                f"No file at '{ref}' in the workspace. Check the project name and path "
                f"(input paths are '<Project>/<relative path>')."
            )
        data = await asyncio.to_thread(_read_file, path)
        source = "workspace"

    if len(data) > max_bytes:
        raise StorageError(f"Input too large: {len(data) // (1024*1024)} MB (limit {max_bytes // (1024*1024)} MB)")
    mime = sniff_mime(data)
    if kind == "image" and not is_image_mime(mime):
        raise StorageError(f"Input from '{ref}' is not a recognized image (png/jpg/webp/gif/bmp)")
    if kind == "video" and not is_video_mime(mime):
        raise StorageError(f"Input from '{ref}' is not a recognized video (mp4/webm/mov)")
    return ResolvedInput(data=data, mime=mime, source=source)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def validate_project(project: str, *, cfg: Config) -> str:
    if not project or "/" in project or "\\" in project or project.startswith("."):
        raise StorageError(f"Invalid project name: {project!r}")
    path = os.path.join(cfg.workspace_root, project)
    if os.path.isdir(path):
        return project
    try:
        existing = [d for d in os.listdir(cfg.workspace_root)
                    if os.path.isdir(os.path.join(cfg.workspace_root, d)) and not d.startswith(".")]
    except OSError as e:
        raise StorageError(f"Workspace not reachable ({e}); cannot validate project '{project}'") from e
    close = difflib.get_close_matches(project, existing, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise StorageError(f"No workspace project named '{project}'.{hint} (Use list_projects to see all.)")


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name).strip(" .")
    return name[:120] or "output"


def _unique_path(directory: str, base: str, ext: str) -> str:
    candidate = os.path.join(directory, f"{base}.{ext}")
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}-{n}.{ext}")
        n += 1
    return candidate


def _image_dims(data: bytes):
    try:
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            return im.width, im.height
    except Exception:
        return None, None


async def save_result(data: bytes, *, project, subpath, filename, ext, cfg: Config) -> dict:
    """Write to results cache (always) + workspace (best effort). Returns the tool result dict."""
    base = safe_filename(filename or "output")
    file_id = secrets.token_urlsafe(18)
    cache_dir = os.path.join(cfg.results_root, "files", file_id)
    await asyncio.to_thread(os.makedirs, cache_dir, True)
    cache_name = f"{base}.{ext}"
    await asyncio.to_thread(_write_file, os.path.join(cache_dir, cache_name), data)

    result = {
        "url": f"{cfg.public_url}/files/{file_id}/{cache_name}",
        "bytes": len(data),
    }
    if ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp"):
        w, h = _image_dims(data)
        if w:
            result["width"], result["height"] = w, h

    if project:
        sub = (subpath or "assets/forge").strip("/\\")
        try:
            validate_project(project, cfg=cfg)
            rel_dir = os.path.join(project, *sub.replace("\\", "/").split("/"))
            target_dir = _workspace_abs(rel_dir, cfg)
            await asyncio.to_thread(os.makedirs, target_dir, True)
            target = await asyncio.to_thread(_unique_path, target_dir, base, ext)
            await asyncio.to_thread(_write_file, target, data)
            rel_to_root = os.path.relpath(target, cfg.workspace_root)
            result["workspace_path"] = os.path.join(cfg.workspace_display_root, rel_to_root)
        except (StorageError, OSError) as e:
            result["workspace_write_error"] = (
                f"Result NOT written to the workspace ({e}). It is still available at the url above."
            )
    return result


def _write_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def prune_cache_once(cfg: Config):
    """Delete /results/files/<id> dirs older than the TTL."""
    files_root = os.path.join(cfg.results_root, "files")
    if not os.path.isdir(files_root):
        return
    cutoff = time.time() - cfg.cache_ttl_days * 86400
    for entry in os.listdir(files_root):
        path = os.path.join(files_root, entry)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


async def janitor_loop(cfg: Config):
    while True:
        await asyncio.to_thread(prune_cache_once, cfg)
        await asyncio.sleep(24 * 3600)
```

Note: `os.makedirs(dir, True)` is wrong — the second positional arg is `mode`. Use `functools.partial(os.makedirs, dir, exist_ok=True)` or a small `_makedirs` helper:

```python
def _makedirs(path: str):
    os.makedirs(path, exist_ok=True)
```
and call `await asyncio.to_thread(_makedirs, cache_dir)` / `(_makedirs, target_dir)`. Implement it that way.

- [ ] **Step 3.3: Run tests**

`python -m pytest tests/test_storage.py -v` → expect ALL PASS.

- [ ] **Step 3.4: Commit**

```bash
git add forge_mcp/storage.py tests/test_storage.py
git commit -m "feat(mcp): storage — input resolution, dual-write results, URL minting, cache janitor"
git push
```

---

## Task 4: `forge_mcp/jobs.py` — Veo job registry with persistence

**Files:**
- Create: `forge_mcp/jobs.py`, `tests/test_jobs.py`

- [ ] **Step 4.1: Write failing tests `tests/test_jobs.py`**

```python
from forge_mcp.jobs import JobStore


def test_create_update_get(tmp_path):
    store = JobStore(str(tmp_path / "jobs.json"))
    job = store.create(kind="veo", model="veo-3.0-fast-generate-001", prompt="a cat",
                       project="Proj", subpath=None, filename="cat")
    assert job["status"] == "running"
    store.update(job["id"], status="done", results=[{"url": "https://x/y.mp4"}])
    got = store.get(job["id"])
    assert got["status"] == "done" and got["results"][0]["url"] == "https://x/y.mp4"


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "jobs.json")
    s1 = JobStore(path)
    job = s1.create(kind="veo", model="m", prompt="p", project=None, subpath=None, filename=None)
    s1.update(job["id"], operation_name="operations/abc")
    s2 = JobStore(path)  # fresh load from disk
    got = s2.get(job["id"])
    assert got["operation_name"] == "operations/abc"
    assert got["status"] == "running"


def test_mark_interrupted_without_operation(tmp_path):
    path = str(tmp_path / "jobs.json")
    s1 = JobStore(path)
    a = s1.create(kind="veo", model="m", prompt="p", project=None, subpath=None, filename=None)
    b = s1.create(kind="veo", model="m", prompt="p2", project=None, subpath=None, filename=None)
    s1.update(b["id"], operation_name="operations/xyz")
    s2 = JobStore(path)
    resumable = s2.mark_interrupted()
    assert s2.get(a["id"])["status"] == "failed"          # no operation name -> lost
    assert [j["id"] for j in resumable] == [b["id"]]       # has operation -> caller resumes

def test_get_unknown(tmp_path):
    store = JobStore(str(tmp_path / "jobs.json"))
    assert store.get("nope") is None
```

Run: `python -m pytest tests/test_jobs.py -v` → expect FAIL.

- [ ] **Step 4.2: Implement `forge_mcp/jobs.py`**

```python
"""Veo job registry — in-memory dict mirrored to a JSON file (atomic writes)."""
import json
import os
import secrets
import threading
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._jobs = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._jobs = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._jobs = {}

    def _flush(self):
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._jobs, f, indent=1)
        os.replace(tmp, self.path)

    def create(self, *, kind, model, prompt, project, subpath, filename) -> dict:
        job = {
            "id": "job_" + secrets.token_urlsafe(8),
            "kind": kind, "model": model, "prompt": prompt,
            "project": project, "subpath": subpath, "filename": filename,
            "status": "running", "message": "submitted",
            "operation_name": None, "results": [], "error": None,
            "created_at": _now(), "updated_at": _now(),
        }
        with self._lock:
            self._jobs[job["id"]] = job
            self._flush()
        return dict(job)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = _now()
            self._flush()

    def get(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def mark_interrupted(self) -> list:
        """Call on startup. Jobs mid-flight without a Google operation are dead;
        jobs WITH an operation name are returned for the caller to resume polling."""
        resumable = []
        with self._lock:
            for job in self._jobs.values():
                if job["status"] != "running":
                    continue
                if job.get("operation_name"):
                    resumable.append(dict(job))
                else:
                    job["status"] = "failed"
                    job["error"] = "interrupted by service restart before submission completed"
                    job["updated_at"] = _now()
            self._flush()
        return resumable
```

- [ ] **Step 4.3: Run tests** → `python -m pytest tests/test_jobs.py -v` → PASS.

- [ ] **Step 4.4: Commit**

```bash
git add forge_mcp/jobs.py tests/test_jobs.py
git commit -m "feat(mcp): persistent Veo job registry with restart resume/interrupt semantics"
git push
```

---

## Task 5: `forge_mcp/generation.py` — Imagen / Gemini / Veo client (port of main.js)

**Files:**
- Create: `forge_mcp/generation.py`, `tests/test_generation.py`

- [ ] **Step 5.1: Write failing tests (pure parsers + aliases) `tests/test_generation.py`**

```python
import base64
import pytest
from forge_mcp import generation as g


def test_model_aliases():
    assert g.resolve_image_model("imagen-4") == "imagen-4.0-generate-001"
    assert g.resolve_image_model("imagen-4-fast") == "imagen-4.0-fast-generate-001"
    assert g.resolve_image_model("imagen-4-ultra") == "imagen-4.0-ultra-generate-001"
    assert g.resolve_video_model("veo-3") == "veo-3.0-generate-001"
    assert g.resolve_video_model("veo-3-fast") == "veo-3.0-fast-generate-001"
    assert g.resolve_video_model("veo-2") == "veo-2.0-generate-001"
    with pytest.raises(g.GenerationError):
        g.resolve_image_model("dalle-3")


def test_max_batch():
    assert g.imagen_max_batch("imagen-4.0-ultra-generate-001") == 1
    assert g.imagen_max_batch("imagen-4.0-generate-001") == 4


def test_parse_imagen_predictions():
    b64 = base64.b64encode(b"PNGDATA").decode()
    assert g.parse_imagen_predictions({"predictions": [{"bytesBase64Encoded": b64}]}) == [b"PNGDATA"]
    assert g.parse_imagen_predictions({"generatedImages": [{"image": {"imageBytes": b64}}]}) == [b"PNGDATA"]
    assert g.parse_imagen_predictions({}) == []


def test_parse_gemini_parts():
    b64 = base64.b64encode(b"IMG").decode()
    json_resp = {"candidates": [{"content": {"parts": [
        {"text": "here you go"},
        {"inlineData": {"mimeType": "image/png", "data": b64}},
    ]}}]}
    assert g.parse_gemini_parts(json_resp) == [b"IMG"]


def test_parse_veo_done_extracts_sample():
    done = {"done": True, "response": {"generateVideoResponse": {"generatedSamples": [
        {"video": {"uri": "https://dl/v.mp4"}}]}}}
    assert g.parse_veo_done(done) == {"uri": "https://dl/v.mp4", "inline_b64": None}


def test_parse_veo_done_safety_filtered():
    with pytest.raises(g.GenerationError, match="safety"):
        g.parse_veo_done({"done": True, "response": {}})


def test_veo_request_body_audio_only_for_veo3():
    b3 = g.build_veo_body("veo-3.0-generate-001", "a dog", None, "16:9", 8, True)
    assert b3["parameters"]["generateAudio"] is True
    b2 = g.build_veo_body("veo-2.0-generate-001", "a dog", None, "16:9", 8, True)
    assert "generateAudio" not in b2["parameters"]
```

Run: `python -m pytest tests/test_generation.py -v` → FAIL.

- [ ] **Step 5.2: Implement `forge_mcp/generation.py`**

```python
"""Google generativelanguage client — direct port of main.js generation logic."""
import asyncio
import base64

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
    import time as _time
    start = _time.monotonic()
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
        elapsed = int(_time.monotonic() - start)
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
```

- [ ] **Step 5.3: Run tests** → `python -m pytest tests/test_generation.py -v` → PASS.

- [ ] **Step 5.4: Commit**

```bash
git add forge_mcp/generation.py tests/test_generation.py
git commit -m "feat(mcp): Imagen/Gemini/Veo generation client ported from main.js"
git push
```

---

## Task 6: `forge_mcp/video.py` — ffmpeg ops

**Files:**
- Create: `forge_mcp/video.py`, `tests/test_video.py`

- [ ] **Step 6.1: Write failing tests `tests/test_video.py`** (pure helpers; execution is smoke-tested in Task 8)

```python
import pytest
from forge_mcp import video as v


def test_to_seconds():
    assert v.to_seconds("90") == 90.0
    assert v.to_seconds("01:30") == 90.0
    assert v.to_seconds("00:01:30.5") == 90.5
    assert v.to_seconds(12.5) == 12.5
    with pytest.raises(v.VideoError):
        v.to_seconds("abc")


def test_trim_cmd_stream_copy_vs_reencode():
    copy_cmd = v.build_trim_cmd("in.mp4", "out.mp4", 1.0, 5.0, reencode=False)
    assert "-c" in copy_cmd and "copy" in copy_cmd
    re_cmd = v.build_trim_cmd("in.mp4", "out.mp4", 1.0, 5.0, reencode=True)
    assert "libx264" in re_cmd


def test_convert_cmd_formats():
    assert "libx264" in v.build_convert_cmd("a.webm", "out.mp4", "mp4", crf=23, scale=None)
    assert "libvpx-vp9" in v.build_convert_cmd("a.mp4", "out.webm", "webm", crf=32, scale=720)
    gif = v.build_convert_cmd("a.mp4", "out.gif", "gif", crf=None, scale=480)
    assert "palettegen" in " ".join(gif) and "paletteuse" in " ".join(gif)


def test_frames_cmd():
    ts = v.build_frame_cmd("a.mp4", "out.png", 2.5, "png")
    assert "-ss" in ts and "2.5" in ts
```

Run: `python -m pytest tests/test_video.py -v` → FAIL.

- [ ] **Step 6.2: Implement `forge_mcp/video.py`**

```python
"""ffmpeg wrappers — trim, frame extraction, conversion. Bytes in -> bytes out via temp files."""
import asyncio
import os
import re
import tempfile

FFMPEG_TIMEOUT_S = 600


class VideoError(Exception):
    """Readable, user-facing video-processing failure."""


def to_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", s)
    if not m:
        raise VideoError(f"Bad time '{value}' — use seconds or [HH:]MM:SS[.ms]")
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def build_trim_cmd(inp, outp, start_s, end_s, *, reencode):
    cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-to", str(end_s), "-i", inp]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    return cmd + [outp]


def build_convert_cmd(inp, outp, fmt, *, crf, scale):
    vf = []
    if scale:
        vf.append(f"scale=-2:{scale}")
    cmd = ["ffmpeg", "-y", "-i", inp]
    if fmt == "mp4":
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf or 23), "-c:a", "aac"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
    elif fmt == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-crf", str(crf or 32), "-b:v", "0", "-c:a", "libopus"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
    elif fmt == "gif":
        scale_f = f"scale=-2:{scale or 480}:flags=lanczos"
        cmd += ["-vf", f"fps=12,{scale_f},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"]
    else:
        raise VideoError(f"Unsupported format '{fmt}' (mp4, webm, gif)")
    return cmd + [outp]


def build_frame_cmd(inp, outp, ts_s, fmt):
    return ["ffmpeg", "-y", "-ss", str(ts_s), "-i", inp, "-frames:v", "1", outp]


async def _run(cmd) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        raise VideoError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s")
    if proc.returncode != 0:
        tail = (stderr or b"").decode(errors="replace")[-800:]
        raise VideoError(f"ffmpeg failed (exit {proc.returncode}): {tail}")


class _Tmp:
    """Temp file that survives Windows handle semantics (close before ffmpeg reads it)."""
    def __init__(self, suffix, data=None):
        fd, self.path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            if data:
                f.write(data)
    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()
    def cleanup(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass


async def trim(data: bytes, start, end, in_ext="mp4") -> bytes:
    start_s, end_s = to_seconds(start), to_seconds(end)
    if end_s <= start_s:
        raise VideoError(f"end ({end_s}s) must be after start ({start_s}s)")
    src, dst = _Tmp(f".{in_ext}", data), _Tmp(".mp4")
    try:
        await _run(build_trim_cmd(src.path, dst.path, start_s, end_s, reencode=False))
        out = dst.read()
        if len(out) < 1024:  # stream-copy produced garbage (keyframe issues) -> re-encode
            await _run(build_trim_cmd(src.path, dst.path, start_s, end_s, reencode=True))
            out = dst.read()
        if len(out) < 1024:
            raise VideoError("Trim produced an empty file")
        return out
    finally:
        src.cleanup(); dst.cleanup()


async def extract_frames(data: bytes, timestamps, fmt="png", in_ext="mp4") -> list:
    src = _Tmp(f".{in_ext}", data)
    out = []
    try:
        for ts in timestamps:
            dst = _Tmp(f".{fmt}")
            try:
                await _run(build_frame_cmd(src.path, dst.path, to_seconds(ts), fmt))
                frame = dst.read()
                if not frame:
                    raise VideoError(f"No frame at {ts} (past end of video?)")
                out.append(frame)
            finally:
                dst.cleanup()
        return out
    finally:
        src.cleanup()


async def convert(data: bytes, fmt, *, crf=None, scale=None, in_ext="mp4") -> bytes:
    src, dst = _Tmp(f".{in_ext}", data), _Tmp(f".{fmt}")
    try:
        await _run(build_convert_cmd(src.path, dst.path, fmt, crf=crf, scale=scale))
        out = dst.read()
        if not out:
            raise VideoError("Conversion produced an empty file")
        return out
    finally:
        src.cleanup(); dst.cleanup()
```

- [ ] **Step 6.3: Run tests** → `python -m pytest tests/test_video.py -v` → PASS.

- [ ] **Step 6.4: Commit**

```bash
git add forge_mcp/video.py tests/test_video.py
git commit -m "feat(mcp): ffmpeg video ops — trim (copy w/ re-encode fallback), frames, convert"
git push
```

---

## Task 7: `forge_mcp/engine.py` — async bridge to the processing engine

**Files:**
- Create: `forge_mcp/engine.py`

- [ ] **Step 7.1: Implement (no unit test — the math is the proven desktop code; smoke in Task 8)**

```python
"""Async bridge to backend.processing. CPU-heavy work runs in threads behind a
small semaphore so concurrent MCP calls can't thrash the box."""
import asyncio
import threading

from backend import processing
from backend.processing import PipelineOptions, AVAILABLE_MODELS, DEFAULT_MODEL  # re-export

_cpu_gate = asyncio.Semaphore(2)
_load_lock = threading.Lock()


def _ensure_session(model_id: str):
    with _load_lock:  # rembg new_session is not safe to race
        processing.get_session(model_id)


async def preload_default_model():
    await asyncio.to_thread(_ensure_session, DEFAULT_MODEL)


async def run_pipeline(data: bytes, opts: PipelineOptions) -> bytes:
    async with _cpu_gate:
        await asyncio.to_thread(_ensure_session, opts.model)
        return await asyncio.to_thread(processing.run_pipeline, data, opts)


async def run_split_pipeline(data: bytes, opts: PipelineOptions, min_area: int) -> list:
    async with _cpu_gate:
        await asyncio.to_thread(_ensure_session, opts.model)
        return await asyncio.to_thread(processing.run_split_pipeline, data, opts, min_area)


async def split_only(data: bytes, min_area: int) -> list:
    async with _cpu_gate:
        return await asyncio.to_thread(processing.split_sprites, data, min_area)


def status() -> dict:
    return {
        "loaded_models": sorted(processing.sessions.keys()),
        "lama_model_present": __import__("os").path.exists(processing.LAMA_MODEL_PATH),
        "lama_loaded": processing.lama_session is not None,
    }
```

`backend/` needs an `__init__.py` for clean package imports — create empty `backend/__init__.py` (PyInstaller desktop build is unaffected; it bundles `server.py` as a script).

- [ ] **Step 7.2: Commit**

```bash
git add forge_mcp/engine.py backend/__init__.py
git commit -m "feat(mcp): async engine bridge with CPU gate and safe model loading"
git push
```

---

## Task 8: Tools — registration + local end-to-end smoke

**Files:**
- Create: `forge_mcp/tools/__init__.py`, `forge_mcp/tools/proc.py`, `forge_mcp/tools/gen.py`, `forge_mcp/tools/vid.py`, `forge_mcp/tools/meta.py`
- Modify: `forge_mcp/server.py`

- [ ] **Step 8.1: `forge_mcp/tools/__init__.py`**

```python
from forge_mcp.tools import gen, meta, proc, vid


def register_all(mcp, ctx):
    proc.register(mcp, ctx)
    gen.register(mcp, ctx)
    vid.register(mcp, ctx)
    meta.register(mcp, ctx)
```

`ctx` is a plain namespace built in server.py: `cfg`, `jobs` (JobStore), `http` (shared `httpx.AsyncClient`).

- [ ] **Step 8.2: `forge_mcp/tools/proc.py`**

```python
"""Image-processing tools — all delegate to the shared desktop pipeline."""
from backend.processing import PipelineOptions, parse_colors
from forge_mcp import engine, storage

IMAGE_PARAM_DOC = "Image as an https URL or a workspace path '<Project>/<relative path>'"
DEST_DOC = "project: workspace folder to save into; subpath (default assets/forge); filename (no extension)"


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def remove_background(
        image: str,
        project: str,
        model: str = "u2net",
        alpha_matting: bool = False,
        fg_threshold: int = 240,
        bg_threshold: int = 10,
        erode_size: int = 10,
        color_remove: bool = False,
        colors: list[str] | None = None,
        color_auto_detect: bool = False,
        color_tolerance: int = 20,
        edge_smooth: bool = False,
        edge_strength: int = 50,
        edge_trim: int = 0,
        auto_trim: bool = True,
        watermark_remove: bool = False,
        watermark_position: str = "bottom-right",
        watermark_size_pct: int = 15,
        subpath: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Remove the background from an image (rembg). Optional pipeline steps: watermark
        removal first, then color removal, edge smoothing, and transparent-edge trim.
        Models: u2net (default), u2netp (fast), u2net_human_seg, isnet-general-use, silueta."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        opts = PipelineOptions(
            model=model, alpha_matting=alpha_matting, fg_threshold=fg_threshold,
            bg_threshold=bg_threshold, erode_size=erode_size, color_remove=color_remove,
            colors=parse_colors(colors), color_auto_detect=color_auto_detect,
            color_tolerance=color_tolerance, edge_smooth=edge_smooth,
            edge_strength=edge_strength, edge_trim=edge_trim, auto_trim=auto_trim,
            watermark_remove=watermark_remove, watermark_position=watermark_position,
            watermark_size_pct=watermark_size_pct,
        )
        out = await engine.run_pipeline(src.data, opts)
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "no-bg", ext="png", cfg=cfg)

    @mcp.tool()
    async def split_sprites(
        image: str,
        project: str,
        min_sprite_area: int = 400,
        skip_bg: bool = False,
        model: str = "u2net",
        subpath: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Split a sprite sheet into individual trimmed PNG sprites. By default runs
        background removal + auto color/edge cleanup first; set skip_bg=true if the
        image is already transparent."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        if skip_bg:
            sprites = await engine.split_only(src.data, min_sprite_area)
        else:
            sprites = await engine.run_split_pipeline(src.data, PipelineOptions(model=model), min_sprite_area)
        base = filename or "sprite"
        results = []
        for i, s in enumerate(sprites, 1):
            results.append(await storage.save_result(
                s, project=project, subpath=subpath, filename=f"{base}-{i}", ext="png", cfg=cfg))
        return {"count": len(results), "sprites": results}

    @mcp.tool()
    async def trim_image(image: str, project: str, subpath: str | None = None,
                         filename: str | None = None) -> dict:
        """Crop away transparent padding around an image (keeps 1px margin)."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        out = await engine.run_pipeline(src.data, PipelineOptions(skip_bg=True, auto_trim=True))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "trimmed", ext="png", cfg=cfg)

    @mcp.tool()
    async def remove_watermark(image: str, project: str, position: str = "bottom-right",
                               size_pct: int = 15, subpath: str | None = None,
                               filename: str | None = None) -> dict:
        """Remove a watermark via LaMa inpainting. position: bottom-right, bottom-left,
        top-right, top-left, bottom-center, top-center. size_pct: region size as % of image."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        out = await engine.run_pipeline(src.data, PipelineOptions(
            skip_bg=True, watermark_remove=True, watermark_position=position,
            watermark_size_pct=size_pct))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "no-watermark", ext="png", cfg=cfg)

    @mcp.tool()
    async def remove_colors(image: str, project: str, colors: list[str] | None = None,
                            tolerance: int = 20, auto_detect: bool = False,
                            subpath: str | None = None, filename: str | None = None) -> dict:
        """Make pixels near the given colors transparent. colors: hex strings like '#ffffff'.
        auto_detect samples the image edges for the background color."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        out = await engine.run_pipeline(src.data, PipelineOptions(
            skip_bg=True, color_remove=True, colors=parse_colors(colors),
            color_auto_detect=auto_detect or not colors, color_tolerance=tolerance))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "color-removed", ext="png", cfg=cfg)

    @mcp.tool()
    async def smooth_edges(image: str, project: str, strength: int = 50, trim_px: int = 0,
                           subpath: str | None = None, filename: str | None = None) -> dict:
        """Smooth jagged alpha edges and defringe discolored edge pixels on a transparent image."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        out = await engine.run_pipeline(src.data, PipelineOptions(
            skip_bg=True, edge_smooth=True, edge_strength=strength, edge_trim=trim_px))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "smoothed", ext="png", cfg=cfg)
```

Wrap every tool body's storage/engine errors? No — FastMCP converts raised exceptions into tool errors with the message; `StorageError`/`GenerationError`/`VideoError` messages are already user-readable. That is the error contract.

- [ ] **Step 8.3: `forge_mcp/tools/gen.py`**

```python
"""Generation tools — Imagen, Gemini edit, Veo (async jobs)."""
import asyncio

from forge_mcp import generation as g
from forge_mcp import storage


def register(mcp, ctx):
    cfg, jobs, http = ctx.cfg, ctx.jobs, ctx.http

    def _require_key():
        if not cfg.gemini_api_key:
            raise g.GenerationError("GEMINI_API_KEY is not configured on the forge service")

    @mcp.tool()
    async def generate_image(prompt: str, project: str, model: str = "imagen-4",
                             count: int = 1, aspect_ratio: str = "1:1",
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate image(s) from a text prompt with Imagen 4.
        model: imagen-4 | imagen-4-fast | imagen-4-ultra. count: 1-4 (ultra max 1).
        aspect_ratio: 1:1, 3:4, 4:3, 9:16, 16:9."""
        _require_key()
        if aspect_ratio not in g.IMAGE_ASPECTS:
            raise g.GenerationError(f"aspect_ratio must be one of {g.IMAGE_ASPECTS}")
        model_id = g.resolve_image_model(model)
        n = max(1, min(count, g.imagen_max_batch(model_id)))
        images = await g.call_imagen(http, cfg.gemini_api_key, model_id, prompt,
                                     sample_count=n, aspect_ratio=aspect_ratio)
        if not images:
            raise g.GenerationError("Imagen returned no images (prompt may have been refused)")
        base = storage.safe_filename(filename or prompt[:48])
        results = []
        for i, img in enumerate(images, 1):
            name = base if len(images) == 1 else f"{base}-{i}"
            results.append(await storage.save_result(img, project=project, subpath=subpath,
                                                     filename=name, ext="png", cfg=cfg))
        return {"count": len(results), "images": results}

    @mcp.tool()
    async def edit_image(prompt: str, reference_images: list[str], project: str,
                         subpath: str | None = None, filename: str | None = None) -> dict:
        """Edit/compose images with Gemini: give 1-6 reference images (https URLs or
        '<Project>/<path>' workspace paths) plus an instruction prompt. Returns the new image(s)."""
        _require_key()
        if not reference_images or len(reference_images) > 6:
            raise g.GenerationError("Provide 1-6 reference_images")
        refs = []
        for r in reference_images:
            resolved = await storage.resolve_input(r, cfg=cfg, kind="image")
            refs.append((resolved.mime, resolved.data))
        images = await g.call_gemini_image(http, cfg.gemini_api_key, prompt, refs)
        base = storage.safe_filename(filename or "edited")
        results = []
        for i, img in enumerate(images, 1):
            name = base if len(images) == 1 else f"{base}-{i}"
            results.append(await storage.save_result(img, project=project, subpath=subpath,
                                                     filename=name, ext="png", cfg=cfg))
        return {"count": len(results), "images": results}

    async def _run_veo_job(job_id: str, model_id: str, prompt: str, start_image,
                           aspect_ratio: str, duration_seconds: int, generate_audio: bool):
        try:
            op = await g.start_veo(http, cfg.gemini_api_key, model_id, prompt,
                                   start_image=start_image, aspect_ratio=aspect_ratio,
                                   duration_seconds=duration_seconds, generate_audio=generate_audio)
            jobs.update(job_id, operation_name=op, message="submitted to Veo")
            await _poll_and_finish(job_id, op)
        except Exception as e:  # job boundary: everything becomes a readable failed status
            jobs.update(job_id, status="failed", error=str(e))

    async def _poll_and_finish(job_id: str, op: str):
        job = jobs.get(job_id)
        sample = await g.poll_veo(http, cfg.gemini_api_key, op,
                                  on_progress=lambda m: jobs.update(job_id, message=m))
        jobs.update(job_id, message="downloading video")
        mp4 = await g.download_veo_video(http, sample, cfg.gemini_api_key)
        res = await storage.save_result(mp4, project=job["project"], subpath=job["subpath"],
                                        filename=job["filename"] or "veo", ext="mp4", cfg=cfg)
        jobs.update(job_id, status="done", message="complete", results=[res])

    ctx.poll_and_finish = _poll_and_finish  # server.py uses this to resume jobs on startup

    @mcp.tool()
    async def generate_video(prompt: str, project: str, model: str = "veo-3-fast",
                             start_image: str | None = None, aspect_ratio: str = "16:9",
                             duration_seconds: int = 8, generate_audio: bool = True,
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate a video with Veo (takes minutes — returns a job_id immediately; poll with
        job_status). model: veo-3 | veo-3-fast | veo-2. aspect_ratio: 16:9 or 9:16.
        start_image: optional image to animate (URL or workspace path). COSTS REAL MONEY per clip."""
        _require_key()
        if aspect_ratio not in g.VIDEO_ASPECTS:
            raise g.GenerationError(f"aspect_ratio must be one of {g.VIDEO_ASPECTS}")
        model_id = g.resolve_video_model(model)
        storage.validate_project(project, cfg=cfg)  # fail fast before paying Google
        img = None
        if start_image:
            r = await storage.resolve_input(start_image, cfg=cfg, kind="image")
            img = (r.mime, r.data)
        job = jobs.create(kind="veo", model=model_id, prompt=prompt, project=project,
                          subpath=subpath, filename=filename)
        asyncio.create_task(_run_veo_job(job["id"], model_id, prompt, img,
                                         aspect_ratio, duration_seconds, generate_audio))
        return {"job_id": job["id"], "status": "running",
                "note": "Video generation takes 1-6 minutes. Poll with job_status."}

    @mcp.tool()
    async def job_status(job_id: str) -> dict:
        """Status of a generate_video job: running (with progress message), done (with results), or failed."""
        job = jobs.get(job_id)
        if not job:
            raise g.GenerationError(f"No job '{job_id}'")
        return {k: job[k] for k in ("id", "status", "message", "results", "error", "created_at", "updated_at")}
```

- [ ] **Step 8.4: `forge_mcp/tools/vid.py`**

```python
"""ffmpeg-backed video tools."""
from forge_mcp import storage, video


def _ext_for(mime: str) -> str:
    return {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov"}.get(mime, "mp4")


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def video_trim(video_input: str, start: str, end: str, project: str,
                         subpath: str | None = None, filename: str | None = None) -> dict:
        """Trim a video to [start, end]. Times as seconds ('12.5') or '[HH:]MM:SS[.ms]'.
        video_input: https URL or workspace path."""
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        out = await video.trim(src.data, start, end, in_ext=_ext_for(src.mime))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "trimmed", ext="mp4", cfg=cfg)

    @mcp.tool()
    async def video_extract_frames(video_input: str, timestamps: list[str], project: str,
                                   image_format: str = "png", subpath: str | None = None,
                                   filename: str | None = None) -> dict:
        """Extract still frames at the given timestamps (max 20). image_format: png or jpg."""
        if not timestamps or len(timestamps) > 20:
            raise video.VideoError("Provide 1-20 timestamps")
        if image_format not in ("png", "jpg"):
            raise video.VideoError("image_format must be png or jpg")
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        frames = await video.extract_frames(src.data, timestamps, fmt=image_format,
                                            in_ext=_ext_for(src.mime))
        base = filename or "frame"
        results = []
        for ts, frame in zip(timestamps, frames):
            safe_ts = str(ts).replace(":", "-").replace(".", "_")
            results.append(await storage.save_result(
                frame, project=project, subpath=subpath,
                filename=f"{base}-{safe_ts}", ext=image_format, cfg=cfg))
        return {"count": len(results), "frames": results}

    @mcp.tool()
    async def video_convert(video_input: str, output_format: str, project: str,
                            crf: int | None = None, scale: int | None = None,
                            subpath: str | None = None, filename: str | None = None) -> dict:
        """Convert/compress a video. output_format: mp4, webm, or gif. crf: quality
        (lower=better; mp4 default 23, webm 32). scale: target height in px (e.g. 720)."""
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        out = await video.convert(src.data, output_format, crf=crf, scale=scale,
                                  in_ext=_ext_for(src.mime))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "converted", ext=output_format, cfg=cfg)
```

- [ ] **Step 8.5: `forge_mcp/tools/meta.py`**

```python
"""Status + discovery tools."""
import os
import shutil
import subprocess

from backend.processing import AVAILABLE_MODELS
from forge_mcp import engine
from forge_mcp.generation import IMAGE_MODEL_ALIASES, VIDEO_MODEL_ALIASES


def register(mcp, ctx):
    cfg, jobs = ctx.cfg, ctx.jobs

    @mcp.tool()
    async def forge_status() -> dict:
        """Health of the forge service: engine models, ffmpeg, workspace mount, jobs, config."""
        ws_ok, ws_err = True, None
        try:
            probe = os.path.join(cfg.workspace_root, ".forge-write-test")
            with open(probe, "w") as f:
                f.write("ok")
            os.unlink(probe)
        except OSError as e:
            ws_ok, ws_err = False, str(e)
        ffmpeg_ok = shutil.which("ffmpeg") is not None
        cache_files = 0
        files_root = os.path.join(cfg.results_root, "files")
        if os.path.isdir(files_root):
            cache_files = len(os.listdir(files_root))
        running = [j for j in (jobs.get(i) for i in list(getattr(jobs, '_jobs', {}))) if j and j["status"] == "running"]
        return {
            "engine": engine.status(),
            "ffmpeg_available": ffmpeg_ok,
            "workspace_writable": ws_ok,
            "workspace_error": ws_err,
            "gemini_key_configured": bool(cfg.gemini_api_key),
            "results_cached": cache_files,
            "jobs_running": len(running),
            "public_url": cfg.public_url,
        }

    @mcp.tool()
    async def list_models() -> dict:
        """Available background-removal models and generation model aliases."""
        return {
            "background_removal": AVAILABLE_MODELS,
            "image_generation": list(IMAGE_MODEL_ALIASES.keys()),
            "image_edit": ["gemini-2.5-flash-image (default — used by edit_image)"],
            "video_generation": list(VIDEO_MODEL_ALIASES.keys()),
        }

    @mcp.tool()
    async def list_projects() -> dict:
        """Workspace project folders that tools can save into (the 'project' parameter)."""
        try:
            dirs = sorted(d for d in os.listdir(cfg.workspace_root)
                          if os.path.isdir(os.path.join(cfg.workspace_root, d)) and not d.startswith("."))
        except OSError as e:
            return {"error": f"workspace not reachable: {e}", "projects": []}
        return {"projects": dirs}
```

Add a public `running_count()` to `JobStore` instead of poking `_jobs` (cleaner — do it):

```python
    def running_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["status"] == "running")
```
and use `jobs.running_count()` in `forge_status`.

- [ ] **Step 8.6: Wire into `forge_mcp/server.py`**

Add after the middleware/route definitions (and import `asyncio`, `contextlib`, `httpx`, `types`, plus `from forge_mcp import storage`, `from forge_mcp.jobs import JobStore`, `from forge_mcp import engine`, `from forge_mcp.tools import register_all`):

```python
ctx = types.SimpleNamespace(
    cfg=cfg,
    jobs=JobStore(os.path.join(cfg.results_root, "jobs.json")),
    http=None,           # created inside lifespan
    poll_and_finish=None,  # set by tools.gen.register
)
register_all(mcp, ctx)


def build_app():
    app = mcp.streamable_http_app()
    app.router.routes.append(Route("/health", health, methods=["GET"]))
    app.router.routes.append(Route("/files/{file_id}/{name}", serve_file, methods=["GET"]))

    original_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app_):
        async with original_lifespan(app_):
            ctx.http = httpx.AsyncClient(timeout=120)
            background = [asyncio.create_task(storage.janitor_loop(cfg)),
                          asyncio.create_task(engine.preload_default_model())]
            for job in ctx.jobs.mark_interrupted():  # resume Veo polls that survived restart
                background.append(asyncio.create_task(
                    ctx.poll_and_finish(job["id"], job["operation_name"])))
            try:
                yield
            finally:
                for t in background:
                    t.cancel()
                await ctx.http.aclose()

    app.router.lifespan_context = lifespan
    return BearerAuthMiddleware(app, cfg.token)
```

Note `_run_veo_job`/`_poll_and_finish` failure inside a resumed task must also mark the job failed — wrap the resume task: create a small `async def _resume(job)` in server.py that try/excepts and calls `ctx.jobs.update(job["id"], status="failed", error=str(e))` on exception. Implement that wrapper.

- [ ] **Step 8.7: Full local smoke test (real engine, temp workspace)**

```powershell
cd "C:\Programming\Carbon Forge"
mkdir $env:TEMP\forge-ws\TestProj -Force
$env:FORGE_TOKEN="testtok"; $env:FORGE_WORKSPACE_ROOT="$env:TEMP\forge-ws"; $env:FORGE_RESULTS_ROOT="$env:TEMP\forge-results"; $env:FORGE_PUBLIC_URL="http://127.0.0.1:5125"
python -m forge_mcp.server
```
(background). Write a one-shot client `tests/manual_client.py`:

```python
"""Manual smoke client: python tests/manual_client.py <image-url-or-path>"""
import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    image = sys.argv[1] if len(sys.argv) > 1 else "https://picsum.photos/400"
    headers = {"Authorization": f"Bearer {os.environ.get('FORGE_TOKEN', 'testtok')}"}
    async with streamablehttp_client("http://127.0.0.1:5125/mcp", headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", sorted(t.name for t in tools.tools))
            res = await s.call_tool("forge_status", {})
            print("STATUS:", res.content[0].text)
            res = await s.call_tool("remove_background",
                                    {"image": image, "project": "TestProj"})
            print("REMOVE_BG:", res.content[0].text)
            out = json.loads(res.content[0].text)
            res = await s.call_tool("trim_image",
                                    {"image": out["url"], "project": "TestProj"})
            print("TRIM:", res.content[0].text)


asyncio.run(main())
```

Run: `python tests/manual_client.py` →
Expected: TOOLS lists all 15 (`remove_background, split_sprites, trim_image, remove_watermark, remove_colors, smooth_edges, generate_image, edit_image, generate_video, job_status, video_trim, video_extract_frames, video_convert, forge_status, list_models, list_projects` — 16 with list_projects), STATUS shows `workspace_writable: true`, REMOVE_BG returns a result dict whose `workspace_path` exists on disk and whose `url` returns `image/png` via `Invoke-WebRequest`. First call downloads u2net (~170 MB) — allow a couple of minutes.
(ffmpeg ops need ffmpeg on PATH locally — if absent, skip the local video smoke; it's covered in the container verify, Task 10.)

- [ ] **Step 8.8: Commit**

```bash
git add forge_mcp/ tests/manual_client.py
git commit -m "feat(mcp): full tool surface — processing, generation, video, meta + lifecycle wiring"
git push
```

---

## Task 9: Dockerfile + compose + .env template

**Files:**
- Create: `Dockerfile.mcp`, `docker-compose.forge.yml`, `.env.forge.example`, `.dockerignore`

- [ ] **Step 9.1: `Dockerfile.mcp`**

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-mcp.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt

COPY backend/__init__.py backend/processing.py backend/
COPY forge_mcp/ forge_mcp/

ENV U2NET_HOME=/models/u2net \
    FORGE_LAMA_DIR=/models \
    PYTHONUNBUFFERED=1

EXPOSE 5125
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:5125/health || exit 1

CMD ["python", "-m", "forge_mcp.server"]
```

- [ ] **Step 9.2: `docker-compose.forge.yml`**

```yaml
name: carbon-forge

services:
  forge:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    image: carbon-forge-mcp
    ports:
      - "5125:5125"
    env_file: .env
    volumes:
      - forge-models:/models
      - forge-results:/results
      - workspace:/workspace
    restart: unless-stopped

volumes:
  forge-models:
  forge-results:
  workspace:
    driver: local
    driver_opts:
      type: cifs
      o: "username=${CIFS_USERNAME},password=${CIFS_PASSWORD},vers=3.0,addr=192.168.0.35,file_mode=0666,dir_mode=0777"
      device: "//192.168.0.35/Workspace"
```

- [ ] **Step 9.3: `.env.forge.example`** (the real `.env` lives only on laybackrig)

```
FORGE_TOKEN=            # bearer token the gateway uses; generate 32+ random chars
GEMINI_API_KEY=         # Google AI Studio key (Imagen/Gemini/Veo billing)
CIFS_USERNAME=forge-svc
CIFS_PASSWORD=          # carbonserver local account password for the Workspace share
FORGE_PUBLIC_URL=https://forge.carbonrouting.dev
FORGE_CACHE_TTL_DAYS=30
FORGE_MAX_IMAGE_MB=50
FORGE_MAX_VIDEO_MB=500
```

And `.dockerignore`:

```
node_modules
dist
build
renderer
docs
.git
*.png
*.ico
backend/models
```

Also append to `.gitignore`: a line `.env` (the repo has no .env today; the forge .env must never be committed).

- [ ] **Step 9.4: Local container build + smoke (Docker Desktop on the dev PC — start it if stopped; check `docker info` first)**

```powershell
cd "C:\Programming\Carbon Forge"
docker build -f Dockerfile.mcp -t carbon-forge-mcp .
# run WITHOUT the cifs volume locally (override workspace with a bind mount):
mkdir $env:TEMP\forge-ws2\TestProj -Force
docker run --rm -d --name forge-smoke -p 5126:5125 -e FORGE_TOKEN=testtok -e FORGE_PORT=5125 -v "$env:TEMP\forge-ws2:/workspace" carbon-forge-mcp
Start-Sleep 5; Invoke-WebRequest http://127.0.0.1:5126/health   # expect 200
docker logs forge-smoke --tail 20                                # expect uvicorn startup, no tracebacks
docker stop forge-smoke
```

- [ ] **Step 9.5: Commit**

```bash
git add Dockerfile.mcp docker-compose.forge.yml .env.forge.example .dockerignore .gitignore
git commit -m "feat(mcp): containerization — Dockerfile, compose with CIFS workspace volume, env template"
git push
```

---

## Task 10: carbonserver SMB share + laybackrig deploy

**Files:** none (infra). All remote work = one script per box per the shell rules.

- [ ] **Step 10.1: Create the Workspace share on carbonserver**

Generate a strong password locally first (`$pw = -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 24 | % {[char]$_})` — keep it for Step 10.2). Then ONE remote script:

```powershell
ssh rober@192.168.0.35 "powershell -NoProfile -Command \"& { `$p = ConvertTo-SecureString '<PW>' -AsPlainText -Force; if (-not (Get-LocalUser forge-svc -ErrorAction SilentlyContinue)) { New-LocalUser -Name forge-svc -Password `$p -PasswordNeverExpires -Description 'Carbon Forge CIFS' } ; if (-not (Get-SmbShare Workspace -ErrorAction SilentlyContinue)) { New-SmbShare -Name Workspace -Path C:\Workspace -FullAccess forge-svc } ; icacls C:\Workspace /grant 'forge-svc:(OI)(CI)M' }\""
```
Verify: `ssh rober@192.168.0.35 "powershell -NoProfile -Command Get-SmbShare Workspace | Format-List"` → share exists, path `C:\Workspace`.

- [ ] **Step 10.2: Clone + configure on laybackrig**

```
ssh 192.168.0.177 "git clone https://github.com/CarbonNode/Carbon-Forge.git C:\Programming\CarbonForge"
```
Generate `FORGE_TOKEN` locally (32 chars, same Get-Random technique). Write the `.env` remotely WITHOUT BOM (PS 5.1 gotcha — use the WriteAllText pattern from the laybackrig notes), filling FORGE_TOKEN, GEMINI_API_KEY (ask the user for the key — the desktop app has one in its settings; do NOT guess), CIFS_USERNAME=forge-svc, CIFS_PASSWORD=<PW from 10.1>.
Check Docker is up first: `ssh 192.168.0.177 "docker info"` — if stopped: `sc start com.docker.service`, then `docker desktop start`, wait 1-2 min.

- [ ] **Step 10.3: Build via one-shot scheduled task (credential-helper gotcha)**

Create `C:\Programming\CarbonForge\build-forge.bat` on laybackrig:

```bat
@echo off
cd /d C:\Programming\CarbonForge
docker compose -f docker-compose.forge.yml build forge >> build-forge.log 2>&1
if %errorlevel%==0 (echo BUILD_OK >> build-forge.log) else (echo BUILD_FAIL >> build-forge.log)
docker compose -f docker-compose.forge.yml up -d forge >> build-forge.log 2>&1
echo DEPLOY_DONE >> build-forge.log
```

```
ssh 192.168.0.177 "schtasks /create /tn ForgeBuild /tr C:\Programming\CarbonForge\build-forge.bat /sc ONCE /st 23:59 /f && schtasks /run /tn ForgeBuild"
```
Wait a few minutes (base pull + pip install ≈ 5-10 min), then check ONCE directly (no poll loops):
```
ssh 192.168.0.177 "type C:\Programming\CarbonForge\build-forge.log | findstr /C:BUILD_OK /C:BUILD_FAIL /C:DEPLOY_DONE"
ssh 192.168.0.177 "schtasks /delete /tn ForgeBuild /f"
```

- [ ] **Step 10.4: Verify service + CIFS write end-to-end**

From the dev PC:
```powershell
Invoke-WebRequest http://192.168.0.177:5125/health      # 200 {"status":"ok"}
```
Then call `forge_status` via MCP (reuse `tests/manual_client.py` pointed at `http://192.168.0.177:5125/mcp` with the real FORGE_TOKEN) → expect `workspace_writable: true`, `ffmpeg_available: true`, `gemini_key_configured: true`. If `workspace_writable` is false: the CIFS volume is the issue — check `docker volume inspect carbon-forge_workspace`, confirm the WSL2 kernel mounts cifs (`docker run --rm -v carbon-forge_workspace:/w alpine ls /w`); fallback documented in the spec is an in-container `mount.cifs` with `cap_add: [SYS_ADMIN]` — only if the volume driver path fails.

---

## Task 11: Public URL — tunnel ingress for forge.carbonrouting.dev

- [ ] **Step 11.1:** Consult the memory file `reference_cloudflare-tunnel-setup.md` (auto-memory dir) for the exact cloudflared-gateway config location + restart steps on carbonserver. Read the current config (`type` it over SSH), append an ingress rule **above the catch-all**:

```yaml
  - hostname: forge.carbonrouting.dev
    service: http://192.168.0.177:5125
```

- [ ] **Step 11.2:** Create the DNS CNAME `forge.carbonrouting.dev → <tunnel-id>.cfargotunnel.com` (same mechanism as the existing gateway record — the cloudflare connector token or `cloudflared tunnel route dns` per the memory doc), then `docker restart cloudflared-gateway`.

- [ ] **Step 11.3: Verify**

```powershell
Invoke-WebRequest https://forge.carbonrouting.dev/health    # 200 {"status":"ok"}
# and a /files URL from a Task 10 result serves image/png
```

---

## Task 12: Gateway registration (Carbon-Cortex repo)

**Files:**
- Create: `C:\Programming\Carbon Cortex\scripts\register-forge-connector.mjs`

- [ ] **Step 12.1: Write the register script** (pattern: `scripts/register-memory-connector.mjs`; adds the `media` category WITH grants — root-cause fix for the documented "media had 0 tools" failure)

```javascript
// Registers the `forge` mcp-proxy connector (Carbon Forge on laybackrig) for the Carbon org.
// Run on carbonserver:  cd "C:\Programming\carbon-cortex" && FORGE_TOKEN=... node scripts/register-forge-connector.mjs
// Idempotent: re-runnable; replaces any existing `forge` row.
//
// Creates the `media` tool category AND grants it to owner/admin roles (canAccess is
// deny-by-default; a category without permission rows filters every tool to zero).

import { getPool, addConnector, removeConnector } from '../src/db/gateway-db.ts';

const OWNER_ROLE = '00000000-0000-0000-0000-000000000001';
const ADMIN_ROLE = '00000000-0000-0000-0000-000000000002';

async function main() {
  const forgeToken = process.env.FORGE_TOKEN;
  if (!forgeToken) { console.error('[register] Set FORGE_TOKEN (the forge service bearer token).'); process.exitCode = 1; return; }

  const pool = getPool();
  const { rows } = await pool.query('SELECT * FROM organizations WHERE slug = $1', [process.env.ORG_SLUG || 'carbon']);
  const org = rows[0];
  if (!org) { console.error('[register] No carbon org.'); process.exitCode = 1; return; }

  await pool.query(
    `INSERT INTO tool_categories (id, name, description)
     VALUES ('media', 'Media', 'Carbon Forge — image generation/editing, video generation, processing')
     ON CONFLICT (id) DO NOTHING`);
  for (const role of [OWNER_ROLE, ADMIN_ROLE]) {
    await pool.query(
      `INSERT INTO permissions (role_id, scope_type, scope_value, access_level)
       VALUES ($1, 'category', 'media', 'admin')
       ON CONFLICT (role_id, scope_type, scope_value) DO UPDATE SET access_level = EXCLUDED.access_level`,
      [role]);
  }
  console.log('[register] media category + owner/admin grants ensured');

  await removeConnector(org.id, 'forge').catch(() => {});
  const created = await addConnector(
    org.id,
    'forge',
    'mcp-proxy',
    { bearerToken: forgeToken },
    { url: 'http://192.168.0.177:5125/mcp', transport: 'streamable', category: 'media' },
    'Carbon Forge — generate images (Imagen 4), edit with reference images (Gemini), generate video (Veo), remove backgrounds, split sprites, trim, remove watermarks, ffmpeg video ops. Results land in C:\\Workspace\\<project> + a shareable URL.',
    'generate image, create image, edit image, reference image, generate video, veo, imagen, remove background, transparent, sprite, watermark, trim image, trim video, extract frame, convert video, gif',
    org.encryption_key_wrapped,
  );
  console.log(`[register] Inserted ${created.name} (${created.type}) — id ${created.id}`);
  await pool.end();
}

main().catch((err) => { console.error('[register] Failed:', err); process.exitCode = 1; });
```

- [ ] **Step 12.2:** Commit to Carbon-Cortex, push, `git pull` on carbonserver (`C:\Programming\carbon-cortex`), run it the same way the previous register scripts were run there (check `node --version` ≥ 23 for .ts imports; otherwise run with `npx tsx`). Pass the real `FORGE_TOKEN`.
Expected output: `media category + owner/admin grants ensured`, `Inserted forge (mcp-proxy)`.

- [ ] **Step 12.3: Verify through the gateway** — `/api/connectors` shows `forge` with 16 tools (not 0 — if 0, the category grant didn't apply; recheck Step 12.1 SQL), and from a Claude session on the gateway run `forge__forge_status`. Sessions pick it up on next connect; no gateway restart needed.

---

## Task 13: Console UI wiring (Carbon-Cortex repo, mandatory per CLAUDE.md)

**Files:**
- Modify: `web/src/pages/Connectors.tsx` (HOME_APP_GROUPS ~line 89, `formatConnectorName` labels ~line 101)
- Modify: `web/src/lib/connector-icons.ts`
- Create: `web/public/icons/carbon-forge.png`

- [ ] **Step 13.1:** Icon: downscale `C:\Programming\Carbon Forge\Carbon Forge Icon.png` to 256×256 PNG (Pillow one-liner) → save as `web/public/icons/carbon-forge.png` in the Carbon-Cortex repo.

- [ ] **Step 13.2:** `Connectors.tsx` — add to `HOME_APP_GROUPS`: `forge: 'Carbon Systems',` (first-party app; type=mcp-proxy so it must be name-routed). Add to the `labels` map in `formatConnectorName`: `forge: 'Carbon Forge',`.

- [ ] **Step 13.3:** `connector-icons.ts` — add under the Carbon Systems block (first-party = full color, no greyscale):

```typescript
  forge:            { icon: '/icons/carbon-forge.png',    color: '#E8590C', noFilter: true },
```

- [ ] **Step 13.4:** Deploy the console via the FAST path (no docker build): scp the icon + changed files to carbonserver, `npm run build` on the host, `docker cp` dist into `carbon-cortex-cortex-console-1`, copy `carbon-forge.png` into the running container's icons dir too, then `purge_everything` on zone `cd3f1e2ac516d67f6c638610bcc45928`. Verify `curl http://localhost:3500/icons/carbon-forge.png` on carbonserver → `image/png` at real byte size. ⚠️ docker cp is layer-only — note that the next full console rebuild bakes it in (the files are committed in git, so any future image build includes them).

- [ ] **Step 13.5:** Commit + push (Carbon-Cortex). Check `/connectors` page on a phone width too (no layout change expected — existing card grid).

---

## Task 14: End-to-end verification + live generation test

- [ ] **Step 14.1:** From a fresh Claude session (or this one once tools reload): `forge__forge_status` → all green. `forge__list_projects` → real workspace folders.
- [ ] **Step 14.2:** `forge__generate_image` (model `imagen-4-fast`, count 1, project e.g. `Carbon Forge`) → expect file at `C:\Workspace\Carbon Forge\assets\forge\…png` (verify over SSH) + URL serves `image/png` publicly. (~$0.02 — fine to run without asking.)
- [ ] **Step 14.3:** `forge__edit_image` with the Step 14.2 output as a reference image + a simple instruction → image returned.
- [ ] **Step 14.4:** **ASK THE USER before this one (real cost, ~$1-3):** `forge__generate_video` veo-3-fast, 8 s → poll `forge__job_status` until done → then `forge__video_trim` the result to 0–3 s and `forge__video_extract_frames` at 1 s. Verify files in the workspace.
- [ ] **Step 14.5:** `forge__remove_background` on a URL input; `forge__split_sprites` on any sheet-like image. Spot-check audit log shows the calls (`gateway recent_activity`).

---

## Task 15: Documentation + memory

- [ ] **Step 15.1 (Carbon-Forge repo):** Create `CLAUDE.md` documenting: repo layout (desktop app + `forge_mcp` service), the shared `backend/processing.py` contract ("desktop and MCP service share this module — never fork the logic"), deploy runbook (laybackrig path, scheduled-task build, compose file, .env keys, CIFS mount), verify commands, and the `forge` connector relationship. Also update the README if one is added later. Commit.
- [ ] **Step 15.2 (Carbon-Cortex repo):** `docs/features/README.md` — add `forge` row to the Connectors table (16 tools, laybackrig, mcp-proxy) per the Documentation Discipline rule. Commit (same commit as Task 12/13 if not yet pushed, otherwise its own).
- [ ] **Step 15.3:** Update the user CLAUDE.md "Carbon Cortex" connector table is auto-maintained? No — it's manual: add a `forge` row to the connectors table in `C:\Users\rober\.claude\CLAUDE.md` and a line in the laybackrig section (forge container, port 5125, compose at `C:\Programming\CarbonForge`).
- [ ] **Step 15.4:** Write project memory `project_carbon-forge-mcp.md` (auto-memory dir): what forge is, where it runs, the FORGE_TOKEN/.env location, CIFS share + forge-svc account, tunnel hostname, register script. Add MEMORY.md index line.

---

## Self-review (done at plan time)

- **Spec coverage:** all 16 tools (15 spec + list_projects addition) ✓; URL+workspace inputs ✓; dual-write + degradation ✓; Veo async jobs + restart resume ✓; ffmpeg ops ✓; auth/bearer + unguessable /files ✓; janitor/TTL ✓; size caps + magic sniffing ✓; desktop refactor with byte-identical behavior (incl. the color-auto-detect quirk) ✓; CIFS volume + SMB share ✓; scheduled-task build ✓; tunnel ingress ✓; register script with category grants ✓; console UI wiring ✓; docs ✓.
- **Placeholders:** none — every module has full code; infra steps have exact commands.
- **Type consistency:** `PipelineOptions` field names match between processing.py, proc.py; `save_result` signature/return keys match all callers; `JobStore` API (`create/update/get/mark_interrupted/running_count`) matches gen.py + meta.py + server.py usage; `ctx` members (`cfg/jobs/http/poll_and_finish`) consistent.
- **Known risks called out:** SDK DNS-rebinding behavior (spike in 2.5), CIFS-on-WSL2 (verify in 10.4 with fallback), Node .ts-import for register script (12.2), Veo cost gate (14.4).
