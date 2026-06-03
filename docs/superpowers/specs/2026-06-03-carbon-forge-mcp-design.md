# Carbon Forge MCP Service — Design

**Date:** 2026-06-03
**Status:** Approved (pending spec review)
**Repos touched:** `CarbonNode/Carbon-Forge` (service), `CarbonNode/Carbon-Cortex` (gateway integration only)

## Goal

Host Carbon Forge's engine as a self-contained Docker stack on **laybackrig** (192.168.0.177) that
**speaks MCP directly**, so any Claude session (via the Carbon Cortex gateway) can generate images,
edit with reference images, generate video, and run the full local processing suite (background
removal, sprite split, trim, watermark removal, edge cleanup) plus basic ffmpeg video ops.
Results land as real files inside Conduit project workspaces on carbonserver.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Image I/O | URL in / hosted URL out; inputs also accepted as workspace-relative paths |
| Generation scope | Full parity: Imagen 4 family, Gemini image edit (reference images first-class), Veo 2/3 |
| Video ops | Yes — new ffmpeg tools: trim, extract frames, convert |
| Result storage | Conduit project folders: `C:\Workspace\<Project>\` on carbonserver, **configurable root** |
| Destination contract | `project` param + optional `subpath` (default `assets/forge/`) |
| Topology | Whole service = one Docker compose stack on laybackrig; it IS an MCP server; gateway proxies to it |
| GPU | CPU onnxruntime initially; `onnxruntime-gpu` documented as opt-in upgrade (shares 4090 with Ollama) |

## Architecture

```
┌─ laybackrig (192.168.0.177) ── compose stack "carbon-forge" ─────────┐
│  forge container (Python 3.12, port 5125)                            │
│   ├─ FastMCP server, streamable-HTTP at /mcp (bearer-token auth)     │
│   ├─ processing engine: rembg (5 models) + LaMa ONNX + scipy ops     │
│   │   (extracted from backend/server.py into backend/processing.py)  │
│   ├─ generation client: Imagen 4 / Gemini image / Veo                │
│   │   (ported from main.js; GEMINI_API_KEY in container env)         │
│   ├─ ffmpeg (in image) for video ops                                 │
│   ├─ GET /files/<id>/<name> — unauthenticated, unguessable id        │
│   ├─ GET /health                                                     │
│   └─ volumes:                                                        │
│       forge-models   → /models        (rembg + LaMa cache)           │
│       forge-results  → /results       (served cache + job state)     │
│       cifs workspace → /workspace     (\\192.168.0.35\Workspace)     │
└───────────────────────────────────────────────────────────────────────┘
         ▲ MCP over LAN (http://192.168.0.177:5125/mcp)        ▲ HTTPS
┌─ carbonserver ──────────────────────┐      forge.carbonrouting.dev
│ gateway: mcp-proxy connector         │      (new ingress rule on the
│ "carbon-forge" → carbon-forge__*     │       existing cloudflared-gateway
│ tools for all users, audit-logged    │       tunnel → 192.168.0.177:5125)
└──────────────────────────────────────┘
```

### Components

1. **`backend/processing.py` (refactor, Carbon-Forge repo)** — pure functions extracted from
   `server.py`: `remove_watermark`, `remove_colors`, `smooth_edges`, `trim_transparent`,
   `detect_bg_color`, `split_sprites`, rembg session management. `server.py` (the desktop app's
   bundled backend) imports from it — **desktop app behavior unchanged**, PyInstaller spec updated
   to include the new module.
2. **`mcp/` (new, Carbon-Forge repo)** — the hosted service:
   - `mcp/server.py` — FastMCP app: tool definitions, auth middleware, `/files` + `/health` routes.
   - `mcp/generation.py` — Google API client: `call_gemini_image`, `call_imagen`, `start_veo`,
     `poll_veo`, `download_veo_video` (direct port of main.js logic incl. retry/backoff on 429/5xx).
   - `mcp/storage.py` — input resolution (URL download / workspace read), output writing
     (workspace + results cache), URL minting, filename collision handling (`name-2.png`),
     cache janitor.
   - `mcp/jobs.py` — Veo job registry persisted to `/results/jobs.json` (survives restarts).
   - `mcp/video.py` — ffmpeg wrappers (trim, frames, convert) with timeout + stderr capture.
   - `Dockerfile.mcp`, `docker-compose.forge.yml`.
3. **Gateway integration (Carbon-Cortex repo + carbonserver config)** — mcp-proxy connector row in
   the gateway DB (via `addConnector()` inside the gateway container, the established pattern);
   `/connectors` UI wiring: category in `groupDefs`/`HOME_APP_GROUPS`, greyscale icon in
   `connector-icons.ts` + `web/public/icons/carbon-forge.png`, display name override. Tools
   registered under an **already-granted permission category** (deny-by-default `canAccess` gotcha).
4. **carbonserver one-time setup** — SMB share: `New-SmbShare -Name Workspace -Path C:\Workspace`
   restricted to a dedicated local account used only by the forge mount. Tunnel: add
   `forge.carbonrouting.dev → http://192.168.0.177:5125` ingress to cloudflared-gateway config +
   CNAME in the carbonrouting.dev zone.

## I/O contract

**Inputs** — every image/video parameter accepts either:
- `https://…` URL (downloaded server-side, 60 s timeout, size-capped), or
- workspace-relative path `<Project>/<relative path>` (read from `/workspace`).

**Outputs** — every producing tool takes `project` (must match an existing `/workspace/<Project>`
folder; clear error listing close matches if not) and optional `subpath` (default `assets/forge/`)
and `filename` (default derived from prompt/op + timestamp). Returns per file:

```json
{
  "workspace_path": "C:\\Workspace\\<Project>\\assets\\forge\\<name>.png",
  "url": "https://forge.carbonrouting.dev/files/<24-char-id>/<name>.png",
  "bytes": 123456,
  "width": 1024,
  "height": 1024
}
```

Files are written to **both** the workspace (canonical) and the results cache (URL serving).
If the CIFS mount is unavailable, the tool still succeeds with `url` + a `workspace_write_error`
field explaining the failure — no silent loss. Results cache entries expire after
`FORGE_CACHE_TTL_DAYS` (default 30, daily janitor).

## Tool surface

### Generation (Google API)

| Tool | Parameters | Notes |
|---|---|---|
| `generate_image` | `prompt`, `model` = `imagen-4` \| `imagen-4-fast` \| `imagen-4-ultra` (mapped to `imagen-4.0-{,fast-,ultra-}generate-001`), `count` 1–4 (ultra: 1), `aspect_ratio` (`1:1`,`3:4`,`4:3`,`9:16`,`16:9`), `project`, `subpath?`, `filename?` | `:predict`, `personGeneration: allow_adult` |
| `edit_image` | `prompt`, `reference_images[]` (1–6, URL or workspace path), `project`, … | `gemini-2.5-flash-image` `:generateContent`, refs as `inlineData` parts before the text part |
| `generate_video` | `prompt`, `model` = `veo-3` \| `veo-3-fast` \| `veo-2`, `start_image?`, `aspect_ratio` (`16:9`,`9:16`), `duration_seconds` (default 8), `generate_audio` (veo-3 only), `project`, … | `:predictLongRunning` → returns `{job_id}` immediately |
| `job_status` | `job_id` | running → progress; done → result object(s); failed → readable error (safety-filter detection per main.js) |

### Processing (local engine — behavior identical to the desktop app)

| Tool | Parameters |
|---|---|
| `remove_background` | `image`, `model` (u2net default, u2netp, u2net_human_seg, isnet-general-use, silueta), `alpha_matting` + thresholds/erode, `color_remove` + `colors[]`/`auto_detect`/`tolerance`, `edge_smooth` + `strength`/`trim_px`, `auto_trim`, watermark pre-step flags, `project`, … |
| `split_sprites` | `image`, full pipeline flags, `min_sprite_area` (default 400) → N files, returns array of result objects |
| `trim_image` | `image` → transparent-edge trim only |
| `remove_watermark` | `image`, `position` (6 anchors), `size_pct` (default 15) |
| `remove_colors` | `image`, `colors[]` (hex or RGB) or `auto_detect`, `tolerance` |
| `smooth_edges` | `image`, `strength`, `trim_px` |

### Video (ffmpeg)

| Tool | Parameters |
|---|---|
| `video_trim` | `video`, `start`, `end` (HH:MM:SS.ms or seconds), stream-copy when possible, re-encode fallback |
| `video_extract_frames` | `video`, `timestamps[]` OR `fps`, `format` (png/jpg) → N files |
| `video_convert` | `video`, `format` (mp4/webm/gif), `crf?`, `scale?` (e.g. `720`, `1080`) |

### Meta

| Tool | Returns |
|---|---|
| `forge_status` | loaded rembg models, LaMa state, ffmpeg version, workspace mount ok?, results cache size, pending Veo jobs |
| `list_models` | rembg models + generation model aliases with descriptions |

## Security & auth

- **MCP endpoint**: bearer token (`FORGE_TOKEN` env) checked by middleware; the gateway's mcp-proxy
  connector stores it server-side. FastMCP DNS-rebinding host check configured for
  `forge.carbonrouting.dev` + LAN IP (same knob as cortex-memory).
- **`/files`**: unauthenticated but 24-char-random ids (cdrop model — link = capability).
- **`/health`**: unauthenticated, no secrets.
- **CIFS credentials**: dedicated carbonserver local account, granted only on the `Workspace`
  share; password only in laybackrig compose `.env` (gitignored).
- **GEMINI_API_KEY**: laybackrig `.env` (gitignored). Deviation from the "all credentials on
  carbonserver" convention, accepted for self-containment.

## Error handling

- Google 429/5xx → retry ×3 with backoff (port of `fetchJson`); 4xx → readable tool error with the
  API's message (e.g. quota, safety block) — never a stack trace.
- Veo "finished with no samples" → explicit "likely safety-filtered" message (parity with main.js).
- Input caps: images 50 MB, videos 500 MB (env-tunable); URL fetch timeout 60 s; non-image bytes
  rejected by sniffing, not extension.
- ffmpeg ops run with a 10-min timeout; on failure return trimmed stderr tail.
- Unknown `project` → error listing nearest existing workspace folder names.
- CIFS down → succeed with cache URL + `workspace_write_error` (see I/O contract).

## Deploy

1. **Carbon-Forge repo**: refactor + `mcp/` + Dockerfile + compose, committed and pushed.
2. **laybackrig**: clone to `C:\Programming\CarbonForge`; `.env` (FORGE_TOKEN, GEMINI_API_KEY,
   CIFS creds); image build via the **one-shot scheduled task** pattern (Docker credential-helper
   gotcha; rober stays logged in); `docker compose -f docker-compose.forge.yml up -d`;
   `restart: unless-stopped`. Note: Docker Desktop idle-pauses with no containers — forge running
   24/7 keeps it warm.
3. **carbonserver**: create the SMB share + service account; add tunnel ingress + DNS CNAME;
   add mcp-proxy connector row; deploy `/connectors` UI wiring per the web-console fast-deploy path.
4. **Verify end-to-end**: `forge_status` through the gateway from a fresh session; one
   `generate_image` → file appears in `C:\Workspace\<proj>\assets\forge\` and URL serves PNG;
   one `remove_background` from a URL input; one `video_trim` on a sample clip.

## Testing

- Unit tests for `storage.py` (input resolution, collision naming, cache janitor) and `jobs.py`
  (persistence round-trip) — pytest in the repo, run in CI-less mode locally.
- Engine functions are already proven in production desktop use; the refactor is import-moves only,
  verified by running the desktop backend (`python backend/server.py`) and hitting `/health` +
  one `/remove-bg` smoke call.
- Generation client tested live with the real key (1 fast Imagen call, 1 short Veo-fast call) —
  there is no meaningful mock for these.

## Out of scope (explicit)

- GPU acceleration (documented upgrade: `onnxruntime-gpu` + WSL2 CUDA passthrough).
- Desktop app switching to the hosted engine (it keeps its bundled local backend).
- Image editing UI / gallery on top of the results cache.
- Audit/billing dashboards for Google API spend (gateway audit log covers call history).
