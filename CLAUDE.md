# Carbon Forge

AI-powered asset generation & refinement. Repo: `CarbonNode/Carbon-Forge`. Two deliverables share one engine:

1. **Desktop app** — Electron (`main.js`, `renderer/`) + bundled Python Flask backend (`backend/server.py`, PyInstaller via `npm run build-backend`). Runs locally on port 5123.
2. **Hosted MCP service** — `forge_mcp/` package, Docker container on **super-server** (192.168.0.197:5125; split 2026-07-01 — the API/orchestrator lives here, GPU generation workers stay on laybackrig/maingamingrig and are dispatched via `FORGE_*_URL`). Proxied by the Carbon Cortex gateway as connector **`forge`** (tools surface as `forge__*` in every gateway session).

## The one shared-engine rule

**`backend/processing.py` is the single image engine** (rembg pipelines, LaMa watermark removal, color/edge ops, sprite splitting). Both `backend/server.py` (desktop) and `forge_mcp/engine.py` (hosted) import it. **Never fork the logic** — change `processing.py` and both stay in sync. `server.py` is only Flask routing; `forge_mcp` is only MCP plumbing around the same functions.

## Layout

```
backend/
  processing.py   # THE engine — pure functions, PipelineOptions, run_pipeline, run_split_pipeline
  server.py       # Desktop Flask wrapper (port 5123); PyInstaller entry
forge_mcp/        # Hosted MCP service (named forge_mcp, NOT mcp — would shadow the pip `mcp` package)
  server.py       # FastMCP assembly: bearer auth on /mcp, /health, /files/<id>/<name>, lifespan
  config.py       # FORGE_* env config
  storage.py      # input resolution (URL | '<Project>/<path>'), dual-write results, URL minting, janitor
  generation.py   # Imagen 4 / Gemini image / Veo + ElevenLabs TTS + local ComfyUI (SDXL/Flux/Wan/ESRGAN)
  jobs.py         # persistent Veo job registry (/results/jobs.json), restart resume
  video.py        # ffmpeg wrappers: video trim/frames/convert + audio convert/trim + ffprobe
  imaging.py      # Pillow format conversion (image_convert — plain convert/resize, no AI)
  assets3d.py     # GLB helpers: Draco compression (gltf-transform CLI, node in image) + stats
  engine.py       # async bridge to backend.processing (CPU semaphore, model-load lock)
  tools/          # MCP tool definitions: proc, gen, vid, audio, extract, util, meta (incl. local Wan T2V/I2V, ESRGAN upscale, IPAdapter reference gen, saved characters, audio TTS, batch/montage, generate_clip pipeline)
                  #   audio.py = generate_speech / list_voices. TWO TTS providers:
                  #   ElevenLabs (cloud) + Chatterbox (local, isolated GPU container, see below)
                  #   util.py = quick conversions: audio_convert / audio_trim / image_convert /
                  #   draco_compress (.glb) / media_info — chat uploads are workspace paths
                  #   ('<Project>/.conduit/uploads/<name>'), so they feed straight in
tests/            # pytest (26 tests) + manual_* live-smoke clients
Dockerfile.mcp, docker-compose.forge.yml, .env.forge.example
```

## Hosted service — how it works

- **I/O contract:** every tool takes inputs as https URLs or workspace paths `<Project>/<relative path>`; outputs are written BOTH to `\\carbonserver\Workspace\<project>\assets\forge\` (CIFS volume → Conduit project folders) AND a results cache served at `https://forge.carbonrouting.dev/files/<id>/<name>`. CIFS down ⇒ tool still succeeds with `workspace_write_error` + the cache URL.
- **Auth:** bearer token (`FORGE_TOKEN`) on `/mcp`; `/health` + `/files` open (unguessable ids).
- **Veo videos** are async: `generate_video` → `job_id`, poll `job_status`. Jobs persist across restarts (in-flight operations resume).
- **Generation needs `GEMINI_API_KEY`** in the service `.env`; without it those tools return a readable error and everything else works.

## Deploy (hosted service) — one call, automated since 2026-07-28

```
commit + push to origin/main, then:  deployer__ship {project: "carbon-forge"}
```

That's the whole deploy. The org deployer (carbon-cortex gateway tool, registered in
`config/deploy-registry.json`) runs the standard compose-build recipe on super-server:
deploy-coordinator lease → dirty-tree guard + `git pull --ff-only` on
`C:\Programming\CarbonForge-src` → interactive scheduled-task `docker compose build forge`
(sidesteps the Windows credential-helper hang) → `up -d --no-deps forge` → health check →
release. Poll `deployer__ship_status {job_id}` until ok/failed. Dry-run with
`deployer__explain {project: "carbon-forge"}`.

Topology on **super-server** (192.168.0.197):
- `C:\Programming\CarbonForge-src` — the git checkout (tracks `origin/main`) that OWNS the
  stack: the repo's root `docker-compose.yml` (project `carbon-forge`, service `forge`,
  container `carbon-forge-adhoc`) runs from here. `.env` sits beside it (gitignored:
  FORGE_TOKEN, GEMINI keys, ELEVENLABS, CIFS creds, `FORGE_COMFY_URL`/`FORGE_CHATTERBOX_URL`
  etc. → laybackrig/maingamingrig GPU workers; the CIFS `workspace` volume interpolates
  `CIFS_USERNAME`/`CIFS_PASSWORD` from it). Never hand-edit the checkout — push, then ship.
- `C:\Programming\CarbonForge` — legacy runtime dir; keeps the master `.env` backup and the
  superseded pre-adoption compose. Nothing runs from it anymore.

Notes: a recreate drops forge MCP sessions for a few seconds (they reconnect), and gateway
MCP sessions opened BEFORE the deploy keep serving the old forge tool schemas (new tool
params get silently stripped) until they reconnect — new sessions are fine. The old
laybackrig checkout (192.168.0.177 `C:\Programming\CarbonForge`) is SUNSET for the API — it
still hosts the GPU worker stacks (ComfyUI :8188, Chatterbox :5126, Trellis :8082) and its
git tree is stale/ahead; don't build the API there.

**Verify after deploy:** ship_status ends `ok`; `docker exec carbon-forge-adhoc sh -c
"grep -c <new symbol> /app/forge_mcp/<file>"` (a stale image is silent); then exercise a
changed tool live (`forge__*`).

## Audio / TTS (two providers)

`forge__generate_speech(text, project, provider=...)` + `forge__list_voices()`:

- **`provider='elevenlabs'`** (default) — cloud, expressive, large voice library + cloning. Needs `ELEVENLABS_API_KEY` in the forge `.env`. ⚠️ Free-tier keys **cannot use Voice-Library voices** via the API (HTTP 402); omit `voice` and it defaults to the account's **own first voice** (`_default_eleven_voice` in `generation.py`), or pass a `voice_id` from `list_voices()` whose category is `premade`/`cloned`.
- **`provider='chatterbox'`** — local, free, on-prem; zero-shot voice cloning (`voice` = a workspace clip/URL) + emotion control (`exaggeration`, `cfg_weight`). Runs as its **own isolated GPU container** (`chatterbox` service, `Dockerfile.chatterbox`, `chatterbox_service/server.py`, FastAPI on :5126) — kept separate from ComfyUI so its pinned `transformers==4.46.3` can't conflict with the image/video stack (transformers 5.x). Forge routes primary→overflow across boxes (`select_chatterbox` in `generation.py`, reusing the ComfyUI presence/yield endpoints) and yields while a box is being gamed on. Deploy it like `forge`: `docker compose -f docker-compose.forge.yml build chatterbox` + `up -d chatterbox` (first request lazy-loads + caches the model into the `chatterbox-models` volume). Second box (maingamingrig) runs its own `chatterbox` container; set `FORGE_CHATTERBOX_OVERFLOW_URL` in the forge `.env`.

## Replicate (the whole catalog, three tools)

`replicate_search(query)` finds any of Replicate's thousands of hosted models;
`replicate_model('owner/name')` returns its input schema (call it before running);
`replicate_run(model, input, project, files?)` runs it — every file output is saved into the
project workspace like any other forge tool, text outputs (LLMs) come back as `output_text`,
and runs that outlast `wait_seconds` (max 300) return a `job_id` to poll with `job_status`.
`files` maps input-field names to workspace paths / URLs (uploaded through Replicate's files
API — for img2img, upscalers, transcription, …). `model` accepts `owner/name` (latest) or
`owner/name:version` (pinned). Needs `REPLICATE_API_TOKEN` in the service `.env` (org account
`carbonnode`); billed per run — image models are typically sub-cent, video models $0.10+.
Source: `forge_mcp/replicate_api.py` (HTTP client, no SDK) + `forge_mcp/tools/replicate.py`.

## Supporting infrastructure (touch points outside this repo)

| Piece | Where | Notes |
|---|---|---|
| SMB share `Workspace` | carbonserver, `C:\Workspace`, account `forge-svc` | CIFS volume in docker-compose.forge.yml |
| Public URL | `forge.carbonrouting.dev` ingress on super-server's `QuestbookCloudflared` tunnel (`C:\Programming\QuestbookTunnel\config.yml`) | → `http://localhost:5125` (repointed 2026-07-01; a dead ingress for it also lingers on carbonserver's mcp-gateway tunnel — harmless) |
| Gateway connector | row `forge` (mcp-proxy) in gateway DB, category `media` | re-register: `scripts/register-forge-connector.mjs` in Carbon-Cortex (run via tsx inside `carbon-cortex-gateway-1`, needs FORGE_TOKEN env) |
| Console UI | Carbon-Cortex `web/`: `HOME_APP_GROUPS.forge`, `connector-icons.ts`, `/icons/carbon-forge.png` | |

## Desktop app deploy

Unchanged by the MCP work: `npm run build-backend` (PyInstaller — picks up `processing.py` automatically as a normal import), then `npm run build`.

## Tests

`python -m pytest tests/ -q` (no network/model downloads). Live smokes: `tests/manual_client.py` (local service), `tests/manual_gateway_e2e.py` (full chain through the gateway, needs GW_AUTH env).
