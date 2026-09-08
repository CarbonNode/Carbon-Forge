# Carbon Forge

AI-powered asset generation & refinement. Repo: `CarbonNode/Carbon-Forge`. Two deliverables share one engine:

1. **Desktop app** — Electron (`main.js`, `renderer/`) + bundled Python Flask backend (`backend/server.py`, PyInstaller via `npm run build-backend`). Runs locally on port 5123.
2. **Hosted MCP service** — `forge_mcp/` package, Docker container on **laybackrig** (192.168.0.177:5125). Proxied by the Carbon Cortex gateway as connector **`forge`** (tools surface as `forge__*` in every gateway session).

## The one shared-engine rule

**`backend/processing.py` is the single image engine** (rembg pipelines, LaMa watermark removal, color/edge ops, sprite splitting). Both `backend/server.py` (desktop) and `forge_mcp/engine.py` (hosted) import it. **Never fork the logic** — change `processing.py` and both stay in sync. `server.py` is only Flask routing; `forge_mcp` is only MCP plumbing around the same functions.

## Layout

```
backend/
  processing.py   # THE engine — pure functions, PipelineOptions, run_pipeline, run_split_pipeline
  pixel_art.py    # PixelRefiner port (pure NumPy, no models): pixel-grid detection,
                  #   cell resampling (Oklab medoid), k-means quantization, retro
                  #   palettes, dithering, outline/trim/scale → `pixel_refine` MCP tool
  sprite_anim.py  # Sprite ANIMATION engine (pure NumPy, builds on pixel_art): video frames →
                  #   ONE locked grid + ONE locked palette (from the source sprite) → shared
                  #   bounding box → dedupe → sprite sheet + Aseprite/Phaser atlas JSON + GIF.
                  #   → `animate_sprite` / `video_to_sprite_sheet` / `pack_sprite_sheet` tools
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
  tools/          # MCP tool definitions: proc, gen, vid, sprite, audio, extract, util, meta (incl. local Wan T2V/I2V, ESRGAN upscale, IPAdapter reference gen, saved characters, audio TTS, batch/montage, generate_clip pipeline)
                  #   sprite.py = "pixel animation is solved": animate_sprite (sprite → Wan I2V →
                  #   frames → refine on the sprite's own grid/palette → sheet bundle, one async
                  #   job), video_to_sprite_sheet (same refine for ANY clip: Veo/Kling/Pixel
                  #   Engine/screen capture), pack_sprite_sheet (frames you already have). Bundles
                  #   = <name>.png sheet + <name>.json atlas + <name>.gif + preview.html + frames,
                  #   one save_bundle id. Refinement is never per-frame-independent — see the
                  #   module docstring for why (grid/palette/bbox "boil").
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

## Deploy (hosted service)

Lives at `C:\Programming\CarbonForge` on laybackrig. `.env` (gitignored) holds FORGE_TOKEN, GEMINI_API_KEY, CIFS creds.

```
ssh 192.168.0.177 "cd /d C:\Programming\CarbonForge && git pull && schtasks /create /tn ForgeBuild /tr C:\Programming\CarbonForge\build-forge.bat /sc ONCE /st 23:59 /f && schtasks /run /tn ForgeBuild"
# poll build-forge.log for BUILD_OK/DEPLOY_DONE, then: schtasks /delete /tn ForgeBuild /f
```

⚠️ `docker compose build` over plain SSH **hangs** on the Windows credential helper — the scheduled task (interactive session) is mandatory. `build-forge.bat` is in the repo root on laybackrig (gitignored content? no — created at deploy; recreate from CLAUDE.md if missing):

```bat
@echo off
cd /d C:\Programming\CarbonForge
docker compose -f docker-compose.forge.yml build forge >> build-forge.log 2>&1
if %errorlevel%==0 (echo BUILD_OK >> build-forge.log) else (echo BUILD_FAIL >> build-forge.log)
docker compose -f docker-compose.forge.yml up -d forge >> build-forge.log 2>&1
echo DEPLOY_DONE >> build-forge.log
```

**Verify after deploy:** `https://forge.carbonrouting.dev/health` → 200; `python tests/manual_status.py http://192.168.0.177:5125/mcp` (FORGE_TOKEN env) → `workspace_writable: true`, `ffmpeg_available: true`.

## Audio / TTS (two providers)

`forge__generate_speech(text, project, provider=...)` + `forge__list_voices()`:

- **`provider='elevenlabs'`** (default) — cloud, expressive, large voice library + cloning. Needs `ELEVENLABS_API_KEY` in the forge `.env`. ⚠️ Free-tier keys **cannot use Voice-Library voices** via the API (HTTP 402); omit `voice` and it defaults to the account's **own first voice** (`_default_eleven_voice` in `generation.py`), or pass a `voice_id` from `list_voices()` whose category is `premade`/`cloned`.
- **`provider='chatterbox'`** — local, free, on-prem; zero-shot voice cloning (`voice` = a workspace clip/URL) + emotion control (`exaggeration`, `cfg_weight`). Runs as its **own isolated GPU container** (`chatterbox` service, `Dockerfile.chatterbox`, `chatterbox_service/server.py`, FastAPI on :5126) — kept separate from ComfyUI so its pinned `transformers==4.46.3` can't conflict with the image/video stack (transformers 5.x). Forge routes primary→overflow across boxes (`select_chatterbox` in `generation.py`, reusing the ComfyUI presence/yield endpoints) and yields while a box is being gamed on. Deploy it like `forge`: `docker compose -f docker-compose.forge.yml build chatterbox` + `up -d chatterbox` (first request lazy-loads + caches the model into the `chatterbox-models` volume). Second box (maingamingrig) runs its own `chatterbox` container; set `FORGE_CHATTERBOX_OVERFLOW_URL` in the forge `.env`.

## Supporting infrastructure (touch points outside this repo)

| Piece | Where | Notes |
|---|---|---|
| SMB share `Workspace` | carbonserver, `C:\Workspace`, account `forge-svc` | CIFS volume in docker-compose.forge.yml |
| Public URL | `forge.carbonrouting.dev` ingress on `cloudflared-gateway` tunnel (carbonserver, `C:\Programming\mcp-gateway\cloudflared\config.yml`) | → `http://192.168.0.177:5125` |
| Gateway connector | row `forge` (mcp-proxy) in gateway DB, category `media` | re-register: `scripts/register-forge-connector.mjs` in Carbon-Cortex (run via tsx inside `carbon-cortex-gateway-1`, needs FORGE_TOKEN env) |
| Console UI | Carbon-Cortex `web/`: `HOME_APP_GROUPS.forge`, `connector-icons.ts`, `/icons/carbon-forge.png` | |

## Desktop app deploy

Unchanged by the MCP work: `npm run build-backend` (PyInstaller — picks up `processing.py` automatically as a normal import), then `npm run build`.

## Tests

`python -m pytest tests/ -q` (no network/model downloads). Live smokes: `tests/manual_client.py` (local service), `tests/manual_gateway_e2e.py` (full chain through the gateway, needs GW_AUTH env).

## Sprite animation (`animate_sprite`) — how it is meant to be used

The recipe that makes AI pixel animation usable: draw the sprite ONCE, let Wan 2.2 I2V move it,
then force every sampled frame back onto the SOURCE sprite's pixel grid and palette
(`backend/sprite_anim.py`). Per-frame `pixel_refine` is the wrong tool for a clip — each frame
gets its own grid, palette and bounding box and the result boils.

- **Input sprite:** a transparent PNG is best (`generate_image` on flat magenta → `remove_background`
  → `pixel_refine`, or any true pixel sprite). `prepare_reference` flattens it on the key color and
  integer-upscales (nearest) into the 512×512 I2V frame — no resampling blur, no stretching.
- **Motion prompt:** the tool wraps `motion` in the camera-lock / flat-background / loop scaffold
  (`SPRITE_MOTION_TEMPLATE`); pass only the action ("walk cycle", "sword slash").
- **Re-cutting:** the raw I2V mp4 is in the job results — rerun `video_to_sprite_sheet` with other
  `frames` / `palette` / `fps` without paying the GPU again. `reference_sprite` should be the
  `-start.png` from the job (or the original sprite) so the lock matches.
- **Atlas format:** Aseprite JSON-array (`frames[]` + `meta.frameTags` + our `meta.layout`); Phaser
  `this.load.aseprite(key, sheet, atlas)`, Godot/Unity Aseprite importers read it directly.
  `columns=0` = one horizontal strip (CSS `steps()` in the bundled `preview.html`).
- **Tests:** `tests/test_sprite_anim.py` simulates an I2V clip (shift + blur + noise + magenta key)
  and asserts the lock (cell size, palette, shared box, ≤ source colors).
