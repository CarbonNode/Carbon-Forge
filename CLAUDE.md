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
                  #   resolve_grid() is THE auto-grid decision (pixel_refine, sprite
                  #   lock_style, prepare_native_frame all call it): detect → normalize a
                  #   FRACTIONAL cell (Gemini/Imagen draw a 100-px sprite into 1024 px =
                  #   10.24 px cells; integer detection locked onto 41) → square the cell
                  #   (8x24 → 8) → reconstruction gate → harmonic rescue (a sparse sheet's
                  #   sprite pitch out-scores its 4 px grid) → else 1:1
  sprite_anim.py  # Sprite ANIMATION engine (pure NumPy, builds on pixel_art): video frames →
                  #   ONE locked grid + ONE locked palette (from the source sprite) → shared
                  #   bounding box → dedupe → sprite sheet + Aseprite/Phaser atlas JSON + GIF.
                  #   Also fixed-cell SHEETS in (slice_sheet / build_from_sheet: one frameTag
                  #   per row + per-row GIFs — Retro Diffusion / PixelLab / Aseprite exports).
                  #   → `animate_sprite` / `video_to_sprite_sheet` / `pack_sprite_sheet` /
                  #     `import_sprite_sheet` tools
  bake_spec.py    # 3D→pixel bake CONVENTIONS, stdlib-only (both sides of the Blender process boundary):
                  #   direction tags (s se e ne n nw w sw; facing 0 looks at the camera, model turns CCW),
                  #   frame sampling (loop drops the last frame), spec validation, action-name matching
  blender_bake.py # runs INSIDE Blender (bpy): GLB → union bounds over every sampled frame of every action
                  #   (evaluated vertices, not rest-pose bound_box) → ortho cam + sun + flat world → render
                  #   action × direction × frame, transparent, Standard view → manifest.json + PNGs
  sprite_bake.py  # service half: run_blender (subprocess, never in-process) + compose (medoid downsample
                  #   → ONE k-means palette over ALL frames → outline → ONE shared crop box → pack_sheet
                  #   with `<action>_<facing>` frameTags + per-row GIFs) → tools/sprite3d.py
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
  meshy_api.py    # Meshy.ai client (image→3D / text→3D / rig / animate / poll / GLB pick) + ACTIONS
                  #   name→action_id map (idle 0, walk 30, run 14, attack 4, death 8, hurt 178, jump 466…)
  tools/          # MCP tool definitions: proc, gen, vid, sprite, sprite3d, audio, extract, util, meta (incl. local Wan T2V/I2V, ESRGAN upscale, IPAdapter reference gen, saved characters, audio TTS, batch/montage, generate_clip pipeline)
                  #   sprite.py = "pixel animation is solved": animate_sprite (sprite → sheet bundle,
                  #   one async job; engine 'wan' = local I2V → frames → refine on the sprite's own
                  #   grid/palette, engine 'retro-diffusion' = Astropulse's rd-animation on
                  #   Replicate → 4-facing preset sheet, rows tagged down/right/up/left),
                  #   video_to_sprite_sheet (same refine for ANY clip: Veo/Kling/Pixel
                  #   Engine/screen capture), pack_sprite_sheet (frames you already have),
                  #   import_sprite_sheet (a fixed-cell sheet you already have). Bundles
                  #   = <name>.png sheet + <name>.json atlas + <name>.gif + preview.html + frames,
                  #   one save_bundle id. Refinement is never per-frame-independent — see the
                  #   module docstring for why (grid/palette/bbox "boil").
                  #   sprite3d.py = the 3D-first route for ANIMATED units: bake_sprite_sheet (any animated
                  #   .glb → N-facing sheet), character_to_sprites (concept image → Meshy image→3D → rig →
                  #   clips → bake, one job), meshy_actions (clip names + balance)
                  #   audio.py = generate_speech / list_voices. TWO TTS providers:
                  #   ElevenLabs (cloud) + Chatterbox (local, isolated GPU container, see below)
                  #   util.py = quick conversions: audio_convert / audio_trim / image_convert /
                  #   draco_compress (.glb) / media_info — chat uploads are workspace paths
                  #   ('<Project>/.conduit/uploads/<name>'), so they feed straight in
tests/            # pytest (26 tests) + manual_* live-smoke clients
Dockerfile.mcp, docker-compose.forge.yml, .env.forge.example
  ⚠️ Dockerfile.mcp COPIES backend/ FILES BY NAME (no rembg/torch bloat from the desktop tree) —
  a new backend/*.py module MUST be added to that COPY line or the container crash-loops on import.
```

## Hosted service — how it works

- **I/O contract:** every tool takes inputs as https URLs or workspace paths `<Project>/<relative path>`; outputs are written BOTH to `\\carbonserver\Workspace\<project>\assets\forge\` (CIFS volume → Conduit project folders) AND a results cache served at `https://forge.carbonrouting.dev/files/<id>/<name>`. CIFS down ⇒ tool still succeeds with `workspace_write_error` + the cache URL.
- **Auth:** bearer token (`FORGE_TOKEN`) on `/mcp`; `/health` + `/files` open (unguessable ids).
- **Veo videos** are async: `generate_video` → `job_id`, poll `job_status`. Jobs persist across restarts (in-flight operations resume).
- **Generation needs `GEMINI_API_KEY`** in the service `.env`; without it those tools return a readable error and everything else works.

## Deploy (hosted service)

> ### ⛔ Production is **laybackrig**, NOT super_server — verified live 2026-09-14
>
> An earlier note here said "2026-07-28: the API container moved to super_server". **That move never
> took over.** What actually serves `forge__*` today:
>
> | | Box | Path | Container | Reality |
> |---|---|---|---|---|
> | **PRODUCTION** | `laybackrig` | `C:\Programming\CarbonForge` | `carbon-forge-forge-1` (`docker-compose.forge.yml`) | what the tunnel + gateway point at |
> | stale | `super_server` | `C:\Programming\CarbonForge-src` | `carbon-forge-adhoc` | dead since 2026-09-09, nothing routes to it |
>
> The proof chain, all pointing at laybackrig: the gateway connector row `forge` has
> `url: https://forge.carbonrouting.dev/mcp` plus `wake: {node: "laybackrig", containers:
> ["carbon-forge-forge-1", ...], health: "http://192.168.0.177:5125/health"}`; the
> `cloudflared-gateway` ingress maps `forge.carbonrouting.dev → http://192.168.0.177:5125`
> (= laybackrig); and `docker start carbon-forge-forge-1` on laybackrig takes the public
> `/health` from 502 to 200 immediately.
>
> ⚠️ **`deployer__explain` / `deployer__ship {project:'carbon-forge'}` resolve to super_server** —
> they find the compose *labels* there and cannot see that nothing routes to it. Shipping with them
> deploys to a box no traffic reaches, and the new code silently never goes live. Use the laybackrig
> scheduled-task recipe below instead.
>
> 🔌 **The containers are wake-on-demand, so `Exited (0)` is NORMAL, not an outage.** The gateway
> starts them from that `wake` block on the first `forge__*` call. Do not diagnose a stopped
> `carbon-forge-forge-1` as broken — check `https://forge.carbonrouting.dev/health` (or just call a
> forge tool) first. super_server's copy, by contrast, is genuinely broken: it dies with
> `failed to mount local volume //192.168.0.35/Workspace` (`Exited (255)`).
>
> laybackrig's `docker-compose.forge.yml` also carries an **uncommitted local production fix** —
> the CIFS volume points at `//192.168.65.254:14450` (the Docker Desktop host gateway) instead of
> `//192.168.0.35:445`. Never `git checkout` that file, and never commit it either: it is
> box-specific. `git pull --ff-only` preserves it as long as upstream leaves the file alone.

Lives at `C:\Programming\CarbonForge` on laybackrig. `.env` (gitignored) holds FORGE_TOKEN, GEMINI_API_KEY, CIFS creds.

```
ssh 192.168.0.177 "cd /d C:\Programming\CarbonForge && git pull && schtasks /create /tn ForgeBuild /tr C:\Programming\CarbonForge\build-forge.bat /sc ONCE /st 23:59 /f && schtasks /run /tn ForgeBuild"
# poll build-forge.log for BUILD_OK/DEPLOY_DONE, then: schtasks /delete /tn ForgeBuild /f
```

⚠️ `docker compose build` over plain SSH **hangs** on the Windows credential helper — the scheduled task (interactive session) is mandatory. `build-forge.bat` is in the repo root on laybackrig (gitignored content? no — created at deploy; recreate from CLAUDE.md if missing):

```bat
@echo off
cd /d C:\Programming\CarbonForge
del build-forge.log 2>nul
docker compose -f docker-compose.forge.yml build forge >> build-forge.log 2>&1
if errorlevel 1 goto failed
echo BUILD_OK >> build-forge.log
docker compose -f docker-compose.forge.yml up -d forge >> build-forge.log 2>&1
echo DEPLOY_DONE >> build-forge.log
exit /b 0
:failed
echo BUILD_FAIL >> build-forge.log
echo DEPLOY_SKIPPED >> build-forge.log
exit /b 1
```

⚠️ The `goto failed` guard matters: the **old** version of this script used
`if %errorlevel%==0 (...) else (...)` and then ran `up -d` **unconditionally**, so a failed build
still recreated the container — off the previous image. It logged `BUILD_FAIL` and `DEPLOY_DONE`
together and looked like a successful deploy. Always read the log for `BUILD_OK`, never just
`DEPLOY_DONE`.

⚠️ **Slow, noisy first boot is NORMAL after a rebuild (2026-09-14).** `rembg` pulls `pymatting`,
whose numba kernels compile with `cache=True` on first import. The container can crash-loop for
**~2.5 minutes and ~8 restarts** — `EOFError: Ran out of input` (a half-written numba cache index)
then `SystemError: unknown opcode 218` — before it settles and serves normally. Give it ~3 minutes
before concluding a deploy failed. If it never settles, `docker compose ... up -d --force-recreate
forge` gives it a clean writable layer: `restart: unless-stopped` restarts the SAME container, so a
cache truncated by the first crash is re-read forever and the loop is self-sustaining. Rolling back
is NOT an option — the build tags `carbon-forge-mcp:latest` in place and the previous image is
pruned, so always fix forward.

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
- **The pixel-art FEEL (2026-09-08, Rober: "didn't feel pixel art style"):** a 100-px sprite cut into
  16 evenly spaced frames reads as a downscaled cartoon — video models tween, hand animation holds.
  What reads as pixel art: `cell_size` so the sprite lands at **32-64 logical px**, **6-8 frames**
  chosen as distinct KEY POSES (`frame_select:"poses"` — dense 6× sample → `select_poses`
  farthest-point on changed pixels with a minimum temporal gap), **`hold_timing:true`** (each pose
  holds for the span it stands for; per-frame durations in atlas + GIF), **12-16 colors**,
  `outline:"sharp"`. Two things silently broke the look before: the chroma key's tint smeared into
  edge pixels by video compression, and k-means then LEARNED magenta as a palette entry so every
  fringe snapped back to it — `despill_key` (on by default with a key) + `is_key_like` palette
  filtering fix both. Same clip, re-cut: `slam-poses8b` vs the original 16-frame cut.
- **Tests:** `tests/test_sprite_anim.py` simulates an I2V clip (shift + blur + noise + magenta key)
  and asserts the lock (cell size, palette, shared box, ≤ source colors), plus sheet slicing /
  row tags / shared crop for the sheet path.

### Making pixel UNITS that don't look cutesy (2026-09-14, verified against Orc Incremental)

Rober's reference is Orc Incremental: tiny (~20-30 px) units, **flat hand-drawn** fills, hard
outlines, muted earthy palette, **side view only** (mirrored left/right — no 8-facing rotation).
The recipe that landed, and the traps that cost iterations getting there:

**Route:** `generate_image` (coarse + anti-chibi prompt, MAGENTA background) → `pixel_refine`
**with `preset:"sprite"`** and a forced `cell_size`. NOT the 3D bake — see below.

> ### ✅ You do not have to remember the rest of this list — `preset:"sprite"` IS the list
> `pixel_refine { preset: "sprite" }` sets `sampling:"hard"`, `bg_tolerance:70`, `despeckle:4`,
> `outline:"none"`, `max_colors:12` — every setting below that was learned by staring at bad
> output. Anything you pass explicitly still wins, so the preset is a floor, not a cage.
>
> And **every call now returns `analysis.warnings`**, which names the failure modes actually
> present in YOUR output: soft/blurry edges, leftover background key, floating pixel islands, a
> subject clipped at the canvas edge, failed grid detection. **Read that array instead of
> eyeballing the sprite** — it is the difference between one round trip and five. The only thing
> it cannot choose for you is `cell_size` (that is a judgement about the art's real grid) and
> whether the result looks good.

- **The 3D bake is the WRONG tool for small units.** `bake_sprite_sheet` renders *shaded 3D*
  and downscales; the reference is *flat drawn*. At 14x18 px a KayKit knight bakes to an
  unreadable grey blob (the 35° elevation crushes the figure). Keep `bake_sprite_sheet` for
  cases that genuinely need 8 consistent facings; a side-view game needs two.
- **Anti-chibi prompting is mandatory and it works.** "GRIM, NOT cute, NOT chibi, NOT a mascot,
  SMALL head on a heavy hunched body, heavy brow, muted desaturated palette." Without those,
  every model (Gemini and Retro Diffusion alike) drifts to big-head mascot proportions.
- **RD imposes its own proportions — you cannot prompt them away.** `animate_sprite
  engine:"retro-diffusion"` *re-draws* the character in its own 48 px chibi style. If "too
  cutesy" is the complaint, RD is the cause, and the fix is `engine:"wan"` (animates YOUR
  sprite, keeps your proportions) or the `action=` official-API path — not better prompting.
- **`lock_palette:true` can destroy an RD result.** RD already returns true pixel art (~13-23
  colours); snapping that onto a darker source palette turned a clean orc into mud. Lock only
  when the source palette is genuinely the one you want.
- **Re-cutting is FREE.** The raw RD sheet is saved in the job results (`rd_sheet`), so
  `import_sprite_sheet` rebuilds the bundle with different tags/fps/palette at zero cost.
  Never pay a second prediction to fix framing.
- **`generate_image` ignores exact hex backgrounds.** Ask for `#00FF00`, get `#339B42`. ALWAYS
  sample the actual corner pixel before keying, or the key silently misses.
- **Never key on a colour near the subject's own palette.** Green background + green orc ate
  the orc. Magenta is right for green creatures.
- **`outline:"sharp"` on art that ALREADY has an outline doubles it** — and it traces the keyed
  edge, tripling the chroma fringe (77 stray px vs 23). Use `outline:"none"` on generated art
  that drew its own outline; reserve the outline pass for renders and video frames.
- **Detailed art does not survive being crushed small.** A 200 px-detailed orc forced to 45 px
  is mud. Generate AT unit coarseness ("VERY COARSE pixel grid, ~28 px tall, large pixel
  blocks") rather than shrinking a detailed piece.
- **Grid auto-detection fails on detailed/painterly art** (`analysis.grid.detected: false`, or a
  1:1 fallback at the input size), so you must force `cell_size` — but force the grid the art was
  actually DRAWN on, not the sprite size you wish you had. Guessing too coarse silently destroys
  the face: the same 1024 px orc at cell 30 is a 32x32 mud blob, while cell 12-14 gives a 66-82 px
  sprite with a readable brow, eye and tusks. **Sweep 12 / 14 / 16 / 20 and pick by eye** — a
  detailed generation almost always sits in the 12-16 range.
- **Use `sampling:"hard"` for sprites — the `"medoid"` default is what makes linework look
  blurry.** Medoid leaves anti-aliased edge pixels: measured 368 semi-transparent pixels forming a
  grey halo around every outline on one 79x84 orc, vs **0** with `hard` (medoid + binary alpha).
  This is the single biggest quality lever in the whole pipeline.
- **`bg_tolerance` 45 is too tight for a Gemini magenta field.** Gemini paints faint off-magenta
  haze bands across the canvas — one measured ~28,900 px at `(209,37,214)`, distance **46** from
  pure magenta, so a tolerance of 45 missed the lot by ONE unit and it survived as background junk
  floating above the sprite. Use **70**.
- **Keying leaves FLOATING PIXELS — use `despeckle:4`.** Even at the right tolerance a few
  off-key specks survive as islands beside the sprite: one 94x98 orc came out as **5 connected
  components** — the body at 6057 px plus four islands totalling 6 px, all from the haze band.
  Invisible at 1x, obvious the moment an engine scales the sprite up. `despeckle` drops opaque
  components below N pixels; it is OFF by default because it deletes content, and the threshold
  must stay well under any real detached part (a held weapon is its own component and must
  survive — 4 is safe, the weapons measured here are 16+ px).
- **Ask for margin, or the model clips the subject.** A generation came back with 183 opaque
  pixels on the bottom row — the character's feet cut off by the canvas. Put "full body inside the
  frame with margin on all sides, feet fully visible" in every sprite prompt.
- **Refining cannot invent a grid the art never had.** Sweeping cell sizes 10-21 x every offset
  (186 combinations) on an `edit_image` result gave a nearly FLAT reconstruction-error surface
  (12.56 best vs 13.54 for a bad guess) — there was no true grid to find. Art prompted coarse from
  the start snaps cleanly; art that has been through `edit_image` re-renders softer and never
  fully crisps. Prefer regenerating over editing when the sprite must be crisp.
- **Below ~48 px a face cannot exist**, in any pipeline — that is 3-4 pixels of head. Orc
  Incremental itself does not have faces on its battlefield units; they are silhouettes, and the
  faces live in the big portrait row. Decide which register you are making: ~25 px silhouette
  units, or 64-80 px sprites you can actually read. Do not expect one asset to be both.

### Engine `retro-diffusion` ("astro") — Astropulse's rd-animation on Replicate (2026-09-08)

`animate_sprite { engine: "retro-diffusion", style, subject }` sends the sprite as the
`input_image` reference of `retro-diffusion/rd-animation` (`RD_MODEL` in `tools/sprite.py`,
billed to the org Replicate token, ~$0.07-0.25 and ~30 s per run) and turns the returned sheet
into the same bundle. What was verified on live output (don't guess — re-check if RD changes it):

| style | cell | sheet | rows (top→bottom) | columns |
|---|---|---|---|---|
| `four_angle_walking` | 48 | 192×192 | down, right, up, left | 4 walk frames |
| `walking_and_idle` | 48 | 192×192 | down, right, up, left | 3 walk frames + 1 idle pose |
| `small_sprites` | 32 | 160×128 | down, right, up, left | 5 action poses (idle/walk/attack/hurt/down) |
| `vfx` | 24-96 (`size`) | one row | — | frames |

- The output is transparent RGBA, already true pixel art (~13-23 colors) — so it is **sliced 1:1,
  not refined** (`sa.build_from_sheet`, cell size 1). `lock_palette:true` snaps it to the SOURCE
  sprite's palette (`max_colors` / `palette` / `palette_colors`) for consistency with other assets.
- It is a **rendition**, not your pixels: the model re-draws the character in its own 48 px style
  from the reference + `subject` prompt (defaults to `motion`). Same seed ≠ same result across
  `return_spritesheet` true/false. The reference is flattened on WHITE (`prepare_reference`, 256²) —
  magenta leftovers in the sprite show up as pink accents in the result, so feed clean cut-outs.
- Rows become Aseprite `frameTags` (`down` 0-3, `right` 4-7 …) and each facing also gets
  `<name>-<tag>.gif` (`tag_gifs` in the result); the sheet keeps the row structure (`columns=0`).
- Job results: `reference` (what RD saw), `rd_sheet` (raw, with `prediction_id`), the bundle.
  Predictions are created ONCE and never retried (a blind retry double-charges); a gateway
  timeout on the tool call does NOT mean the prediction failed — check `job_status`.
- **`action=walking|idle|jump|crouch|attack|destroy|custom_action|subtle_motion`** uses the OFFICIAL
  `api.retrodiffusion.ai` v2 "advanced animations" (`forge_mcp/retrodiffusion_api.py`) — it animates
  YOUR EXACT frame at its native 32-256 px size (`prepare_native_frame`), `frames` snapped to
  4/6/8/10/12/16, `motion` is the motion text, ~$0.14 ($0.25 custom/subtle); needs
  `RETRO_DIFFUSION_API_KEY` in the service `.env` (none in the vault as of 2026-09-08 — the code
  path is written and unit-tested but NOT verified live; first run with a key: check `job_status`
  results `rd_sheet.balance_cost`). Without the key the tool returns a readable error.
- `import_sprite_sheet { sheet, frame_w, frame_h, row_tags }` is the same slicer for any sheet
  you already have (an RD web download, PixelLab, an Aseprite export).

## 3D → pixel sprites (`character_to_sprites` / `bake_sprite_sheet`) — the animated-unit route (2026-09-14)

Diffusion / video / puppet-rig generators drift between frames and cannot keep 8 facings
consistent; a RIG cannot drift. So animated units go 3D-first (how Warcraft II / Diablo made their
sprites) and the pixel look is applied afterwards on one locked grid + palette:

```
concept PNG ──Meshy image→3D──▶ textured .glb ──Meshy rig──▶ armature (+ FREE walk/run clips)
   ──Meshy animate (action_id)──▶ one .glb per clip ──backend/blender_bake.py (bpy, subprocess)──▶
   action × facing × frame PNGs ──sprite_bake.compose──▶ sheet + atlas (`walk_s`, `walk_se`, … `death_sw`)
```

- **`character_to_sprites { image | prompt, project, actions, directions, cell, … }`** runs the whole
  chain as one job (`job_status`). Every intermediate `.glb` (model, rigged, each clip) is saved to
  the results, so a re-bake with other `cell`/`palette`/`elevation` is **`bake_sprite_sheet` on the
  clip .glb — no credits**. Meshy credits: image→3D ~20, rig 5, animate 3 per clip that is not free
  (`walk`/`run` come with the rig). `meshy_actions` = name→id map + live balance. Humanoids only
  (Meshy's rigger rejects animals/props/vehicles). Needs `MESHY_API_KEY` in the service `.env` (vault
  label `meshy_api_key`, the same key the Cortex blender connector uses).
- **Conventions (backend/bake_spec.py — never change one side only):** facing 0 = toward the camera
  (`s`), tags run `s se e ne n nw w sw` (model rotates counter-clockwise seen from above; glTF +Z
  forward becomes Blender −Y, the camera sits on −Y). `loop=true` drops the last sampled frame
  (== first) for cycles; one-shots (attack/death/hurt/jump) keep the final pose; `loop=null` guesses
  from the action name (`LOOPING` in tools/sprite3d.py). `cell` is the box the WHOLE animation's union
  bounds fit into — the standing figure is smaller than `cell` (48-64 reads as Warcraft II).
- **What the bake locks (why it does not boil):** ONE camera framing from the union bounds of every
  sampled frame of every action (measured on evaluated vertices — a skinned mesh's `bound_box` is the
  rest pose and was off by a metre); ONE k-means palette over ALL frames; ONE crop box across all rows
  so the ground line is identical in every frame/facing. Render at `cell × supersample` (default 4),
  Cycles on CPU (16 samples — the medoid downsample eats the noise), `Standard` view transform
  (AgX/Filmic would desaturate the texture), transparent film, flat white world 0.55 + one sun that
  does NOT turn with the character, so every facing is lit from screen-left like a real sprite set.
- **Blender lives in its own venv:** the `bpy` wheel is CPython 3.11-only, so `Dockerfile.mcp` makes
  `/opt/bpy` on Debian's python3.11 (`FORGE_BPY_PYTHON`) — the 3.12 service graph is untouched, same
  isolation as gltf-transform/extractor. `sprite_bake.find_blender` also honours `FORGE_BLENDER_BIN`.
  Gotchas found live: the glTF importer creates an unlinked `Icosphere` bone-shape in a
  `glTF_not_exported` collection (skip anything `hide_render`/not `visible_get()`);
  `read_factory_settings(use_empty=True)` can leave orphan datablocks (the script purges them);
  Blender 4.4+ slotted actions need `animation_data.action_slot` set or the action does nothing.
- **Tests:** `tests/test_sprite_bake.py` (conventions, subprocess plumbing with a fake Blender,
  compose: palette lock, shared crop, ground line, tags/atlas) + `tests/test_meshy_api.py` — no
  Blender needed. Live check after a deploy: `forge_status` → `blender_bake.available: true`,
  `meshy_key_configured: true`; then `bake_sprite_sheet` on any animated .glb (Khronos CesiumMan:
  8 dirs × 6 frames in ~4 s on CPU, verified locally 2026-09-14).
