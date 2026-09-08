"""Sprite ANIMATION tools — one sprite in, a game-ready animation out.

The pipeline ("pixel animation is solved"): a still sprite → a video model
moves it (local Wan 2.2 I2V, free + uncensored) → frames are sampled from the
clip → every frame is brought back to TRUE pixel art on ONE locked grid and
ONE locked palette (backend.sprite_anim) → sprite sheet + Aseprite/Phaser
atlas JSON + looping GIF + individual frames, delivered as one bundle.

Four entry points, lowest to highest level:
  pack_sprite_sheet      — frames you already have (PNGs) → sheet/atlas/GIF
  import_sprite_sheet    — a fixed-cell SHEET you already have (Retro Diffusion,
                           PixelLab, Aseprite export) → tagged atlas bundle
  video_to_sprite_sheet  — ANY video of a sprite (Wan, Veo, Kling, Pixel Engine,
                           a screen recording…) → refined sheet bundle
  animate_sprite         — the whole thing as one async job: sprite → sheet, with
                           two engines: local Wan 2.2 I2V (free, any motion, one
                           facing) or Retro Diffusion "rd-animation" by Astropulse
                           on Replicate (paid, ~30s, 4-direction walk/idle presets
                           that come back as TRUE pixel art)
"""
import asyncio
import json

from backend import pixel_art as pa
from backend import sprite_anim as sa
from forge_mcp import generation as g
from forge_mcp import replicate_api as R
from forge_mcp import retrodiffusion_api as RD
from forge_mcp import storage, video

# Retro Diffusion's sprite-animation model (Astropulse) as hosted on Replicate.
# Its presets return a fixed-cell grid PNG: rows = facings (sa.RD_DIRECTIONS),
# columns = frames. Layouts verified on live output 2026-09-08.
RD_MODEL = "retro-diffusion/rd-animation"
RD_STYLES = {
    "four_angle_walking": {"size": 48, "rows": sa.RD_DIRECTIONS, "fps": 8,
                           "note": "4 facings x 4 walk frames, humanoids, 48x48"},
    "walking_and_idle": {"size": 48, "rows": sa.RD_DIRECTIONS, "fps": 8,
                         "note": "4 facings x (3 walk frames + 1 idle pose), humanoids, 48x48"},
    "small_sprites": {"size": 32, "rows": sa.RD_DIRECTIONS, "fps": 6,
                      "note": "4 facings x 5 action poses (idle/walk/attack/hurt/down), 32x32"},
    "vfx": {"size": None, "rows": None, "fps": 12,
            "note": "effects (fire, smoke, slash…), 24-96 px, frames in one row"},
}

# Motion prompt scaffolding for I2V. Wan follows the start frame closely; the
# words that matter are "static camera", "flat background", "stays centered".
SPRITE_MOTION_TEMPLATE = (
    "pixel art sprite animation, {motion}. 2D side-scroller game sprite, the character stays "
    "centered in frame, static camera, no camera movement, no zoom, completely flat solid "
    "{key} background that never changes, crisp chunky pixels, limited color palette, "
    "looping animation cycle, consistent character design"
)
SPRITE_NEGATIVE = (
    "camera pan, camera zoom, camera shake, background scenery, background change, gradient, "
    "blur, motion blur, smooth shading, 3d render, realistic, extra characters, text, watermark, "
    "morphing, color shift, new colors, the character leaving the frame"
)


def _preview_html(name, atlas, sheet_name, gif_name):
    lay = atlas["meta"]["layout"]
    fw, fh, n, cols, fps = lay["frame_w"], lay["frame_h"], lay["count"], lay["columns"], lay["fps"]
    return f"""<!doctype html><meta charset="utf-8"><title>{name} — sprite preview</title>
<style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#1a1a1f;color:#ddd;font:14px system-ui}}
.wrap{{text-align:center}} .stage{{display:inline-block;image-rendering:pixelated;background:
repeating-conic-gradient(#2a2a30 0 25%,#222 0 50%) 0 0/16px 16px;border:1px solid #444;padding:16px}}
.sprite{{width:{fw}px;height:{fh}px;background:url('{sheet_name}') 0 0 no-repeat;image-rendering:pixelated;
transform:scale(var(--z,3));transform-origin:top left;animation:play {n / fps:.3f}s steps({n}) infinite}}
.box{{width:calc({fw}px * var(--z,3));height:calc({fh}px * var(--z,3));display:inline-block}}
@keyframes play{{to{{background-position:-{fw * n}px 0}}}}
p{{opacity:.7}} a{{color:#8ab4ff}}
</style>
<div class="wrap"><div class="stage"><div class="box"><div class="sprite"></div></div></div>
<p>{n} frames · {fw}×{fh} · {fps} fps · <a href="{sheet_name}">sheet</a> · <a href="{gif_name}">gif</a> ·
<a href="{name}.json">atlas.json</a></p></div>
<script>
// CSS steps() needs a single-row strip; multi-row sheets fall back to the GIF.
if ({cols} < {n}) {{ document.querySelector('.stage').innerHTML =
  '<img src="{gif_name}" style="image-rendering:pixelated;transform:scale(3);transform-origin:top left" class="box">'; }}
</script>"""


async def _deliver(res, *, name, project, subpath, cfg, extra=None):
    """Write the animation bundle (sheet, gif, atlas, preview, frames) and shape the result."""
    files = [
        (f"{name}.png", res["sheet"]),
        (f"{name}.gif", res["gif"]),
        (f"{name}.json", res["atlas_json"].encode("utf-8")),
        ("preview.html", _preview_html(name, res["atlas"], f"{name}.png", f"{name}.gif").encode("utf-8")),
    ]
    for i, fr in enumerate(res["frames"]):
        files.append((f"{name}-{i:02d}.png", fr))
    row_gifs = res.get("row_gifs") or {}
    for tag, data in row_gifs.items():
        files.append((f"{name}-{storage.safe_filename(tag)}.gif", data))
    bundle = await storage.save_bundle(files, project=project, subpath=subpath, cfg=cfg)
    out = {
        "sheet": bundle["files"][f"{name}.png"],
        "gif": bundle["files"][f"{name}.gif"],
        "atlas": bundle["files"][f"{name}.json"],
        "preview": bundle["files"]["preview.html"],
        "frames": [bundle["files"][f"{name}-{i:02d}.png"] for i in range(len(res["frames"]))],
        "layout": res["atlas"]["meta"]["layout"],
        "report": res["report"],
        "bytes": bundle["bytes"],
    }
    if row_gifs:
        out["tags"] = res["report"].get("tags")
        out["tag_gifs"] = {tag: bundle["files"][f"{name}-{storage.safe_filename(tag)}.gif"]
                           for tag in row_gifs}
    if "workspace_dir" in bundle:
        out["workspace_dir"] = bundle["workspace_dir"]
    if "workspace_write_error" in bundle:
        out["workspace_write_error"] = bundle["workspace_write_error"]
    if extra:
        out.update(extra)
    return out


def _check_palette(palette):
    if palette and palette not in pa.RETRO_PALETTES:
        raise ValueError(f"Unknown palette '{palette}' — available: {', '.join(sorted(pa.RETRO_PALETTES))}")


def register(mcp, ctx):
    cfg, jobs = ctx.cfg, ctx.jobs

    async def _frames_from_video(data, mime, frames, start, end, loop):
        raw = await video.probe(data, in_ext={"video/webm": "webm", "video/quicktime": "mov"}.get(mime, "mp4"))
        info = video.summarize_probe(raw)
        duration = info.get("duration_s") or 0
        if not duration:
            vs = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {})
            duration = float(vs.get("duration") or 0)
        if not duration:
            raise video.VideoError("Could not read the video's duration")
        times = sa.frame_timestamps(duration, frames, start_s=video.to_seconds(start) if start else 0.0,
                                    end_s=video.to_seconds(end) if end else None, loop=loop)
        pngs = await video.extract_frames(data, times, fmt="png",
                                          in_ext={"video/webm": "webm", "video/quicktime": "mov"}.get(mime, "mp4"))
        return pngs, times, info

    @mcp.tool()
    async def video_to_sprite_sheet(
        video_input: str,
        project: str,
        frames: int = 8,
        reference_sprite: str | None = None,
        key_color: str | None = "auto",
        key_tolerance: int = sa.DEFAULT_KEY_TOLERANCE,
        cell_size: int = 0,
        max_colors: int = sa.DEFAULT_MAX_COLORS,
        palette: str | None = None,
        palette_colors: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        loop: bool = True,
        dedupe: bool = True,
        outline: str = "none",
        outline_color: str = "#000000",
        dither: str = "none",
        fps: int = 12,
        columns: int = 0,
        padding: int = 0,
        scale: int = 1,
        name: str = "sprite",
        subpath: str | None = None,
    ) -> dict:
        """Turn a VIDEO of a pixel sprite into a game-ready animation: sprite sheet PNG +
        Aseprite/Phaser atlas JSON + looping GIF + per-frame PNGs + an HTML preview. Works
        on ANY clip — Wan I2V (animate_image / animate_sprite), Veo, Kling, Pixel Engine,
        a screen capture — as long as the sprite sits on a flat background.

        WHY NOT just pixel_refine each frame: each frame would get its own grid, palette and
        bounding box and the animation "boils". This locks all three: ONE cell size + ONE
        palette derived from reference_sprite (the still the video was made from — pass it
        whenever you have it; else the first frame), every frame resampled on that grid
        (Oklab medoid), background keyed at LOW resolution (clean after compression), snapped
        to the palette, cropped to one shared bounding box, duplicates dropped.

        video_input / reference_sprite: https URL or workspace path '<Project>/<relative path>'.
        frames: how many to sample, evenly spaced over [start, end] (default the whole clip;
          loop=true stops one step short of the end so a cycle doesn't repeat its first pose).
        key_color: '#rrggbb' flat background to key out, 'auto' (default — sample the first
          frame's border) or null to keep frames opaque. key_tolerance: 0-255 (48 default;
          raise for heavily compressed clips).
        cell_size: force the logical pixel size in video px (0 = derive from the reference).
        max_colors (k-means from the reference, 16) / palette (retro name: pico8, nes, gb_*,
          c64, msx, pc98, arne16, sfc_*) / palette_colors (custom '#rrggbb' list) lock colors.
        outline 'rounded'|'sharp' adds a 1px outline; dither: none|floyd-steinberg|bayer-*.
        fps: playback rate written into the atlas + GIF. columns: 0 = one horizontal strip
          (what CSS steps() and most engines expect); N wraps the sheet. scale: integer
          nearest-neighbor export scale (atlas coordinates match the exported PNG).
        Returns {sheet, gif, atlas, preview, frames[], layout, report}."""
        _check_palette(palette)
        if dither not in pa.DITHER_MODES:
            return {"error": f"dither must be one of {', '.join(pa.DITHER_MODES)}"}
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        ref_bytes = None
        if reference_sprite:
            ref_bytes = (await storage.resolve_input(reference_sprite, cfg=cfg, kind="image")).data
        frames = max(1, min(64, int(frames)))
        pngs, times, info = await _frames_from_video(src.data, src.mime, frames, start, end, loop)
        res = await asyncio.to_thread(
            sa.build_animation, pngs, reference=ref_bytes, cell_size=cell_size, max_colors=max_colors,
            palette=palette, palette_colors=palette_colors, key_color=key_color,
            key_tolerance=key_tolerance, dither=dither, outline=outline, outline_color=outline_color,
            dedupe=dedupe, columns=columns, padding=padding, scale=scale, fps=fps,
            name=storage.safe_filename(name), source_times=times)
        return await _deliver(res, name=storage.safe_filename(name), project=project, subpath=subpath,
                              cfg=cfg, extra={"video": info, "sampled_times_s": times})

    @mcp.tool()
    async def pack_sprite_sheet(
        images: list[str],
        project: str,
        refine: bool = False,
        reference_sprite: str | None = None,
        key_color: str | None = None,
        max_colors: int = sa.DEFAULT_MAX_COLORS,
        palette: str | None = None,
        cell_size: int = 0,
        dedupe: bool = False,
        fps: int = 12,
        columns: int = 0,
        padding: int = 0,
        scale: int = 1,
        name: str = "sprite",
        subpath: str | None = None,
    ) -> dict:
        """Pack frame images you ALREADY have (2-64 PNGs, in order — e.g. split_sprites output,
        generate_image_grid cells, hand-drawn frames) into a sprite sheet + atlas JSON + GIF +
        HTML preview. Frames are aligned on one shared bounding box (never trimmed
        individually) and padded to one size. refine=true runs them through the same
        grid+palette lock as video_to_sprite_sheet first (for AI 'fake pixel' frames of
        differing sizes; frames must then share one pixel scale). Other params as there."""
        _check_palette(palette)
        if not images or len(images) < 2 or len(images) > 64:
            return {"error": "Pass 2-64 frame images, in playback order"}
        pngs = [(await storage.resolve_input(ref, cfg=cfg, kind="image")).data for ref in images]
        safe = storage.safe_filename(name)
        if refine:
            ref_bytes = (await storage.resolve_input(reference_sprite, cfg=cfg, kind="image")).data \
                if reference_sprite else None
            res = await asyncio.to_thread(
                sa.build_animation, pngs, reference=ref_bytes, cell_size=cell_size, max_colors=max_colors,
                palette=palette, key_color=key_color, dedupe=dedupe, columns=columns, padding=padding,
                scale=scale, fps=fps, name=safe)
        else:
            res = await asyncio.to_thread(sa.pack_existing, pngs, dedupe=dedupe, columns=columns,
                                          padding=padding, scale=scale, fps=fps, name=safe)
        return await _deliver(res, name=safe, project=project, subpath=subpath, cfg=cfg)

    @mcp.tool()
    async def import_sprite_sheet(
        sheet: str,
        frame_w: int,
        frame_h: int,
        project: str,
        columns: int = 0,
        rows: int = 0,
        row_tags: list[str] | None = None,
        reference_sprite: str | None = None,
        key_color: str | None = None,
        max_colors: int = 0,
        palette: str | None = None,
        palette_colors: list[str] | None = None,
        outline: str = "none",
        outline_color: str = "#000000",
        out_columns: int = 0,
        padding: int = 0,
        scale: int = 1,
        fps: int = 8,
        name: str = "sprite",
        subpath: str | None = None,
    ) -> dict:
        """Turn a fixed-cell sprite SHEET you already have — a Retro Diffusion / PixelLab
        download, an Aseprite or Unity export, a sheet from another animate_sprite job —
        into the standard bundle: repacked sheet PNG + Aseprite/Phaser atlas JSON with ONE
        frameTag PER ROW + looping GIF (all frames) + one GIF per row + frame PNGs + preview.

        sheet: https URL or '<Project>/<path>'. frame_w/frame_h: the cell size in sheet px
          (48 for Retro Diffusion walking presets, 32 for small_sprites). columns/rows: 0 =
          as many cells as fit; trailing empty cells are dropped.
        row_tags: names for the rows, in order — e.g. ["down","right","up","left"] (Retro
          Diffusion's facing order) or ["idle","walk","attack"]; default row0, row1…
        Pixels are kept 1:1 (no resampling). Optional cleanup: key_color '#rrggbb' or 'auto'
          keys an opaque background out; max_colors>0 (k-means over reference_sprite, else
          the first frame) / palette / palette_colors snap every frame to ONE palette.
        Frames are cropped to ONE box shared by every row so all facings align.
        out_columns: 0 keeps the sheet's row structure; N repacks N per row. scale: integer
          nearest-neighbor export scale. Returns {sheet, gif, atlas, tags, tag_gifs, frames…}."""
        _check_palette(palette)
        src = await storage.resolve_input(sheet, cfg=cfg, kind="image")
        ref_bytes = (await storage.resolve_input(reference_sprite, cfg=cfg, kind="image")).data \
            if reference_sprite else None
        safe = storage.safe_filename(name)
        res = await asyncio.to_thread(
            sa.build_from_sheet, src.data, int(frame_w), int(frame_h), columns=columns, rows=rows,
            row_tags=row_tags, reference=ref_bytes, max_colors=max_colors, palette=palette,
            palette_colors=palette_colors, key_color=key_color, outline=outline,
            outline_color=outline_color, out_columns=out_columns, padding=padding, scale=scale,
            fps=fps, name=safe)
        return await _deliver(res, name=safe, project=project, subpath=subpath, cfg=cfg)

    async def _run_animate_sprite_rd_job(job_id, src_png, ref_png, prompt, style, size, seed,
                                         lock_palette, refine_kwargs, name):
        """Retro Diffusion engine: reference sprite → rd-animation on Replicate → fixed-cell
        sheet → tagged bundle. One prediction, never retried (a blind retry double-charges)."""
        try:
            token = cfg.replicate_api_token
            job = jobs.get(job_id)
            spec = RD_STYLES[style]
            results = []
            ref_res = await storage.save_result(ref_png, project=job["project"], subpath=job["subpath"],
                                                filename=f"{name}-reference", ext="png", cfg=cfg)
            ref_res["kind"] = "reference"
            results.append(ref_res)
            jobs.update(job_id, message="uploading reference to Replicate…", results=results)
            ref_url = await R.upload_file(ctx.http, token, ref_png, f"{name}-reference.png")
            payload = {"prompt": prompt, "style": style, "width": size, "height": size,
                       "input_image": ref_url, "return_spritesheet": True}
            if seed is not None:
                payload["seed"] = int(seed)
            pred = await R.create_prediction(ctx.http, token, RD_MODEL, payload, wait=60)
            jobs.update(job_id, operation_name=pred.get("id"),
                        message=f"rd-animation {style} on Replicate ({pred.get('status')})…")
            pred = await R.wait_prediction(ctx.http, token, pred, budget_s=900, poll_s=5)
            if pred.get("status") != "succeeded":
                raise g.GenerationError(f"Replicate prediction {pred.get('status')}: "
                                        f"{pred.get('error') or 'no error detail'}")
            urls = R.collect_file_urls(pred.get("output"))
            if not urls:
                raise g.GenerationError("rd-animation returned no file")
            sheet_png = await R.download_output(ctx.http, token, urls[0], cfg.max_video_mb * 1024 * 1024)
            raw = await storage.save_result(sheet_png, project=job["project"], subpath=job["subpath"],
                                            filename=f"{name}-rd-sheet", ext="png", cfg=cfg)
            raw["kind"] = "rd_sheet"; raw["engine"] = f"{RD_MODEL} {style}"
            raw["prediction_id"] = pred.get("id"); raw["frame_size"] = [size, size]
            metrics = pred.get("metrics") or {}
            if metrics.get("predict_time"):
                raw["predict_time_s"] = round(metrics["predict_time"], 2)
            results.append(raw)
            jobs.update(job_id, message="slicing sheet + packing atlas…", results=results)
            res = await asyncio.to_thread(
                sa.build_from_sheet, sheet_png, size, size, row_tags=spec["rows"],
                reference=src_png if lock_palette else None,
                max_colors=refine_kwargs["max_colors"] if lock_palette else 0,
                palette=refine_kwargs["palette"], palette_colors=refine_kwargs["palette_colors"],
                outline=refine_kwargs["outline"], outline_color=refine_kwargs["outline_color"],
                out_columns=refine_kwargs["columns"], padding=refine_kwargs["padding"],
                scale=refine_kwargs["scale"], fps=refine_kwargs["fps"], name=name)
            sheet = await _deliver(res, name=name, project=job["project"], subpath=job["subpath"], cfg=cfg,
                                   extra={"kind": "animation", "engine": f"{RD_MODEL} {style}"})
            results.append(sheet)
            rep = res["report"]
            jobs.update(job_id, status="done", results=results,
                        message=f"complete: {rep['frames_out']} frames @ {rep['frame_size'][0]}x"
                                f"{rep['frame_size'][1]}, {len(rep['tags'])} tags ({RD_MODEL} {style})")
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    async def _run_animate_sprite_rd_advanced_job(job_id, src_png, frame_png, size, prompt, action, frames, seed,
                                                  lock_palette, refine_kwargs, name):
        """Retro Diffusion OFFICIAL API, advanced animation: YOUR frame, animated at its native
        size → one-row strip → bundle. One submit, never retried (it bills the balance)."""
        try:
            key = cfg.rd_api_key
            job = jobs.get(job_id)
            w, h = size
            results = []
            ref_res = await storage.save_result(frame_png, project=job["project"], subpath=job["subpath"],
                                                filename=f"{name}-frame", ext="png", cfg=cfg)
            ref_res["kind"] = "reference"; ref_res["note"] = "the exact frame Retro Diffusion animated"
            results.append(ref_res)
            payload = RD.advanced_payload(prompt, action, frame_png, w, h, frames=frames, seed=seed,
                                          spritesheet=True)
            jobs.update(job_id, message=f"submitting rd_advanced_animation__{action} ({w}x{h}, "
                                        f"{payload['frames_duration']} frames)…", results=results)
            submitted = await RD.submit(ctx.http, key, payload)
            jobs.update(job_id, operation_name=submitted.get("task_id"),
                        message=f"Retro Diffusion task {submitted.get('status')}…")
            result = await RD.wait_task(ctx.http, key, submitted, budget_s=900, poll_s=3,
                                        on_status=lambda st: jobs.update(job_id, message=f"Retro Diffusion: {st}…"))
            strip_png = await RD.first_image(ctx.http, key, result)
            raw = await storage.save_result(strip_png, project=job["project"], subpath=job["subpath"],
                                            filename=f"{name}-rd-strip", ext="png", cfg=cfg)
            raw["kind"] = "rd_sheet"; raw["engine"] = f"retrodiffusion.ai rd_advanced_animation__{action}"
            raw["task_id"] = submitted.get("task_id"); raw["frame_size"] = [w, h]
            for k in ("balance_cost", "remaining_balance", "model"):
                if result.get(k) is not None:
                    raw[k] = result[k]
            results.append(raw)
            jobs.update(job_id, message="slicing strip + packing atlas…", results=results)
            res = await asyncio.to_thread(
                sa.build_from_sheet, strip_png, w, h, row_tags=[action],
                reference=src_png if lock_palette else None,
                max_colors=refine_kwargs["max_colors"] if lock_palette else 0,
                palette=refine_kwargs["palette"], palette_colors=refine_kwargs["palette_colors"],
                outline=refine_kwargs["outline"], outline_color=refine_kwargs["outline_color"],
                out_columns=refine_kwargs["columns"], padding=refine_kwargs["padding"],
                scale=refine_kwargs["scale"], fps=refine_kwargs["fps"], name=name)
            sheet = await _deliver(res, name=name, project=job["project"], subpath=job["subpath"], cfg=cfg,
                                   extra={"kind": "animation", "engine": raw["engine"]})
            results.append(sheet)
            rep = res["report"]
            jobs.update(job_id, status="done", results=results,
                        message=f"complete: {rep['frames_out']} frames @ {rep['frame_size'][0]}x"
                                f"{rep['frame_size'][1]} ({raw['engine']})")
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    async def _run_animate_sprite_job(job_id, start_png, prep_info, motion, neg, w, h, length, steps, seed,
                                      fps_video, refine_kwargs, name):
        try:
            backends = [
                {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
                {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url,
                 "label": "maingamingrig"},
            ]
            job = jobs.get(job_id)
            spec = g.LOCAL_VIDEO_MODELS["wan-i2v"]
            chosen, sel = await g.select_comfy(ctx.http, backends, require_unets=[spec["high"], spec["low"]])
            if not chosen:
                raise g.GenerationError(f"No I2V backend available ({sel}) — all busy/gaming or missing Wan I2V")
            results = []
            start_res = await storage.save_result(start_png, project=job["project"], subpath=job["subpath"],
                                                  filename=f"{name}-start", ext="png", cfg=cfg)
            start_res["kind"] = "start_frame"; start_res["prepare"] = prep_info
            results.append(start_res)
            jobs.update(job_id, message=f"animating on {sel}…", results=results)
            mp4 = await g.call_comfy_i2v(ctx.http, chosen, start_png, motion, model="wan-i2v",
                                         negative_prompt=neg, width=w, height=h, length=length,
                                         steps=steps, seed=seed, fps=fps_video)
            vid_res = await storage.save_result(mp4, project=job["project"], subpath=job["subpath"],
                                                filename=f"{name}-i2v", ext="mp4", cfg=cfg)
            vid_res["kind"] = "video"; vid_res["engine"] = f"wan-i2v@{sel}"
            results.append(vid_res)
            jobs.update(job_id, message="sampling frames + refining to pixel art…", results=results)
            pngs, times, info = await _frames_from_video(mp4, "video/mp4", refine_kwargs.pop("frames"),
                                                         refine_kwargs.pop("start"), refine_kwargs.pop("end"),
                                                         refine_kwargs.pop("loop"))
            res = await asyncio.to_thread(sa.build_animation, pngs, reference=start_png, name=name,
                                          source_times=times, **refine_kwargs)
            sheet = await _deliver(res, name=name, project=job["project"], subpath=job["subpath"], cfg=cfg,
                                   extra={"kind": "animation", "sampled_times_s": times, "video": info})
            results.append(sheet)
            jobs.update(job_id, status="done",
                        message=f"complete: {res['report']['frames_out']} frames @ "
                                f"{res['report']['frame_size'][0]}x{res['report']['frame_size'][1]} (wan-i2v@{sel})",
                        results=results)
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def animate_sprite(
        image: str,
        project: str,
        motion: str = "walk cycle, walking in place",
        engine: str = "wan",
        style: str = "four_angle_walking",
        action: str | None = None,
        subject: str | None = None,
        size: int = 0,
        lock_palette: bool = False,
        frames: int = 8,
        seconds: float = 2.0,
        fps: int = 12,
        key_color: str = sa.DEFAULT_KEY_COLOR,
        key_tolerance: int = sa.DEFAULT_KEY_TOLERANCE,
        max_colors: int = sa.DEFAULT_MAX_COLORS,
        palette: str | None = None,
        palette_colors: list[str] | None = None,
        cell_size: int = 0,
        outline: str = "none",
        outline_color: str = "#000000",
        dedupe: bool = True,
        columns: int = 0,
        padding: int = 0,
        scale: int = 1,
        negative_prompt: str | None = None,
        steps: int | None = None,
        seed: int | None = None,
        name: str | None = None,
        subpath: str | None = None,
    ) -> dict:
        """Animate ONE pixel sprite into a game-ready animation, end to end: sprite sheet PNG
        + Aseprite/Phaser atlas JSON + looping GIF + frame PNGs + HTML preview, one async job.
        Returns a job_id immediately; poll job_status. Two engines:

        engine 'wan' (default, LOCAL and FREE, a few minutes): sprite → Wan 2.2 I2V video →
          evenly sampled frames → every frame refined back to TRUE pixel art on the sprite's
          own grid + palette. Any `motion` you can describe, ONE facing (the sprite's).
          Results: prepared start frame, raw I2V video, bundle.
        engine 'retro-diffusion' (Astropulse's rd-animation on Replicate, ~$0.07-0.25 and
          ~30 s): the sprite is the REFERENCE for a 4-direction preset that comes back as
          true pixel art already — `style` 'four_angle_walking' (4 facings x 4 walk frames,
          48x48), 'walking_and_idle' (4 facings x 3 walk + 1 idle, 48x48), 'small_sprites'
          (4 facings x 5 action poses, 32x32) or 'vfx' (effects, `size` 24-96, one row).
          The model re-draws the character in its own style at that size (a rendition, not
          your exact pixels); rows are tagged down/right/up/left in the atlas and each facing
          also gets its own GIF (tag_gifs). `subject` is the text prompt ("armored knight with
          sword and shield"; defaults to `motion`). lock_palette=true snaps the result to THIS
          sprite's palette (max_colors/palette/palette_colors) so it matches your other
          assets. Results: reference, raw rd sheet, bundle. Nothing else below applies.
          `action` (needs RETRO_DIFFUSION_API_KEY on the service — the official API, not
          Replicate) animates YOUR EXACT frame instead of re-drawing it: 'walking' | 'idle' |
          'jump' | 'crouch' | 'attack' | 'destroy' | 'custom_action' | 'subtle_motion'
          (~$0.14, custom/subtle $0.25), sprite kept at its native 32-256 px size (tiny
          sprites are integer-upscaled to 32), `frames` snapped to 4/6/8/10/12/16, `motion`
          as the motion text ("slow, heavy steps"), one row tagged with the action.

        image: the sprite — https URL or '<Project>/<path>'. Transparent PNGs are ideal
          (a generate_image/generate_local pixel sprite after remove_background, or a
          pixel_refine output); an opaque sprite on a flat background works too — it is
          composited on key_color (default magenta #FF00FF) and integer-upscaled to the
          512x512 video frame with NO blur, so the model sees crisp pixels and never
          stretches the sprite. Avoid key_color colors that appear in the sprite.
        motion: what the sprite does — "walk cycle", "idle breathing, cape swaying",
          "sword slash attack", "jump", "run cycle", "death animation". The camera-lock /
          flat-background / loop wording is added for you (negative_prompt overrides ours).
        frames: frames to sample (8 for walk/idle, 6 for attacks, up to 32).
        seconds: clip length to generate (2-5; longer = more distinct poses).
        fps: playback rate for the atlas + GIF (not the video's).
        max_colors / palette / palette_colors / cell_size / outline / dedupe / columns /
          padding / scale: see video_to_sprite_sheet — the palette is taken from THIS
          sprite by default, so the frames can't drift to new colors.
        TIP: the raw video is in the results — re-run video_to_sprite_sheet on it with
          other frame counts / palettes without re-generating."""
        _check_palette(palette)
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        storage.validate_project(project, cfg=cfg)
        refine_common = dict(key_color=key_color, key_tolerance=key_tolerance, max_colors=max_colors,
                             palette=palette, palette_colors=palette_colors, cell_size=cell_size,
                             outline=outline, outline_color=outline_color, dedupe=dedupe, columns=columns,
                             padding=padding, scale=scale, fps=fps)
        engine = (engine or "wan").strip().lower().replace("_", "-")
        if engine in ("retro-diffusion", "rd", "astro", "retrodiffusion"):
            if not cfg.replicate_api_token:
                return {"error": "REPLICATE_API_TOKEN is not configured on the forge service "
                                 "(engine 'retro-diffusion' runs on Replicate)"}
            if action:
                action = action.strip().lower().replace("-", "_").replace(" ", "_")
                if action not in RD.ADVANCED_ACTIONS:
                    return {"error": f"action must be one of {', '.join(RD.ADVANCED_ACTIONS)}"}
                if not cfg.rd_api_key:
                    return {"error": "action=… needs the official Retro Diffusion API: set "
                                     "RETRO_DIFFUSION_API_KEY on the forge service (retrodiffusion.ai). "
                                     "Without it use style=… (Replicate presets) or engine 'wan'."}
                frame_png, (w, h), prep_info = await asyncio.to_thread(sa.prepare_native_frame, src.data,
                                                                       RD.MIN_SIZE, RD.MAX_SIZE)
                prompt = (motion or "").strip() or action.replace("_", " ")
                safe = storage.safe_filename(name or f"{(subject or action)[:24]}-{action}")
                job = jobs.create(kind="animate-sprite", model=f"rd_advanced_animation__{action}", prompt=prompt,
                                  project=project, subpath=subpath, filename=safe)
                asyncio.create_task(_run_animate_sprite_rd_advanced_job(
                    job["id"], src.data, frame_png, (w, h), prompt, action, frames, seed, lock_palette,
                    refine_common, safe))
                return {"job_id": job["id"], "status": "running",
                        "engine": f"retrodiffusion.ai rd_advanced_animation__{action}", "prompt": prompt,
                        "frame_size": [w, h], "frames": RD.snap_frames(frames), "prepare": prep_info,
                        "cost_usd": RD.ADVANCED_ACTIONS[action],
                        "note": "Official Retro Diffusion API (async task, ~30-90 s). Poll job_status; the "
                                "bundle is the last entry in results."}
            if style not in RD_STYLES:
                return {"error": f"style must be one of {', '.join(RD_STYLES)}"}
            spec = RD_STYLES[style]
            rd_size = spec["size"] or max(24, min(96, int(size) or 64))
            prompt = (subject or motion or "").strip() or "pixel art game sprite"
            # The model wants an opaque RGB reference: flatten on white (its own background
            # color), integer-upscaled so small sprites read clearly.
            ref_png, prep_info = await asyncio.to_thread(sa.prepare_reference, src.data, (256, 256), "#FFFFFF")
            safe = storage.safe_filename(name or f"{prompt[:24]}-{style}")
            job = jobs.create(kind="animate-sprite", model=f"{RD_MODEL} {style}", prompt=prompt,
                              project=project, subpath=subpath, filename=safe)
            asyncio.create_task(_run_animate_sprite_rd_job(job["id"], src.data, ref_png, prompt, style,
                                                           rd_size, seed, lock_palette, refine_common, safe))
            return {"job_id": job["id"], "status": "running", "engine": f"{RD_MODEL} {style}",
                    "prompt": prompt, "prepare": prep_info, "layout": spec["note"],
                    "note": "Retro Diffusion on Replicate takes ~30-60 s. Poll with job_status; the "
                            "animation bundle is the last entry in results (tags + tag_gifs per facing)."}
        if engine != "wan":
            return {"error": "engine must be 'wan' (local I2V) or 'retro-diffusion'"}
        w, h = g.VIDEO_AR["1:1"]
        start_png, prep_info = await asyncio.to_thread(sa.prepare_reference, src.data, (w, h), key_color)
        fps_video = g.LOCAL_VIDEO_MODELS["wan-i2v"]["fps"]
        seconds = max(1.0, min(6.0, float(seconds)))
        length = max(17, int(round(seconds * fps_video / 4)) * 4 + 1)  # Wan wants 4n+1 frames
        prompt = SPRITE_MOTION_TEMPLATE.format(motion=motion.strip() or "idle animation", key=key_color)
        neg = negative_prompt or SPRITE_NEGATIVE
        safe = storage.safe_filename(name or f"{motion[:24]}-sprite")
        job = jobs.create(kind="animate-sprite", model="wan-i2v", prompt=prompt, project=project,
                          subpath=subpath, filename=safe)
        refine_kwargs = dict(frames=max(1, min(32, int(frames))), start=None, end=None, loop=True,
                             **refine_common)
        asyncio.create_task(_run_animate_sprite_job(job["id"], start_png, prep_info, prompt, neg, w, h,
                                                    length, steps, seed, fps_video, refine_kwargs, safe))
        return {"job_id": job["id"], "status": "running", "prepare": prep_info,
                "motion_prompt": prompt,
                "note": "Local I2V (Wan 2.2) + refine takes a few minutes. Poll with job_status; the "
                        "animation bundle is the last entry in results."}
