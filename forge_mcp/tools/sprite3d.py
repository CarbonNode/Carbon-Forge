"""3D → pixel sprite tools — the pre-rendered-sprite pipeline (Warcraft II, Diablo,
Donkey Kong Country did it by hand in the 90s; this does it unattended).

  bake_sprite_sheet     — an animated GLB you already have (a Meshy rig/animate
                          result, Mixamo, your own Blender export) → N-direction,
                          per-action sprite sheet bundle (headless Blender render →
                          medoid downsample → ONE palette → ONE crop box → atlas)
  character_to_sprites  — the whole chain from a concept image (or a text prompt):
                          Meshy image→3D → auto-rig → library clips (idle / walk /
                          attack / death …) → bake every clip → ONE sheet where every
                          row is `<action>_<facing>` and all rows share palette + box
  meshy_actions         — the clip names character_to_sprites understands, credit
                          prices and the live Meshy balance

Why 3D-first for ANIMATED units: 2D generators (diffusion, video, PixelLab's
puppet rig) drift between frames and cannot keep 8 facings consistent. A rig
cannot drift — silhouettes, proportions and the ground line are consistent by
construction. The pixel-art look (grid, palette, outline) is applied after.
The trade: the result reads as pre-rendered ("Warcraft II"), not hand-clustered.
"""
import asyncio

from backend import pixel_art as pa
from backend import sprite_bake as sb
from forge_mcp import generation as g
from forge_mcp import meshy_api as M
from forge_mcp import storage
from forge_mcp.tools.sprite import _check_palette, _deliver

DEFAULT_ACTIONS = ["idle", "walk", "attack", "death"]
# clips that are cycles (drop the last frame — it equals the first); everything else is a one-shot
LOOPING = {"idle", "idle2", "idle3", "walk", "walk_woman", "walk_fight", "walk_back", "run", "run_fast",
           "jog", "combat_idle", "dance"}


def _is_loop(action_name, loop):
    if loop in (True, False):
        return bool(loop)
    return str(action_name).lower() in LOOPING


def register(mcp, ctx):
    cfg, jobs = ctx.cfg, ctx.jobs

    def _compose_kwargs(supersample, max_colors, palette, palette_colors, dither, outline, outline_color,
                        margin, fps, columns, padding, scale):
        _check_palette(palette)
        if dither not in pa.DITHER_MODES:
            raise ValueError(f"dither must be one of {', '.join(pa.DITHER_MODES)}")
        if outline not in ("none", "rounded", "sharp"):
            raise ValueError("outline must be none, rounded or sharp")
        return dict(supersample=max(1, min(8, int(supersample))), max_colors=int(max_colors), palette=palette,
                    palette_colors=palette_colors, dither=dither, outline=outline, outline_color=outline_color,
                    margin=max(0, int(margin)), fps=max(1, int(fps)), columns=int(columns), padding=int(padding),
                    scale=max(1, min(32, int(scale))))

    def _render_kwargs(cell, supersample, directions, frames, elevation, engine, samples):
        if int(directions) not in sb.bs.DIRECTION_TAGS:
            raise ValueError(f"directions must be one of {sb.bs.SUPPORTED_DIRECTIONS}")
        return dict(cell=max(8, min(512, int(cell))), supersample=max(1, min(8, int(supersample))),
                    directions=int(directions), frames=max(1, min(64, int(frames))),
                    elevation_deg=float(elevation), engine=engine, samples=int(samples))

    def _require_blender():
        if sb.find_blender(bpy_python=cfg.bpy_python or None, blender_bin=cfg.blender_bin or None) is None:
            raise sb.BakeError("No Blender on the forge service: the image needs the bpy venv "
                               "(FORGE_BPY_PYTHON) or a blender binary (FORGE_BLENDER_BIN)")

    def _runner():
        return sb.find_blender(bpy_python=cfg.bpy_python or None, blender_bin=cfg.blender_bin or None)

    async def _finish_sheet(job_id, rows, manifests, compose_kw, name, results, extra):
        job = jobs.get(job_id)
        jobs.update(job_id, message=f"composing {len(rows)} rows → sheet (palette lock, shared crop)…",
                    results=results)
        res = await asyncio.to_thread(sb.compose, rows, name=name, **compose_kw)
        res["report"]["bake"] = sb.bake_report(manifests)
        sheet = await _deliver(res, name=name, project=job["project"], subpath=job["subpath"], cfg=cfg,
                               extra=dict({"kind": "sheet"}, **extra))
        results.append(sheet)
        rep = res["report"]
        return sheet, (f"complete: {rep['frames_out']} frames @ {rep['frame_size'][0]}x{rep['frame_size'][1]}, "
                       f"{rep['rows']} rows ({', '.join(t['name'] for t in rep['tags'][:8])}"
                       f"{'…' if len(rep['tags']) > 8 else ''})")

    async def _run_bake_job(job_id, glb_bytes, render_kw, compose_kw, actions, loop, action_prefix,
                            clip_actions, name):
        try:
            results = []
            jobs.update(job_id, message=f"rendering {render_kw['directions']} directions x "
                                        f"{render_kw['frames']} frames per action in Blender…")
            spec = sb.render_spec(actions=actions, loop=bool(loop), clip_actions=clip_actions, **render_kw)
            manifest, rows = await sb.render_rows(glb_bytes, spec, action_prefix=action_prefix, runner=_runner())
            _, msg = await _finish_sheet(job_id, rows, [manifest], compose_kw, name, results,
                                         {"engine": f"blender-bake {manifest.get('blender')}",
                                          "actions": manifest["actions_baked"]})
            jobs.update(job_id, status="done", message=msg, results=results)
        except Exception as e:  # noqa: BLE001 — the job row is the error channel
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def bake_sprite_sheet(
        model: str,
        project: str,
        actions: list[str] | None = None,
        directions: int = 8,
        cell: int = 64,
        frames: int = 8,
        elevation: float = 35.0,
        loop: bool | None = None,
        action_prefix: str | None = None,
        supersample: int = 4,
        max_colors: int = 16,
        palette: str | None = None,
        palette_colors: list[str] | None = None,
        outline: str = "none",
        outline_color: str = "#000000",
        dither: str = "none",
        margin: int = 0,
        fps: int = 10,
        columns: int = 0,
        padding: int = 0,
        scale: int = 1,
        engine: str = "cycles",
        samples: int = 16,
        clip_actions: dict | None = None,
        name: str = "sprite",
        subpath: str | None = None,
    ) -> dict:
        """Bake an ANIMATED 3D character (.glb with an armature + actions — a Meshy rig /
        animate result, a Mixamo export, your own Blender file) into a pixel-art sprite
        sheet from N facings: headless Blender renders every action from every direction
        with an orthographic camera, frames are medoid-downsampled to `cell` logical px,
        snapped to ONE palette and cropped to ONE shared box (the ground line never moves),
        then packed as sheet PNG + Aseprite/Phaser atlas JSON (one frameTag per
        `<action>_<facing>` row) + GIF per row + frame PNGs + HTML preview. Async: returns
        a job_id, poll job_status. Takes seconds to a few minutes (CPU render).

        model: https URL or '<Project>/<path>' to a .glb. actions: names (or substrings) of
          the GLB's actions to bake, null = all of them. action_prefix: rename the action in
          the row tags (Meshy names clips 'Armature|Casual_Walk' — pass 'walk').
        directions: 1, 2, 4, 8 or 16. Facing 0 is toward the camera ('s'); tags go
          s, se, e, ne, n, nw, w, sw (the model turns counter-clockwise).
        cell: logical sprite size — the WHOLE animation's bounds fit a cell x cell box, so
          the standing figure is a bit smaller than `cell` (48-64 reads as Warcraft II).
        frames: per action. loop: true = cycle (last frame dropped, it equals the first),
          false = one-shot (keeps the final pose — attacks, deaths); null = guess from the
          action name (idle/walk/run loop, everything else one-shot).
        elevation: camera pitch in degrees (0 side view, 35 classic RTS, 60 near top-down).
        supersample: render at cell*N and medoid-downsample (4 = clean; 1 = raw render).
        max_colors / palette (pico8, nes, gb_*, c64, sfc_*…) / palette_colors lock colors
          across EVERY frame; outline 'sharp'|'rounded' adds a 1px outline; dither as
          pixel_refine. engine 'cycles' (CPU, headless-safe) or 'workbench' (flat, needs GL).
        clip_actions: {"<action>": {"start": f, "end": f}} to trim an action's frame range.
        Returns {job_id}; the job's results carry the bundle {sheet, gif, atlas, tags,
        tag_gifs, frames, preview, layout, report}."""
        compose_kw = _compose_kwargs(supersample, max_colors, palette, palette_colors, dither, outline,
                                     outline_color, margin, fps, columns, padding, scale)
        render_kw = _render_kwargs(cell, supersample, directions, frames, elevation, engine, samples)
        _require_blender()
        storage.validate_project(project, cfg=cfg)
        src = await storage.resolve_input(model, cfg=cfg, kind="model")
        safe = storage.safe_filename(name)
        loop_flag = loop if loop is not None else (_is_loop(action_prefix, None) if action_prefix else True)
        job = jobs.create(kind="bake", model="blender-bake", prompt=f"{model} → {directions} dirs",
                          project=project, subpath=subpath, filename=safe)
        asyncio.create_task(_run_bake_job(job["id"], src.data, render_kw, compose_kw, actions, loop_flag,
                                          action_prefix, clip_actions, safe))
        return {"job_id": job["id"], "status": "running",
                "note": "Rendering in Blender. Poll job_status; the sheet bundle lands in results."}

    async def _run_character_job(job_id, ref_png, ref_ext, prompt, actions, height_meters, target_polycount,
                                 render_kw, compose_kw, loop, name):
        try:
            key = cfg.meshy_api_key
            job = jobs.get(job_id)
            project, subpath = job["project"], job["subpath"]
            results = []
            credits = 0

            def _status(stage):
                return lambda m: jobs.update(job_id, message=f"{stage}: Meshy {m}…")

            # 1. model
            if ref_png is not None:
                ref = await storage.save_result(ref_png, project=project, subpath=subpath,
                                                filename=f"{name}-concept", ext=ref_ext, cfg=cfg)
                ref["kind"] = "concept"
                results.append(ref)
                jobs.update(job_id, message="submitting image→3D to Meshy…", results=results)
                model_task_id = await M.image_to_3d(ctx.http, key, ref["url"], should_texture=True,
                                                    target_polycount=target_polycount)
                model_kind = "image"
                task = await M.wait_task(ctx.http, key, "image", model_task_id, on_status=_status("model"))
            else:
                jobs.update(job_id, message="submitting text→3D (preview) to Meshy…")
                prev_id = await M.text_to_3d(ctx.http, key, prompt, mode="preview",
                                             target_polycount=target_polycount)
                task = await M.wait_task(ctx.http, key, "text", prev_id, on_status=_status("preview"))
                credits += int(task.get("consumed_credits") or 0)
                model_task_id = await M.text_to_3d(ctx.http, key, prompt, mode="refine", preview_task_id=prev_id)
                model_kind = "text"
                task = await M.wait_task(ctx.http, key, "text", model_task_id, on_status=_status("texture"))
            credits += int(task.get("consumed_credits") or 0)
            glb_url = M.pick_glb(model_kind, task)
            if not glb_url:
                raise g.GenerationError(f"Meshy {model_kind} task {model_task_id} has no GLB url")
            max_bytes = cfg.max_video_mb * 1024 * 1024
            model_glb = await M.download(ctx.http, glb_url, max_bytes)
            mres = await storage.save_result(model_glb, project=project, subpath=subpath,
                                             filename=f"{name}-model", ext="glb", cfg=cfg)
            mres.update({"kind": "model", "meshy_task_id": model_task_id, "meshy_kind": model_kind,
                         "thumbnail_url": task.get("thumbnail_url")})
            results.append(mres)

            # 2. rig
            jobs.update(job_id, message="auto-rigging (Meshy)…", results=results)
            rig_id = await M.rig(ctx.http, key, input_task_id=model_task_id, height_meters=height_meters)
            rig_task = await M.wait_task(ctx.http, key, "rig", rig_id, on_status=_status("rig"))
            credits += int(rig_task.get("consumed_credits") or 0)
            basic = M.rig_basic_animations(rig_task)
            rigged_url = M.pick_glb("rig", rig_task)
            if rigged_url:
                rigged = await M.download(ctx.http, rigged_url, max_bytes)
                rres = await storage.save_result(rigged, project=project, subpath=subpath,
                                                 filename=f"{name}-rigged", ext="glb", cfg=cfg)
                rres.update({"kind": "rigged", "meshy_task_id": rig_id, "free_clips": sorted(basic)})
                results.append(rres)

            # 3. clips — free rig clips first, one animate task per remaining action
            clip_urls = {}
            pending = []
            for a in actions:
                if a in basic:
                    clip_urls[a] = basic[a]
                else:
                    pending.append(a)
            if pending:
                jobs.update(job_id, message=f"animating {', '.join(pending)} (Meshy, {len(pending)} clips)…",
                            results=results)
                ids = await asyncio.gather(*[M.animate(ctx.http, key, rig_id, M.resolve_action(a))
                                             for a in pending])
                for a, tid in zip(pending, ids):
                    t = await M.wait_task(ctx.http, key, "animate", tid, on_status=_status(f"clip {a}"))
                    credits += int(t.get("consumed_credits") or 0)
                    url = M.pick_glb("animate", t)
                    if not url:
                        raise g.GenerationError(f"Meshy animate task {tid} ({a}) has no GLB url")
                    clip_urls[a] = url

            # 4. bake every clip, compose ONCE (shared palette + crop across all actions)
            rows, manifests = [], []
            for i, a in enumerate(actions):
                jobs.update(job_id, message=f"rendering clip {a} ({i + 1}/{len(actions)}) in Blender…",
                            results=results)
                clip = await M.download(ctx.http, clip_urls[a], max_bytes)
                cres = await storage.save_result(clip, project=project, subpath=subpath,
                                                 filename=f"{name}-{a}", ext="glb", cfg=cfg)
                cres.update({"kind": "clip", "action": a, "source": "rig (free)" if a in basic else "animate"})
                results.append(cres)
                spec = sb.render_spec(actions=None, loop=_is_loop(a, loop), **render_kw)
                manifest, clip_rows = await sb.render_rows(clip, spec, action_prefix=a, runner=_runner())
                manifests.append(manifest)
                rows.extend(clip_rows)
            _, msg = await _finish_sheet(job_id, rows, manifests, compose_kw, name, results,
                                         {"engine": "meshy + blender-bake", "actions": list(actions),
                                          "meshy_credits_used": credits})
            jobs.update(job_id, status="done", message=f"{msg}; {credits} Meshy credits", results=results)
        except Exception as e:  # noqa: BLE001
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def character_to_sprites(
        project: str,
        image: str | None = None,
        prompt: str | None = None,
        actions: list[str] | None = None,
        directions: int = 8,
        cell: int = 64,
        frames: int = 8,
        elevation: float = 35.0,
        height_meters: float = 1.7,
        target_polycount: int = 30000,
        loop: bool | None = None,
        supersample: int = 4,
        max_colors: int = 16,
        palette: str | None = None,
        palette_colors: list[str] | None = None,
        outline: str = "none",
        outline_color: str = "#000000",
        dither: str = "none",
        margin: int = 0,
        fps: int = 10,
        columns: int = 0,
        padding: int = 0,
        scale: int = 1,
        engine: str = "cycles",
        samples: int = 16,
        name: str = "unit",
        subpath: str | None = None,
    ) -> dict:
        """ONE concept image (or a text prompt) → a complete animated game unit as pixel
        sprites: Meshy turns the image into a textured 3D character, auto-rigs it, bakes
        library clips onto the rig (idle / walk / attack / death by default), and headless
        Blender renders every clip from `directions` facings into ONE sprite sheet — every
        row is `<action>_<facing>` (walk_s, walk_se, … death_sw), all rows share one palette
        and one crop box, Aseprite/Phaser atlas + per-row GIFs + preview included.
        Async: returns a job_id, poll job_status (5-15 minutes: Meshy image→3D ~3-5 min,
        rig ~2 min, clips ~1 min each, the bake seconds to a minute per clip).

        COSTS MESHY CREDITS: image→3D ~20 (text: preview 20 + texture 10), rig 5, and 3 per
        clip that is not free — `walk` and `run` come free with the rig. Default actions ≈ 34
        credits; see meshy_actions for the balance. Tasks are never retried.

        image: https URL or '<Project>/<path>' — a clear full-body concept on a plain
          background, facing the viewer (generate_image / edit_image output works well).
          Humanoid only: Meshy's rigger rejects animals, vehicles and props.
        prompt: text→3D instead of an image (image is better — it pins the design).
        actions: clip names from meshy_actions (idle, walk, run, attack, slash, combo, death,
          hurt, jump, cast, combat_idle …) or numeric Meshy action ids. Order = row order.
        height_meters: the character's real height (scales the rig). target_polycount:
          Meshy remesh target (≤300k faces for rigging; 30k is plenty for sprites).
        directions / cell / frames / elevation / loop / supersample / max_colors / palette /
          palette_colors / outline / dither / fps / scale / engine: as bake_sprite_sheet
          (loop=null guesses per action: idle/walk/run cycle, attack/death/hurt one-shot).
        Results: concept, model .glb, rigged .glb, one .glb per clip (re-bake any of them
        later with bake_sprite_sheet, no credits), then the sheet bundle."""
        if not cfg.meshy_api_key:
            return {"error": "MESHY_API_KEY is not configured on the forge service"}
        if bool(image) == bool(prompt):
            return {"error": "Pass exactly one of image or prompt"}
        compose_kw = _compose_kwargs(supersample, max_colors, palette, palette_colors, dither, outline,
                                     outline_color, margin, fps, columns, padding, scale)
        render_kw = _render_kwargs(cell, supersample, directions, frames, elevation, engine, samples)
        acts = [str(a).strip().lower() for a in (actions or DEFAULT_ACTIONS) if str(a).strip()]
        if not acts:
            return {"error": "actions is empty"}
        for a in acts:
            M.resolve_action(a)  # fail fast on a typo before spending credits
        _require_blender()
        storage.validate_project(project, cfg=cfg)
        ref_png, ref_ext = None, "png"
        if image:
            src = await storage.resolve_input(image, cfg=cfg, kind="image")
            ref_png = src.data
            ref_ext = {"image/jpeg": "jpg", "image/webp": "webp"}.get(src.mime, "png")
        safe = storage.safe_filename(name)
        job = jobs.create(kind="sprite3d", model="meshy+blender-bake", prompt=prompt or image,
                          project=project, subpath=subpath, filename=safe)
        asyncio.create_task(_run_character_job(job["id"], ref_png, ref_ext, prompt, acts, float(height_meters),
                                               int(target_polycount), render_kw, compose_kw, loop, safe))
        return {"job_id": job["id"], "status": "running", "actions": acts,
                "note": "Meshy image→3D → rig → clips → Blender bake. 5-15 minutes; poll job_status. "
                        "Every intermediate .glb is saved so a re-bake costs no credits."}

    @mcp.tool()
    async def meshy_actions() -> dict:
        """The animation clips character_to_sprites understands (name → Meshy action_id),
        which are free with a rig, credit prices, and the live Meshy credit balance."""
        bal = await M.balance(ctx.http, cfg.meshy_api_key) if cfg.meshy_api_key else None
        return {
            "actions": dict(M.ACTIONS),
            "free_with_rig": sorted(M.RIG_BASIC),
            "looping": sorted(LOOPING),
            "default_actions": list(DEFAULT_ACTIONS),
            "credits": dict(M.CREDITS),
            "credits_remaining": bal,
            "meshy_key_configured": bool(cfg.meshy_api_key),
            "note": "Any other Meshy library clip works by numeric action_id "
                    "(https://docs.meshy.ai/en/api/animation-library).",
        }
