"""Game-world tools — one generated map image becomes an explorable, style-consistent
world: labeled detections, cutout obstacle sprites, collision grids, style-matched
interior maps behind every door, and a self-contained playable preview.html
(the "capybara.build" workflow)."""
import asyncio
import json
from io import BytesIO

from backend.processing import PipelineOptions
from forge_mcp import engine, storage, worldgen
from forge_mcp import generation as g

# Same-canvas cutout: alpha the crop without trimming, so the sprite redraws in place.
_CUT_OPTS = dict(model="isnet-general-use", edge_smooth=True, edge_strength=60,
                 edge_trim=1, auto_trim=False)


def register(mcp, ctx):
    cfg = ctx.cfg
    jobs = ctx.jobs

    async def _maybe_upscale(map_bytes: bytes):
        """ESRGAN 4x on a pool GPU, downsampled to 2x (plenty at preview zoom, sane file
        size). Any failure — no GPU free, box being gamed on — falls back to the native
        image with a note; the pipeline never dies on fidelity polish."""
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url,
             "label": "maingamingrig"},
        ]
        try:
            chosen, sel = await g.select_comfy(ctx.http, backends)
            if not chosen:
                return map_bytes, f"skipped ({sel})"
            up = await g.call_comfy_upscale(ctx.http, chosen, map_bytes)
            from PIL import Image
            im = Image.open(BytesIO(up))
            im = im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, "PNG")
            return buf.getvalue(), f"esrgan-4x-to-2x@{sel}"
        except Exception as e:
            return map_bytes, f"skipped ({e})"

    async def _sam_mask_urls(map_bytes):
        """Run meta/sam-2 automatic segmentation once for a map. Returns (urls, err) —
        individual full-resolution binary mask URLs (the first output, an annotated
        visualization, is filtered out later by dimension check)."""
        tok = cfg.replicate_api_token
        if not tok:
            return None, "no REPLICATE_API_TOKEN"
        try:
            from forge_mcp import replicate_api as R
            image_url = await R.upload_file(ctx.http, tok, map_bytes, "map.png")
            pred = await R.create_prediction(ctx.http, tok, "meta/sam-2",
                                             {"image": image_url}, wait=60)
            pred = await R.wait_prediction(ctx.http, tok, pred, budget_s=240)
            if pred.get("status") != "succeeded":
                return None, f"sam-2 {pred.get('status')}: {pred.get('error') or ''}"
            return R.collect_file_urls(pred.get("output")), None
        except Exception as e:
            return None, str(e)

    async def _sam_unions(map_bytes, candidates, width, height):
        """Stream SAM masks one at a time, folding each into the crop-box union of every
        candidate it belongs to. Returns (unions: {label: ndarray|None}, err)."""
        import numpy as np
        from PIL import Image
        from forge_mcp import replicate_api as R
        urls, err = await _sam_mask_urls(map_bytes)
        if not urls:
            return {}, err
        boxes = {obj["label"]: worldgen.box_to_px(obj["box_2d"], width, height)
                 for obj in candidates}
        unions = {label: None for label in boxes}
        map_area = width * height
        for url in urls:
            try:
                data = await R.download_output(ctx.http, cfg.replicate_api_token, url,
                                               32 * 1024 * 1024)
                with Image.open(BytesIO(data)) as im:
                    if im.size != (width, height):
                        continue  # the annotated visualization, not a mask
                    mask = np.array(im.convert("L")) > 127
            except Exception:
                continue
            if mask.sum() > 0.35 * map_area:
                continue  # ground/sky segment, never a prop
            for label, box_px in boxes.items():
                unions[label] = worldgen.accumulate_mask_union(unions[label], mask, box_px)
        return unions, None

    async def _rembg_cut(im, box_px):
        """Fallback matte: rembg on the crop, same canvas (redraws in place)."""
        left, top, right, bottom = box_px
        buf = BytesIO()
        im.crop((left, top, right, bottom)).save(buf, "PNG")
        try:
            return await engine.run_pipeline(buf.getvalue(), PipelineOptions(**_CUT_OPTS))
        except Exception:
            return buf.getvalue()  # keep the raw crop rather than dropping the object

    async def _cut_sprites(map_bytes, objects, key, categories=("obstacle",),
                           triage=True, qa=True, min_sprite_height=None):
        """Crop + alpha-cut sprite candidates. Masks come from one SAM-2 pass per map
        (containment-matched per candidate), falling back to rembg; a batched Gemini QA
        judge then keeps/retries/demotes each cutout. Returns
        (size, bundle_files, sprite_files, stats)."""
        from PIL import Image
        im = Image.open(BytesIO(map_bytes)).convert("RGB")
        width, height = im.size
        min_h = worldgen.MIN_SPRITE_HEIGHT if min_sprite_height is None else min_sprite_height
        candidates = []
        for obj in objects:
            if obj["category"] not in categories:
                continue
            if triage and not worldgen.needs_occlusion_sprite(obj, min_h):
                continue
            left, top, right, bottom = worldgen.box_to_px(obj["box_2d"], width, height)
            if right - left < 4 or bottom - top < 4:
                continue
            candidates.append((obj, (left, top, right, bottom)))

        stats = {"candidates": len(candidates), "sam": 0, "rembg": 0}
        unions, sam_err = ({}, "disabled") if not candidates else \
            await _sam_unions(map_bytes, [o for o, _ in candidates], width, height)
        if sam_err:
            stats["sam_fallback"] = sam_err

        cuts = []  # (obj, box_px, png bytes)
        for obj, box_px in candidates:
            union = unions.get(obj["label"])
            if union is not None and worldgen.union_coverage(union, box_px) >= 0.30:
                left, top, right, bottom = box_px
                buf = BytesIO()
                im.crop((left, top, right, bottom)).save(buf, "PNG")
                cut = worldgen.apply_mask_alpha(buf.getvalue(), union)
                stats["sam"] += 1
            else:
                cut = await _rembg_cut(im, box_px)
                stats["rembg"] += 1
            cuts.append((obj, box_px, cut))

        if qa and cuts and cfg.gemini_api_key:
            kept = []
            for chunk_start in range(0, len(cuts), 14):
                chunk = cuts[chunk_start:chunk_start + 14]
                try:
                    verdicts = await worldgen.judge_sprites(
                        ctx.http, cfg.gemini_api_keys,
                        [worldgen.composite_on_magenta(c[2]) for c in chunk],
                        [c[0]["label"] for c in chunk])
                except Exception as e:
                    stats["qa_error"] = str(e)
                    kept.extend(chunk)
                    continue
                for (obj, box_px, cut), verdict in zip(chunk, verdicts):
                    if verdict == "clean":
                        kept.append((obj, box_px, cut))
                    elif verdict == "clipped":
                        # box was tight — retry once with a wider crop
                        wide = worldgen.box_to_px(obj["box_2d"], width, height, pad_frac=0.05)
                        kept.append((obj, wide, await _rembg_cut(im, wide)))
                        stats["qa_retried"] = stats.get("qa_retried", 0) + 1
                    else:  # contaminated / empty -> background already renders it perfectly
                        stats["qa_demoted"] = stats.get("qa_demoted", 0) + 1
            cuts = kept

        bundle_files, sprite_files = [], {}
        for obj, (left, top, right, bottom), cut in cuts:
            name = f"sprite_{key}_{obj['label']}.png"
            bundle_files.append((name, cut))
            sprite_files[obj["label"]] = {"file": name, "crop_px": [top, left, bottom, right]}
        return (width, height), bundle_files, sprite_files, stats

    async def _process_map(map_bytes, key, *, hints, max_objects, detect_model,
                           collision, band_frac, grid_cols, qa=True, min_sprite_height=None):
        """One map through the shared pipeline. Returns (objects, entry, bundle_files,
        stats) — entry is the world.json map entry, mutated later when links attach."""
        detected = await worldgen.detect_scene(
            ctx.http, cfg.gemini_api_keys, map_bytes, "image/png",
            hints=hints, max_objects=max_objects, model=detect_model)
        objects, spawn = detected["objects"], detected["player_spawn"]
        (width, height), bundle_files, sprite_files, stats = await _cut_sprites(
            map_bytes, objects, key, qa=qa, min_sprite_height=min_sprite_height)
        grid = None
        if collision:
            rows = max(8, round(grid_cols * height / max(1, width)))
            grid = worldgen.build_collision(objects, cols=grid_cols, rows=rows,
                                            band_frac=band_frac)
        bundle_files.insert(0, (f"map_{key}.png", map_bytes))
        entry = worldgen.build_map_entry(f"map_{key}.png", width, height, objects, spawn,
                                         grid, sprite_files)
        return objects, entry, bundle_files, stats

    def _relink(entry, objects):
        """Re-run link attachment after objects gained 'link' fields (entries were built
        before links existed for the main map)."""
        by_label = {o["label"]: o for o in objects}
        for eo in entry["objects"]:
            src = by_label.get(eo["label"])
            if src and "link" in src:
                eo["link"] = src["link"]

    async def _build_world(map_bytes, *, project, name, prompt, hints, max_objects,
                           detect_model, collision, preview, band_frac, grid_cols,
                           expand_enterables, upscale, aspect_ratio, subpath,
                           qa=True, min_sprite_height=None,
                           progress=lambda msg: None):
        notes = {}
        qa_stats = {}
        if upscale:
            progress("upscaling map")
            map_bytes, notes["upscale_main"] = await _maybe_upscale(map_bytes)

        progress("detecting objects")
        main_objects, main_entry, bundle_files, qa_stats["main"] = await _process_map(
            map_bytes, "main", hints=hints, max_objects=max_objects,
            detect_model=detect_model, collision=collision,
            band_frac=band_frac, grid_cols=grid_cols, qa=qa,
            min_sprite_height=min_sprite_height)

        maps = {"main": main_entry}
        doors = worldgen.pick_expandable(main_objects, expand_enterables)
        used_keys = {"main"}
        for door in doors:
            key = worldgen.map_key_for(door["label"])
            n = 2
            while key in used_keys:
                key = f"{worldgen.map_key_for(door['label'])}_{n}"
                n += 1
            used_keys.add(key)
            progress(f"painting interior: {key}")
            try:
                interiors = await g.call_gemini_image(
                    ctx.http, cfg.gemini_api_keys,
                    worldgen.interior_prompt(door["label"], prompt),
                    reference_images=[("image/png", map_bytes)],
                    aspect_ratio=aspect_ratio)
                interior_bytes = interiors[0]
            except Exception as e:
                notes[f"interior_{key}"] = f"skipped ({e})"
                continue
            if upscale:
                interior_bytes, notes[f"upscale_{key}"] = await _maybe_upscale(interior_bytes)
            progress(f"worldifying interior: {key}")
            in_objects, in_entry, in_files, qa_stats[key] = await _process_map(
                interior_bytes, key, hints=worldgen.INTERIOR_DETECT_HINTS,
                max_objects=max_objects, detect_model=detect_model, collision=collision,
                band_frac=band_frac, grid_cols=grid_cols, qa=qa,
                min_sprite_height=min_sprite_height)
            entry_spawn, exit_label = worldgen.interior_entry_exit(in_objects)
            return_link = {"to": "main", "spawn": worldgen.door_return_spawn(door["box_2d"])}
            if exit_label:
                for eo in in_entry["objects"]:
                    if eo["label"] == exit_label:
                        eo["link"] = return_link
            else:
                exit_obj = worldgen.synthesize_exit()
                exit_obj["link"] = return_link
                in_entry["objects"].append(exit_obj)
            in_entry["player_spawn"] = entry_spawn
            door["link"] = {"to": key, "spawn": entry_spawn}
            maps[key] = in_entry
            bundle_files.extend(in_files)
        _relink(main_entry, main_objects)

        manifest = worldgen.build_world_manifest(name, maps, "main")
        bundle_files.append(("world.json", json.dumps(manifest, indent=2).encode()))
        if preview:
            bundle_files.append(("preview.html", worldgen.render_preview_html(manifest).encode()))

        progress("saving bundle")
        sub = (subpath or "assets/forge").strip("/\\") + f"/worlds/{storage.safe_filename(name)}"
        saved = await storage.save_bundle(bundle_files, project=project, subpath=sub, cfg=cfg)

        by_cat = {}
        for obj in main_objects:
            by_cat[obj["category"]] = by_cat.get(obj["category"], 0) + 1
        out = {
            "name": name,
            "maps": list(maps),
            "main_objects": by_cat,
            "sprites_cut": sum(1 for f, _ in bundle_files if f.startswith("sprite_")),
            "sprite_stats": qa_stats,
            "world_json_url": saved["files"].get("world.json"),
            "bundle_url": saved["base_url"],
        }
        if preview:
            out["preview_url"] = saved["files"].get("preview.html")
        interesting = {k: v for k, v in notes.items() if v and not v.startswith("esrgan")}
        if interesting:
            out["notes"] = interesting
        if "workspace_dir" in saved:
            out["workspace_dir"] = saved["workspace_dir"]
        if "workspace_write_error" in saved:
            out["workspace_write_error"] = saved["workspace_write_error"]
        return out

    async def _run_world_job(job_id, map_bytes, kwargs):
        try:
            result = await _build_world(
                map_bytes, progress=lambda msg: jobs.update(job_id, message=msg), **kwargs)
            jobs.update(job_id, status="done", message="complete", results=[result])
        except Exception as e:  # job boundary: everything becomes a readable failed status
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def generate_world(prompt: str, project: str, name: str | None = None,
                             image: str | None = None, model: str = "imagen-4",
                             aspect_ratio: str = "16:9", style: str | None = None,
                             hints: str | None = None, max_objects: int = 48,
                             detect_model: str = worldgen.DETECT_MODEL,
                             expand_enterables: int = 3, upscale: bool = True,
                             collision: bool = True, preview: bool = True,
                             qa: bool = True, min_sprite_height: int | None = None,
                             band_frac: float = 0.30, grid_cols: int = 64,
                             subpath: str | None = None) -> dict:
        """Generate a complete EXPLORABLE 2D game world from one prompt. The outdoor scene
        is painted as ONE image (perfect internal style consistency, baked lighting), AI
        detection labels every prop, obstacles become transparent cutout sprites (characters
        walk BEHIND them), a collision grid is derived from obstacle footprints — and then
        the world grows: up to `expand_enterables` detected doors each get a style-matched
        INTERIOR map (painted with the exterior as the style reference), linked both ways.
        The map is ESRGAN-upscaled 2x on a pool GPU when one is free (`upscale=false` to skip).

        Output bundle: map_*.png + sprite_*.png + world.json + a PLAYABLE preview.html —
        open the preview_url: follow camera, WASD/drag to walk, walk into doors to enter
        buildings, G shows collision + door triggers, Z cycles zoom.

        With expand_enterables > 0 (default 3) this runs as an ASYNC JOB — you get
        {job_id} back immediately; poll job_status(job_id) until done (message shows the
        current stage; a 3-interior world takes ~2-4 minutes). Pass expand_enterables=0
        for a fast synchronous single-map world.

        Vague prompts work ("a spooky forest village"); the default style locks a
        hand-painted 16-bit top-down RPG look (Stardew/Eastward lineage). style="" = no
        style wrapper; custom style strings are used verbatim. Pass image (https URL or
        '<Project>/<path>') to worldify existing art instead of generating.

        world.json (format carbon-forge-world/2) is self-describing: maps: {key: {map,
        width, height, player_spawn, objects, collision}}; objects carry box_2d
        [ymin,xmin,ymax,xmax] normalized 0-1000 (y-first), cutouts carry crop_px for exact
        in-place redraw, enterables carry link {to, spawn}. Categories: obstacle,
        zone_blocked, enterable, decor. hints steers detection ("the river is impassable").

        SPRITE POLISH (all default-on): only TALL obstacles become occlusion sprites
        (min_sprite_height, normalized units, default 70 — short props stay painted in
        the background, which renders them perfectly, and keep their collision);
        silhouettes come from one SAM-2 segmentation pass per map (Replicate,
        containment-matched to each object; rembg fallback); a batched Gemini judge then
        QAs every cutout (qa=false to skip) — clipped ones are retried with a wider
        crop, contaminated/empty ones are demoted to background-only. Per-map counts in
        sprite_stats.

        model: imagen-4 | imagen-4-fast | imagen-4-ultra. aspect_ratio: 16:9, 4:3, 1:1...
        band_frac: how much of an obstacle's base blocks walking (0.30 = bottom 30%).
        Cost: ~1 Imagen + 1 Gemini edit per interior + 1 detection + 1 SAM-2 per map (cents)."""
        world_name = storage.safe_filename(name or prompt[:40] or "world")
        if image:
            src = await storage.resolve_input(image, cfg=cfg, kind="image")
            map_bytes = src.data
        else:
            if aspect_ratio not in g.IMAGE_ASPECTS:
                raise g.GenerationError(f"aspect_ratio must be one of {g.IMAGE_ASPECTS}")
            wrapper = worldgen.DEFAULT_WORLD_STYLE if style is None else style.strip()
            styled = worldgen.world_prompt(prompt, wrapper) if wrapper else prompt.strip()
            images = await g.call_imagen(ctx.http, cfg.gemini_api_keys,
                                         g.resolve_image_model(model), styled,
                                         sample_count=1, aspect_ratio=aspect_ratio)
            if not images:
                raise g.GenerationError("Imagen returned no images (prompt may have been refused)")
            map_bytes = images[0]

        kwargs = dict(project=project, name=world_name, prompt=prompt, hints=hints,
                      max_objects=max_objects, detect_model=detect_model,
                      collision=collision, preview=preview, band_frac=band_frac,
                      grid_cols=grid_cols, expand_enterables=max(0, int(expand_enterables)),
                      upscale=upscale, aspect_ratio=aspect_ratio, subpath=subpath,
                      qa=qa, min_sprite_height=min_sprite_height)
        if kwargs["expand_enterables"] > 0:
            job = jobs.create(kind="world", model=model, prompt=prompt[:200],
                              project=project, subpath=subpath, filename=world_name)
            asyncio.create_task(_run_world_job(job["id"], map_bytes, kwargs))
            return {"job_id": job["id"], "status": "running",
                    "note": ("map painted; interiors + worldification running in the "
                             "background — poll job_status(job_id), the result lands in "
                             "results[0] with preview_url")}
        return await _build_world(map_bytes, **kwargs)

    @mcp.tool()
    async def segment_scene(image: str, project: str, hints: str | None = None,
                            max_objects: int = 48,
                            detect_model: str = worldgen.DETECT_MODEL,
                            categories: list[str] | None = None, qa: bool = True,
                            subpath: str | None = None, name: str | None = None) -> dict:
        """Detect and cut out every object in a scene image as individually placed,
        transparent sprites — turn ANY illustration/map/screenshot into layered game
        pieces. AI detection labels each object with a tight box_2d ([ymin,xmin,ymax,xmax]
        normalized 0-1000, y-first) and a category (obstacle / zone_blocked / enterable /
        decor); objects in `categories` (default ['obstacle']) are alpha-cut from their
        crop so they redraw exactly in place (crop_px in the manifest). Writes a bundle:
        the source image + sprite_*.png + world.json manifest. hints steers detection.
        Cutout masks come from a SAM-2 pass (rembg fallback) and a Gemini judge QAs each
        sprite (qa=false to skip); no tallness triage here — you asked for these
        categories, you get them all. For the full explorable pipeline (interiors,
        collision, playable preview) use generate_world."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        scene_name = storage.safe_filename(name or "scene")
        cats = tuple(c for c in (categories or ["obstacle"]) if c in worldgen.CATEGORIES) or ("obstacle",)
        detected = await worldgen.detect_scene(
            ctx.http, cfg.gemini_api_keys, src.data, src.mime or "image/png",
            hints=hints, max_objects=max_objects, model=detect_model)
        objects = detected["objects"]
        (width, height), bundle_files, sprite_files, stats = await _cut_sprites(
            src.data, objects, "main", cats, triage=False, qa=qa)
        entry = worldgen.build_map_entry("map_main.png", width, height, objects,
                                         detected["player_spawn"], None, sprite_files)
        manifest = worldgen.build_world_manifest(scene_name, {"main": entry})
        bundle_files.insert(0, ("map_main.png", src.data))
        bundle_files.append(("world.json", json.dumps(manifest, indent=2).encode()))
        sub = (subpath or "assets/forge").strip("/\\") + f"/scenes/{scene_name}"
        saved = await storage.save_bundle(bundle_files, project=project, subpath=sub, cfg=cfg)
        return {
            "name": scene_name,
            "objects_detected": len(objects),
            "sprites_cut": len(sprite_files),
            "sprite_stats": stats,
            "labels": [o["label"] for o in objects][:60],
            "world_json_url": saved["files"].get("world.json"),
            "bundle_url": saved["base_url"],
            **({"workspace_dir": saved["workspace_dir"]} if "workspace_dir" in saved else {}),
            **({"workspace_write_error": saved["workspace_write_error"]}
               if "workspace_write_error" in saved else {}),
        }
