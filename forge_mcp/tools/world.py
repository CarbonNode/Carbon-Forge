"""Game-world tools — one generated map image becomes a playable, style-consistent
world: labeled detections, cutout obstacle sprites, a collision grid, and a
self-contained playable preview.html (the "capybara.build" workflow)."""
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

    async def _cut_sprites(map_bytes, objects, categories=("obstacle",)):
        """Crop + alpha-cut each object in `categories`. Returns
        (bundle_files, sprite_files) — sprite_files: label -> {file, crop_px}."""
        from PIL import Image
        im = Image.open(BytesIO(map_bytes)).convert("RGB")
        width, height = im.size
        bundle_files, sprite_files = [], {}
        for obj in objects:
            if obj["category"] not in categories:
                continue
            left, top, right, bottom = worldgen.box_to_px(obj["box_2d"], width, height)
            if right - left < 4 or bottom - top < 4:
                continue
            buf = BytesIO()
            im.crop((left, top, right, bottom)).save(buf, "PNG")
            try:
                cut = await engine.run_pipeline(buf.getvalue(), PipelineOptions(**_CUT_OPTS))
            except Exception:
                cut = buf.getvalue()  # keep the raw crop rather than dropping the object
            name = f"sprite_{obj['label']}.png"
            bundle_files.append((name, cut))
            sprite_files[obj["label"]] = {"file": name, "crop_px": [top, left, bottom, right]}
        return (width, height), bundle_files, sprite_files

    async def _build_world(map_bytes, *, project, name, hints, max_objects, detect_model,
                           collision, preview, band_frac, grid_cols, subpath):
        detected = await worldgen.detect_scene(
            ctx.http, cfg.gemini_api_keys, map_bytes, "image/png",
            hints=hints, max_objects=max_objects, model=detect_model)
        objects, spawn = detected["objects"], detected["player_spawn"]

        (width, height), bundle_files, sprite_files = await _cut_sprites(map_bytes, objects)

        grid = None
        if collision:
            rows = max(8, round(grid_cols * height / max(1, width)))
            grid = worldgen.build_collision(objects, cols=grid_cols, rows=rows,
                                            band_frac=band_frac)
        manifest = worldgen.build_manifest(name, "map.png", width, height,
                                           objects, spawn, grid, sprite_files)
        bundle_files.insert(0, ("map.png", map_bytes))
        bundle_files.append(("world.json", json.dumps(manifest, indent=2).encode()))
        if preview:
            bundle_files.append(("preview.html", worldgen.render_preview_html(manifest).encode()))

        sub = (subpath or "assets/forge").strip("/\\") + f"/worlds/{storage.safe_filename(name)}"
        saved = await storage.save_bundle(bundle_files, project=project, subpath=sub, cfg=cfg)

        by_cat = {}
        for obj in objects:
            by_cat[obj["category"]] = by_cat.get(obj["category"], 0) + 1
        out = {
            "name": name,
            "map_url": saved["files"].get("map.png"),
            "world_json_url": saved["files"].get("world.json"),
            "objects": by_cat,
            "sprites_cut": len(sprite_files),
            "player_spawn": manifest["player_spawn"],
            "bundle_url": saved["base_url"],
        }
        if preview:
            out["preview_url"] = saved["files"].get("preview.html")
        if grid:
            out["collision"] = {"cols": grid["cols"], "rows": grid["rows"],
                                "blocked_cells": len(grid["blocked"])}
        if "workspace_dir" in saved:
            out["workspace_dir"] = saved["workspace_dir"]
        if "workspace_write_error" in saved:
            out["workspace_write_error"] = saved["workspace_write_error"]
        return out

    @mcp.tool()
    async def generate_world(prompt: str, project: str, name: str | None = None,
                             image: str | None = None, model: str = "imagen-4",
                             aspect_ratio: str = "16:9", style: str | None = None,
                             hints: str | None = None, max_objects: int = 48,
                             detect_model: str = worldgen.DETECT_MODEL,
                             collision: bool = True, preview: bool = True,
                             band_frac: float = 0.30, grid_cols: int = 64,
                             subpath: str | None = None) -> dict:
        """Generate a complete playable 2D game WORLD from one prompt — the whole scene is
        painted as ONE image (perfect internal style consistency, baked lighting), then AI
        object detection labels every prop, obstacles are cut out as transparent sprites
        (so characters can walk BEHIND them), a walkability/collision grid is derived from
        obstacle footprints, and everything is written as a bundle:
        map.png + sprite_*.png + world.json + a PLAYABLE preview.html (open the returned
        preview_url — WASD/drag to walk the world, G shows the collision grid).

        Vague prompts work ("a spooky forest village"); the default style locks a
        hand-painted 16-bit top-down RPG look (Stardew/Eastward lineage). Pass style="" for
        no style wrapper, or your own style string. Pass image (https URL or
        '<Project>/<path>') to skip generation and worldify existing art.

        world.json is self-describing: objects carry box_2d [ymin,xmin,ymax,xmax]
        normalized 0-1000 (y-first), sprites carry crop_px for exact in-place redraw,
        collision is a cols x rows grid of blocked cell indices. Categories: obstacle
        (blocks + occludes), zone_blocked (water/cliffs), enterable (doors), decor.
        hints steers detection ("the river is impassable; tag market stalls enterable").

        model: imagen-4 | imagen-4-fast | imagen-4-ultra. aspect_ratio: 16:9, 4:3, 1:1...
        band_frac: how much of an obstacle's base blocks walking (0.30 = bottom 30%).
        Cost: one Imagen image + one Gemini Flash detection (~a cent); cutouts are local."""
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
        return await _build_world(map_bytes, project=project, name=world_name, hints=hints,
                                  max_objects=max_objects, detect_model=detect_model,
                                  collision=collision, preview=preview, band_frac=band_frac,
                                  grid_cols=grid_cols, subpath=subpath)

    @mcp.tool()
    async def segment_scene(image: str, project: str, hints: str | None = None,
                            max_objects: int = 48,
                            detect_model: str = worldgen.DETECT_MODEL,
                            categories: list[str] | None = None,
                            subpath: str | None = None, name: str | None = None) -> dict:
        """Detect and cut out every object in a scene image as individually placed,
        transparent sprites — turn ANY illustration/map/screenshot into layered game
        pieces. AI detection labels each object with a tight box_2d ([ymin,xmin,ymax,xmax]
        normalized 0-1000, y-first) and a category (obstacle / zone_blocked / enterable /
        decor); objects in `categories` (default ['obstacle']) are alpha-cut from their
        crop so they redraw exactly in place (crop_px in the manifest). Writes a bundle:
        the source image + sprite_*.png + world.json manifest. hints steers detection.
        For the full playable pipeline (collision grid + preview.html, optional
        generation) use generate_world."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        scene_name = storage.safe_filename(name or "scene")
        cats = tuple(c for c in (categories or ["obstacle"]) if c in worldgen.CATEGORIES) or ("obstacle",)
        detected = await worldgen.detect_scene(
            ctx.http, cfg.gemini_api_keys, src.data, src.mime or "image/png",
            hints=hints, max_objects=max_objects, model=detect_model)
        objects = detected["objects"]
        (width, height), bundle_files, sprite_files = await _cut_sprites(src.data, objects, cats)
        manifest = worldgen.build_manifest(scene_name, "map.png", width, height,
                                           objects, detected["player_spawn"], None, sprite_files)
        bundle_files.insert(0, ("map.png", src.data))
        bundle_files.append(("world.json", json.dumps(manifest, indent=2).encode()))
        sub = (subpath or "assets/forge").strip("/\\") + f"/scenes/{scene_name}"
        saved = await storage.save_bundle(bundle_files, project=project, subpath=sub, cfg=cfg)
        return {
            "name": scene_name,
            "objects_detected": len(objects),
            "sprites_cut": len(sprite_files),
            "labels": [o["label"] for o in objects][:60],
            "world_json_url": saved["files"].get("world.json"),
            "bundle_url": saved["base_url"],
            **({"workspace_dir": saved["workspace_dir"]} if "workspace_dir" in saved else {}),
            **({"workspace_write_error": saved["workspace_write_error"]}
               if "workspace_write_error" in saved else {}),
        }
