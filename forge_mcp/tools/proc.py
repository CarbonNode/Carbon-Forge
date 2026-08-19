"""Image-processing tools — all delegate to the shared desktop pipeline."""
from backend.pixel_art import DITHER_MODES, RETRO_PALETTES
from backend.processing import PipelineOptions, parse_colors
from forge_mcp import engine, storage


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
        clean_holes: bool = False,
        subpath: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Remove the background from an image (rembg). image: https URL or workspace path
        '<Project>/<relative path>'. Optional pipeline steps: watermark removal first, then
        color removal, edge smoothing, and transparent-edge trim. Models: u2net (default),
        u2netp (fast), u2net_human_seg, isnet-general-use, silueta. Saves into the project's
        workspace folder (subpath default assets/forge) and returns the path + a shareable URL.

        CLEAN-CUTOUT RECIPE (use this, not bare defaults — bare defaults leave a grey/white
        'sticker halo', worst on pale subjects shot on a light background): pass
        model='isnet-general-use' (handles holes/negative space far better than u2net),
        alpha_matting=true, edge_smooth=true, auto_trim=true. If the source was generated on a
        flat solid background (the recommended way — see generate_image), also pass
        color_remove=true + color_auto_detect=true to strip the residual background fringe.
        Halo on an ALREADY-cut transparent image? Rescue it with smooth_edges(trim_px~6,
        strength~3) to erode + defringe the matte. Solid background still TRAPPED inside
        enclosed gaps (letter counters, keyholes, circles) afterwards? Pass clean_holes=true
        here to punch them out in the same pass, or run clean_edges on the cut-out."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        opts = PipelineOptions(
            model=model, alpha_matting=alpha_matting, fg_threshold=fg_threshold,
            bg_threshold=bg_threshold, erode_size=erode_size, color_remove=color_remove,
            colors=parse_colors(colors), color_auto_detect=color_auto_detect,
            color_tolerance=color_tolerance, edge_smooth=edge_smooth,
            edge_strength=edge_strength, edge_trim=edge_trim, auto_trim=auto_trim,
            watermark_remove=watermark_remove, watermark_position=watermark_position,
            watermark_size_pct=watermark_size_pct, clean_iso=clean_holes,
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
        model: str = "isnet-general-use",
        alpha_matting: bool = False,
        fg_threshold: int = 240,
        bg_threshold: int = 10,
        erode_size: int = 10,
        colors: list[str] | None = None,
        color_tolerance: int = 25,
        edge_strength: int = 60,
        edge_trim: int = 2,
        watermark_remove: bool = False,
        watermark_position: str = "bottom-right",
        watermark_size_pct: int = 15,
        subpath: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Split a sheet into individual trimmed, transparent PNG sprites (the "Carbon Isolate"
        split). By default it first isolates the artwork — AI background removal, then keys out
        the (auto-detected) background color, then defringes/erodes edges — so the gaps between
        pieces become fully transparent and connected-component labeling separates them cleanly;
        each sprite is auto-trimmed to its bounding box. Set skip_bg=true to split an image that
        is ALREADY transparent (ignores the cleanup options below).

        image: https URL or workspace path '<Project>/<relative path>'. Returns {count, sprites:[…]}.

        Options (mirror the desktop app):
        - model: rembg model — u2net, u2netp (fast), u2net_human_seg, isnet-general-use (default,
          best edges), silueta.
        - min_sprite_area: drop blobs smaller than this (px²; default 400) — filters specks.
        - alpha_matting (+ fg_threshold/bg_threshold/erode_size): refine soft/semi-transparent edges.
        - colors (hex like '#ffffff') + color_tolerance (1–100, effective floor 25): EXTRA background
          colors to key out on top of the auto-detected one.
        - edge_strength (1–100) / edge_trim (px): defringe strength and edge erosion — edge_trim also
          severs thin bridges so touching pieces don't merge into one sprite.
        - watermark_remove (+ position/size_pct): LaMa-inpaint a watermark before splitting."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        if skip_bg:
            sprites = await engine.split_only(src.data, min_sprite_area)
        else:
            opts = PipelineOptions(
                model=model, alpha_matting=alpha_matting, fg_threshold=fg_threshold,
                bg_threshold=bg_threshold, erode_size=erode_size,
                colors=parse_colors(colors), color_tolerance=color_tolerance,
                edge_smooth=True, edge_strength=edge_strength, edge_trim=edge_trim,
                watermark_remove=watermark_remove, watermark_position=watermark_position,
                watermark_size_pct=watermark_size_pct,
            )
            sprites = await engine.run_split_pipeline(src.data, opts, min_sprite_area)
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

    @mcp.tool()
    async def clean_edges(image: str, project: str, color: str | None = None,
                          tolerance: int = 32, min_pocket_px: int = 4,
                          max_pocket_frac: float = 0.5, feather: int = 0,
                          subpath: str | None = None, filename: str | None = None) -> dict:
        """Clean up a transparent cut-out by removing leftover background TRAPPED
        INSIDE the artwork — the solid fill stuck in the counters of letters (the
        holes in D / A / e / o), keyholes, the dot of a '?', rings and circles —
        that background removal leaves behind because those pockets aren't
        connected to the outer edge.

        A region is cleared only when it's the background colour (default white;
        pass color='#rrggbb' if the trapped fill is another flat colour) AND fully
        sealed — its blob never touches the transparent edge — so every outline and
        the silhouette are preserved. Use this when an already-isolated sticker
        still shows solid background in enclosed gaps (logos, text, emblems, icons).

        This is the companion to smooth_edges: smooth_edges feathers/defringes a
        soft semi-transparent HALO around the silhouette; clean_edges clears SOLID
        trapped fills on a hard binary matte. On portraits it will also clear white
        eye-whites (geometrically identical to a trapped pocket), so it's aimed at
        text / logo / shape stickers.

        image: https URL or workspace path '<Project>/<relative path>'.
        color: trapped background colour as '#rrggbb' (default white '#ffffff').
        tolerance: colour-match radius 0–255 (default 32).
        min_pocket_px: ignore sealed specks smaller than this (default 4).
        max_pocket_frac: never clear a pocket larger than this fraction of the
            sprite (default 0.5) — guards a legitimately large enclosed fill.
        feather: soften the freshly-cut borders by N px (default 0 = crisp)."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        rgb = None
        if color:
            parsed = parse_colors([color])
            rgb = parsed[0] if parsed else None
        out = await engine.run_pipeline(src.data, PipelineOptions(
            skip_bg=True, clean_iso=True, clean_color=rgb, clean_tolerance=tolerance,
            clean_min_px=min_pocket_px, clean_max_frac=max_pocket_frac, clean_feather=feather))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "cleaned", ext="png", cfg=cfg)

    @mcp.tool()
    async def pixel_refine(
        image: str,
        project: str,
        grid: str = "auto",
        cell_size: int = 0,
        max_cells: int = 512,
        sampling: str = "medoid",
        remove_bg: bool = False,
        bg_color: str | None = None,
        bg_tolerance: int = 24,
        max_colors: int = 0,
        palette: str | None = None,
        palette_colors: list[str] | None = None,
        dither: str = "none",
        dither_strength: float = 1.0,
        outline: str = "none",
        outline_color: str = "#000000",
        trim: bool = True,
        scale: int = 1,
        subpath: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Refine AI-generated pixel art into TRUE low-resolution pixel art
        (PixelRefiner pipeline). AI models emit "fake" pixel art: a big image
        whose logical pixels are soft ~NxN blocks with anti-aliased edges and
        thousands of near-duplicate colors. This tool detects the logical
        pixel grid, resamples each cell to one crisp color (Oklab medoid), and
        optionally reduces colors / maps to a retro palette — producing a real
        sprite (e.g. 1024x1024 in → 128x128 out) that game engines can use.
        Use it after generate_image/generate_local whenever the prompt asked
        for pixel art, sprites, or 8/16-bit style. No AI model involved: fast,
        deterministic, and it never repaints — every output color comes from
        the source.

        image: https URL or workspace path '<Project>/<relative path>'.
        grid: 'auto' (detect cell size + offset; the default) or 'off' (keep
          resolution 1:1 — use for already-true pixel art you only want to
          quantize/outline). cell_size > 0 forces a manual cell size instead.
        sampling: 'medoid' (default, crisp — picks the most representative
          real pixel from each cell's core), 'mean' (soft area average), or
          'hard' (medoid + binary alpha, for sprites needing hard edges).
        remove_bg (+ bg_color '#rrggbb', bg_tolerance): key out a flat
          background color first (auto-samples the border when bg_color is
          unset). For complex backgrounds run remove_background first instead.
        max_colors: k-means color reduction in Oklab space (0 = off).
        palette: map to a built-in retro palette — arne16, c64, gb_legacy,
          gb_light, gb_pocket, mono, msx, nes, pc98, pico8, sfc_bg,
          sfc_sprite. palette_colors: custom palette as '#rrggbb' hex strings
          (overrides palette).
        dither: none | floyd-steinberg | bayer-2x2 | bayer-4x4 | bayer-8x8 |
          ordered (+ dither_strength 0-1).
        outline: 'rounded' (8-way) or 'sharp' (4-way) 1px outline in
          outline_color (canvas grows 1px per side).
        trim: crop transparent margins (default true).
        scale: integer nearest-neighbor upscale of the final sprite (1-32) —
          e.g. scale=4 to make the crisp result visible in a chat preview.

        Returns the saved sprite plus an 'analysis' report: detected cell
        size/offset, grid confidence, logical output size, and color counts
        before/after. If the grid detection picked a wrong size (check
        analysis.grid), re-run with cell_size set explicitly."""
        for name, value, allowed in (
            ("grid", grid, ("auto", "off")),
            ("sampling", sampling, ("medoid", "mean", "hard")),
            ("dither", dither, DITHER_MODES),
            ("outline", outline, ("none", "rounded", "sharp")),
        ):
            if value not in allowed:
                return {"error": f"Invalid {name} '{value}' — "
                                 f"use one of: {', '.join(allowed)}"}
        if palette and palette not in RETRO_PALETTES:
            return {"error": f"Unknown palette '{palette}' — available: "
                             f"{', '.join(sorted(RETRO_PALETTES))}"}
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        out, report = await engine.pixel_refine(
            src.data, grid=grid, cell_size=cell_size, max_cells=max_cells,
            sampling=sampling, remove_bg=remove_bg, bg_color=bg_color,
            bg_tolerance=bg_tolerance, max_colors=max_colors,
            palette=palette_colors or palette, dither=dither,
            dither_strength=dither_strength, outline=outline,
            outline_color=outline_color, trim=trim, scale=scale)
        res = await storage.save_result(out, project=project, subpath=subpath,
                                        filename=filename or "pixel-refined",
                                        ext="png", cfg=cfg)
        res["analysis"] = report
        return res
