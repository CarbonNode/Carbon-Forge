"""Image-processing tools — all delegate to the shared desktop pipeline."""
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
        strength~3) to erode + defringe the matte."""
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
