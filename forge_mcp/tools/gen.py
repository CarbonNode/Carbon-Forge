"""Generation tools — Imagen, Gemini edit, Veo (async jobs)."""
import asyncio
from io import BytesIO

from backend.processing import PipelineOptions
from forge_mcp import engine
from forge_mcp import generation as g
from forge_mcp import storage
from forge_mcp import video

# Icon-optimized prompt: clean centered subject on a flat solid background (so rembg cuts cleanly).
ICON_TEMPLATE = (
    "A single {subject} icon, {style}, bold simple shapes, clean thick outlines, "
    "limited 2-3 color palette, centered with generous padding, front-on, no perspective, "
    "no text, no watermark, flat solid #FFFFFF background, app-icon style, crisp high-contrast edges"
)


def _square_pad(data: bytes, size: int = 512, pad_frac: float = 0.12,
                pixel: bool = False) -> bytes:
    """Center a (already cut-out) image on a transparent square canvas with padding,
    sized to `size`. pixel=True keeps a crisp pixel grid: integer nearest-neighbor
    scale only (nearest fractional downscale if the sprite doesn't fit)."""
    from PIL import Image
    im = Image.open(BytesIO(data)).convert("RGBA")
    inner = max(1, int(size * (1 - 2 * pad_frac)))
    w, h = im.size
    if pixel:
        scale = inner // max(w, h)
        if scale >= 1:
            im = im.resize((w * scale, h * scale), Image.NEAREST)
        else:
            frac = inner / max(w, h)
            im = im.resize((max(1, int(w * frac)), max(1, int(h * frac))), Image.NEAREST)
    else:
        scale = inner / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    buf = BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()

# Aspect ratio -> (width, height) at SDXL/Flux-friendly resolutions for local generation.
LOCAL_AR = {
    "1:1": (1024, 1024), "3:4": (896, 1152), "4:3": (1152, 896),
    "9:16": (768, 1344), "16:9": (1344, 768),
}

# Per-kind scaffolding for generate_image_grid: a shared cell style, a background, and
# whether cells get cut out transparent (+ square-padded) after splitting.
GRID_KINDS = {
    "spells": {
        "style": ("a single dramatic magical spell effect icon, luminous glowing energy, vivid "
                  "saturated colors, high contrast, centered, floating in empty space, no hands, "
                  "no characters, no framing ring"),
        "background": ("completely flat solid magenta #FF00FF filling the whole cell and every "
                       "space around and inside the effect"),
        "isolate": True, "pad": True,
    },
    "faces": {
        "style": ("a painted fantasy character portrait, head and shoulders, facing the viewer, "
                  "detailed expressive face, soft dramatic lighting"),
        "background": "plain dark neutral studio",
        "isolate": False, "pad": False,
    },
    "items": {
        "style": ("a single game item, centered, slight three-quarter view, crisp detailed "
                  "game-asset rendering, no border"),
        "background": "completely flat solid magenta #FF00FF filling the whole cell",
        "isolate": True, "pad": True,
    },
    "custom": {"style": "", "background": "plain dark neutral", "isolate": False, "pad": False},
}

GRID_IMAGEN_MODELS = ("imagen-4-fast", "imagen-4", "imagen-4-ultra")
GRID_GEMINI_NAMES = ("auto", "gemini", "nano-banana", "gemini-2.5-flash-image", "flash-image")


def _nearest_aspect(ratio: float) -> str:
    import math
    from forge_mcp import generation as g
    return min(g.IMAGE_ASPECTS,
               key=lambda a: abs(math.log((int(a.split(":")[0]) / int(a.split(":")[1])) / ratio)))


def _grid_layout(n: int, cols: int | None, aspect_ratio: str | None) -> tuple[int, int, str]:
    """(cols, rows, sheet_aspect). Prefers exact-fit, near-square grids, and — when no
    aspect is forced — a sheet aspect matching the grid, so cells stay ~square and the
    model is far more likely to draw the layout it was actually asked for."""
    import math
    if cols:
        c = max(1, min(n, cols))
        r = math.ceil(n / c)
        return c, r, aspect_ratio or _nearest_aspect(c / r)
    best = None
    for c in range(1, n + 1):
        r = math.ceil(n / c)
        ar = aspect_ratio or _nearest_aspect(c / r)
        arw, arh = (int(v) for v in ar.split(":"))
        cell_ar = (arw / arh) * r / c
        score = ((c * r - n) * 0.8 + abs(math.log(cell_ar))
                 + abs(math.log(c / r)) * 0.5)
        if best is None or score < best[0]:
            best = (score, c, r, ar)
    return best[1], best[2], best[3]

_IPA_PRESETS = {"character": "PLUS (high strength)", "face": "PLUS FACE (portraits)",
                "style": "PLUS (high strength)"}


def _ipa_preset(mode):
    """(preset, weight_type) for an IPAdapter mode."""
    return _IPA_PRESETS.get(mode or "character", "PLUS (high strength)"), \
        ("style transfer" if mode == "style" else "linear")


def register(mcp, ctx):
    cfg, jobs, chars = ctx.cfg, ctx.jobs, ctx.characters

    def _require_key():
        if not cfg.gemini_api_key:
            raise g.GenerationError("GEMINI_API_KEY is not configured on the forge service")

    @mcp.tool()
    async def generate_image(prompt: str, project: str, model: str = "imagen-4",
                             count: int = 1, aspect_ratio: str = "1:1",
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate image(s) from a text prompt with Google's Gemini image models
        (the imagen-* aliases are retained; Imagen itself was retired upstream).
        model: imagen-4 | imagen-4-fast | imagen-4-ultra. count: 1-4 (ultra max 1).
        aspect_ratio: 1:1, 3:4, 4:3, 9:16, 16:9. Saves into the project's workspace folder.

        CUT-OUT ASSETS (REQUIRED when the image will later be background-removed — sprites,
        character/object cutouts, stickers, anything destined for remove_background): a vague
        prompt produces a halo'd, hard-to-key result. You MUST prompt for ALL of:
          - a single COMPLETELY FLAT, uniform, solid background color (no gradient, texture,
            scenery, or ground shadow);
          - a background color FAR from every color in the subject — never green for a
            green-skinned subject, never white for a pale/light subject. Safe default: solid
            magenta #FF00FF (far from skin, steel, brown, foliage). State the hex explicitly;
          - that same flat color must also FILL ALL NEGATIVE SPACE *between* limbs, props,
            and weapons (e.g. between two hands gripping a handle, inside a drawn bow) so no
            interior background gets trapped during keying;
          - a crisp clean silhouette, no outline/border, no vignette, no drop shadow.
        Then cut with remove_background(model='isnet-general-use', alpha_matting=true,
        color_remove=true, color_auto_detect=true, edge_smooth=true, auto_trim=true)."""
        _require_key()
        if aspect_ratio not in g.IMAGE_ASPECTS:
            raise g.GenerationError(f"aspect_ratio must be one of {g.IMAGE_ASPECTS}")
        model_id = g.resolve_image_model(model)
        n = max(1, min(count, g.imagen_max_batch(model_id)))
        images = await g.call_imagen(ctx.http, cfg.gemini_api_keys, model_id, prompt,
                                     sample_count=n, aspect_ratio=aspect_ratio)
        if not images:
            raise g.GenerationError("Imagen returned no images (prompt may have been refused)")
        base = storage.safe_filename(filename or prompt[:48])
        results = []
        for i, img in enumerate(images, 1):
            name = base if len(images) == 1 else f"{base}-{i}"
            results.append(await storage.save_result(img, project=project, subpath=subpath,
                                                     filename=name, ext="png", cfg=cfg))
        return {"count": len(results), "images": results}

    @mcp.tool()
    async def generate_image_grid(subjects: list[str], project: str, kind: str = "custom",
                                  model: str = "auto", style: str = "",
                                  reference_images: list[str] | None = None,
                                  aspect_ratio: str | None = None, cols: int | None = None,
                                  background: str | None = None, isolate: bool | None = None,
                                  pad_to_square: bool | None = None,
                                  snap_to_gutters: bool = True, inset: int | None = None,
                                  cell_size: int = 512,
                                  pixel_refine: bool = False,
                                  refine_max_colors: int = 0,
                                  refine_palette: str | None = None,
                                  subpath: str | None = None,
                                  filename: str | None = None) -> dict:
        """Generate a BATCH of distinct images in ONE model call: renders a sheet laid out
        as a grid, then cuts it into separate image files — the cheap, fast way to make
        sets of spell icons, NPC faces, items, monsters (~1 request instead of N).

        EASY CALLS — kind presets do the styling AND the post-processing:
          spell icons, cut transparent + squared:  {subjects:[...], project, kind:"spells"}
          NPC face portraits:                      {subjects:[...], project, kind:"faces"}
          item sprites, cut transparent:           {subjects:[...], project, kind:"items"}
          same face/style in every cell:           add reference_images:[url or '<Proj>/<path>']
        Refine with `style` (appended to the preset), override `background`/`isolate`/
        `pad_to_square` individually, or use kind:"custom" for full manual control.

        subjects: 2-16 short per-cell descriptions, each becoming its own output image, in
        order. Make them clearly distinct or cells look alike. Best detail at 2-9.

        HARDENED PATHWAY: the sheet is cut along the grid the model ACTUALLY drew, not
        blind equal division — drawn gutter lines are detected (outer margins included)
        and each cell is shaved of frame residue. If the model drew a DIFFERENT grid than
        asked (e.g. a 3x3 for 6 subjects, padding with duplicates), ALL drawn cells come
        back cleanly cut, result.grid.mismatch=true, and a warning explains that subject
        labels beyond the drawn order are best-effort — eyeball grid_url. isolate=true
        cuts every cell to a transparent trimmed sprite (rembg clean-cutout recipe);
        pad_to_square centers it on a transparent cell_size square. A cell whose cutout
        comes back empty keeps its uncut version (warned).

        model: "auto" (default) = Gemini 2.5 Flash Image — best at following grid layouts,
        required for reference_images — falling back to imagen-4-fast if Gemini refuses.
        Pass imagen-4-fast | imagen-4 | imagen-4-ultra to force Imagen, or "gemini" to
        forbid the fallback. Multi-key failover + one refusal retry are automatic.
        aspect_ratio/cols: normally leave unset — the layout picker chooses an exact-fit,
        near-square grid AND a sheet aspect matching it (cells stay square, the model
        complies far more often). Returns one image per subject IN ORDER (each with
        .subject, .isolated), plus grid_url (raw sheet), engine, grid{cols,rows,mode,
        mismatch,snapped_lines}, warnings, cost_note.

        pixel_refine=true (for PIXEL-ART sheets — sprites/icons prompted in 8/16-bit
        style): each cut cell is run through the pixel-art refiner instead of the
        LANCZOS enlarge — its logical pixel grid is detected and resampled to true
        low-res pixels, then integer-upscaled toward cell_size. refine_max_colors
        (k-means) / refine_palette (retro palette name, e.g. 'pico8') tune it; each
        image carries a 'refine' report (detected cell size, confidence, colors)."""
        _require_key()
        import io
        from PIL import Image
        from forge_mcp import gridcut
        subs = [s.strip() for s in (subjects or []) if s and s.strip()]
        if not (2 <= len(subs) <= 16):
            raise g.GenerationError("subjects must contain 2-16 non-empty items")
        if aspect_ratio is not None and aspect_ratio not in g.IMAGE_ASPECTS:
            raise g.GenerationError(f"aspect_ratio must be one of {g.IMAGE_ASPECTS}")
        preset = GRID_KINDS.get(kind)
        if preset is None:
            raise g.GenerationError(f"kind must be one of: {', '.join(GRID_KINDS)}")
        if model not in GRID_GEMINI_NAMES and model not in GRID_IMAGEN_MODELS:
            raise g.GenerationError(
                f"model must be one of: {', '.join(('auto', 'gemini') + GRID_IMAGEN_MODELS)}")
        if refine_palette:
            from backend.pixel_art import RETRO_PALETTES
            if refine_palette not in RETRO_PALETTES:
                raise g.GenerationError(
                    f"refine_palette must be one of: {', '.join(sorted(RETRO_PALETTES))}")
        do_isolate = preset["isolate"] if isolate is None else isolate
        do_pad = (preset["pad"] if pad_to_square is None else pad_to_square) and do_isolate
        bg = (background or preset["background"]).strip()
        sty = ", ".join(p for p in (preset["style"], style.strip().rstrip(",")) if p)

        n = len(subs)
        c, r, ar = _grid_layout(n, cols, aspect_ratio)
        cells_txt = "; ".join(f"cell {i + 1} = {s}" for i, s in enumerate(subs))
        filler = c * r - n
        filler_txt = (f" Cells {n + 1}-{c * r}: completely empty, just the plain background."
                      if filler else "")
        sty_txt = f"{sty}, " if sty else ""
        prompt = (f"A clean {c}-column by {r}-row grid of {c * r} equal-size cells with thin "
                  f"plain gutter lines exactly on the grid between cells. Each cell: "
                  f"{sty_txt}on a {bg} background. No text, no labels, no numbers, no "
                  f"captions, no outer border. Each cell a DIFFERENT subject, in reading "
                  f"order left-to-right then top-to-bottom: {cells_txt}.{filler_txt} Every "
                  f"cell visibly distinct — no two alike.")

        refs = []
        for ref in (reference_images or [])[:6]:
            resolved = await storage.resolve_input(ref, cfg=cfg, kind="image")
            refs.append((resolved.mime, resolved.data))

        warnings = []
        sheet = eng = None
        if refs or model in GRID_GEMINI_NAMES:
            try:
                try:
                    imgs = await g.call_gemini_image(ctx.http, cfg.gemini_api_keys, prompt,
                                                     refs, aspect_ratio=ar)
                except g.GenerationError:
                    warnings.append("first Gemini attempt failed; retried")
                    imgs = await g.call_gemini_image(ctx.http, cfg.gemini_api_keys,
                                                     "Stylized digital artwork. " + prompt,
                                                     refs, aspect_ratio=ar)
                sheet, eng = imgs[0], g.DEFAULT_GEMINI_IMAGE_MODEL
            except g.GenerationError as e:
                if refs or model != "auto":
                    raise  # explicitly wanted Gemini (or refs require it) — don't switch looks
                warnings.append(f"Gemini sheet failed ({e}); fell back to imagen-4-fast")
        if sheet is None:
            alias = model if model in GRID_IMAGEN_MODELS else "imagen-4-fast"
            model_id = g.resolve_image_model(alias)
            imgs = await g.call_imagen(ctx.http, cfg.gemini_api_keys, model_id, prompt,
                                       sample_count=1, aspect_ratio=ar)
            if not imgs:
                warnings.append("first Imagen attempt returned nothing; retried")
                imgs = await g.call_imagen(ctx.http, cfg.gemini_api_keys, model_id,
                                           "Stylized digital artwork. " + prompt,
                                           sample_count=1, aspect_ratio=ar)
            if not imgs:
                raise g.GenerationError("Imagen returned no images (grid prompt may have been refused)")
            sheet, eng = imgs[0], alias

        cells, grid_meta = await asyncio.to_thread(
            gridcut.snap_cut, sheet, c, r, n, inset, snap_to_gutters)
        if not grid_meta["mismatch"] and len(cells) > n:
            cells = cells[:n]  # drop the filler cells we asked for
        if grid_meta["mismatch"]:
            d_c, d_r = grid_meta["cols"], grid_meta["rows"]
            warnings.append(
                f"model drew a {d_c}x{d_r} grid instead of the requested {c}x{r} — returned all "
                f"{len(cells)} drawn cells; subject labels are best-effort, check grid_url")
            if len(cells) < n:
                warnings.append(f"only {len(cells)} cells were drawn for {n} subjects")

        iso_opts = PipelineOptions(model="isnet-general-use", alpha_matting=True, edge_smooth=True,
                                   auto_trim=True, color_remove=True, color_auto_detect=True)
        base = storage.safe_filename(filename or (kind if kind != "custom" else (style[:24] or "grid")))
        results = []
        for idx, cell in enumerate(cells):
            # LANCZOS enlarge would smear the logical pixel grid — when refining,
            # the pixel refiner integer-upscales toward cell_size instead
            if cell_size and cell.width < cell_size and not pixel_refine:
                scale = cell_size / cell.width
                cell = cell.resize((cell_size, max(1, int(cell.height * scale))), Image.LANCZOS)
            buf = io.BytesIO(); cell.save(buf, "PNG")
            out_bytes = buf.getvalue()
            isolated = False
            refine_report = None
            if do_isolate:
                cut = await engine.run_pipeline(out_bytes, iso_opts)
                cut_im = Image.open(io.BytesIO(cut))
                alpha_box = (cut_im.getchannel("A").getbbox()
                             if cut_im.mode == "RGBA" else cut_im.getbbox())
                if alpha_box:
                    if pixel_refine:
                        cut, refine_report = await engine.pixel_refine(
                            cut, max_colors=refine_max_colors, palette=refine_palette,
                            target_px=0 if do_pad else (cell_size or 0))
                    out_bytes = (_square_pad(cut, size=cell_size, pixel=pixel_refine)
                                 if do_pad else cut)
                    isolated = True
                else:
                    warnings.append(f"cell {idx + 1}: cutout came back empty; kept the uncut cell")
            elif pixel_refine:
                out_bytes, refine_report = await engine.pixel_refine(
                    out_bytes, max_colors=refine_max_colors, palette=refine_palette,
                    target_px=cell_size or 0)
            res = await storage.save_result(out_bytes, project=project, subpath=subpath,
                                            filename=f"{base}-{idx + 1}", ext="png", cfg=cfg)
            if idx < n:
                res["subject"] = subs[idx]
            res["isolated"] = isolated
            if refine_report is not None:
                res["refine"] = refine_report
            results.append(res)
        grid_res = await storage.save_result(sheet, project=project, subpath=subpath,
                                             filename=f"{base}-GRID", ext="png", cfg=cfg)
        out = {"count": len(results), "images": results, "grid_url": grid_res["url"],
               "engine": eng, "kind": kind,
               "grid": {k: grid_meta[k] for k in ("cols", "rows", "mode", "mismatch",
                                                  "snapped_lines", "expected_lines", "inset")},
               "cost_note": f"{len(results)} images from 1 {eng} request ({c}x{r} grid asked)"}
        if warnings:
            out["warnings"] = warnings
        return out

    @mcp.tool()
    async def generate_local(prompt: str, project: str, model: str = "pony",
                             aspect_ratio: str = "3:4", steps: int | None = None,
                             cfg_scale: float | None = None, negative_prompt: str | None = None,
                             seed: int | None = None, lora: str | None = None,
                             lora_strength: float = 1.0, face_detail: bool = False,
                             fallback: bool = True,
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate an image LOCALLY on the GPU box via ComfyUI — FULLY UNCENSORED (dark/gore/adult
        OK; no cloud content filter). Saves into the project's workspace folder.
          model: 'pony' (fast, anime-lean, max uncensored) | 'cyberrealistic' / 'bigasp' (PHOTOREAL NSFW) |
                 'illustrious' (anime) | 'juggernaut' (general photoreal) | 'flux' (best coherence) | or any
                 installed checkpoint filename (see list_models).
          aspect_ratio: 1:1, 3:4, 4:3, 9:16, 16:9.
        The model's quality-tag dialect + a sane negative prompt are auto-applied (override via negative_prompt).
        If the GPU box is unreachable/busy and fallback=True, falls back to cloud Imagen 4 (filtered).
        lora: optional SDXL LoRA — a curated alias OR an installed .safetensors filename (see
        list_models -> local_loras); lora_strength ~0.6-1.0. SDXL models only; ignored on cloud fallback.
        face_detail: run an ADetailer face-restore pass (detect + high-res inpaint each face) — a big
        quality lift on full-body / group shots where faces come out small; SDXL only, adds ~10-20s."""
        w, h = LOCAL_AR.get(aspect_ratio, (896, 1152))
        loras = [(g.resolve_lora(lora), lora_strength)] if lora else None
        engine = f"comfy:{model}"
        images, reason = None, None
        # Routing policy: laybackrig's ComfyUI first; overflow to maingamingrig when laybackrig
        # is busy (a gen already running) — and never a box being gamed on (presence-gated).
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        if any(b["url"] for b in backends):
            # Route only to a box that actually has this checkpoint (auto-discovered); skip the gate
            # for the always-present defaults to save a probe.
            req_ckpt = None
            if model not in ("pony", "flux"):
                try:
                    req_ckpt = [g.resolve_model(model)[0]]
                except g.GenerationError:
                    req_ckpt = None
            chosen_url, sel = await g.select_comfy(ctx.http, backends, require_checkpoints=req_ckpt)
            if chosen_url:
                engine = f"comfy:{model}{('+lora:' + lora) if lora else ''}{'+facedetail' if face_detail else ''}@{sel}"
                try:
                    images = await g.call_comfy(ctx.http, chosen_url, prompt, model=model,
                                                negative_prompt=negative_prompt, width=w, height=h,
                                                steps=steps, cfg=cfg_scale, seed=seed, loras=loras,
                                                face_detail=face_detail)
                except g.GenerationError as e:
                    reason = str(e)
            else:
                reason = sel
        else:
            reason = "no ComfyUI backend configured"
        if images is None:
            if not fallback:
                raise g.GenerationError(f"Local generation failed: {reason}")
            _require_key()
            engine = "imagen-4 (fallback)"
            ar = aspect_ratio if aspect_ratio in g.IMAGE_ASPECTS else "1:1"
            images = await g.call_imagen(ctx.http, cfg.gemini_api_keys,
                                         g.resolve_image_model("imagen-4"), prompt,
                                         sample_count=1, aspect_ratio=ar)
            if not images:
                raise g.GenerationError("Local generation failed and Imagen fallback returned nothing")
        base = storage.safe_filename(filename or prompt[:48])
        results = []
        for i, img in enumerate(images, 1):
            name = base if len(images) == 1 else f"{base}-{i}"
            results.append(await storage.save_result(img, project=project, subpath=subpath,
                                                     filename=name, ext="png", cfg=cfg))
        out = {"count": len(results), "images": results, "engine": engine}
        if engine.endswith("(fallback)"):
            out["fallback_reason"] = reason
        return out

    @mcp.tool()
    async def generate_image_batch(prompt: str, project: str, count: int = 4, model: str = "pony",
                                   aspect_ratio: str = "3:4", negative_prompt: str | None = None,
                                   subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate COUNT variations of a prompt IN PARALLEL across the local GPU pool (laybackrig +
        maingamingrig), then return them all — a fast 'contact sheet' that uses every free box at
        once. Variations are spread round-robin across eligible boxes (skipping any being gamed on /
        running chim); each gets a fresh seed. model: pony | flux. aspect_ratio: 1:1,3:4,4:3,9:16,16:9."""
        count = max(1, min(count, 12))
        w, h = LOCAL_AR.get(aspect_ratio, (896, 1152))
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        elig = await g.eligible_comfy_backends(ctx.http, backends)
        if not elig:
            raise g.GenerationError("No local GPU backend available for batch (all busy/gaming/chim)")
        base = storage.safe_filename(filename or prompt[:40])

        async def _one(i, backend):
            imgs = await g.call_comfy(ctx.http, backend["url"], prompt, model=model,
                                      negative_prompt=negative_prompt, width=w, height=h,
                                      free_after=False)  # free once at the end, not between variations
            res = await storage.save_result(imgs[0], project=project, subpath=subpath,
                                            filename=f"{base}-{i + 1}", ext="png", cfg=cfg)
            res["engine"] = f"comfy:{model}@{backend['label']}"
            return res

        settled = await asyncio.gather(*[_one(i, elig[i % len(elig)]) for i in range(count)],
                                       return_exceptions=True)
        for b in elig:
            await g.comfy_free(ctx.http, b["url"])  # hand the cards back to chim/games
        images = [r for r in settled if not isinstance(r, Exception)]
        errors = [str(r) for r in settled if isinstance(r, Exception)]
        return {"count": len(images), "requested": count, "images": images,
                "spread_across": [b["label"] for b in elig], "errors": errors or None}

    @mcp.tool()
    async def edit_image(prompt: str, reference_images: list[str], project: str,
                         subpath: str | None = None, filename: str | None = None) -> dict:
        """Edit/compose images with Gemini: give 1-6 reference images (https URLs or
        '<Project>/<path>' workspace paths) plus an instruction prompt. Returns the new image(s)."""
        _require_key()
        if not reference_images or len(reference_images) > 6:
            raise g.GenerationError("Provide 1-6 reference_images")
        refs = []
        for r in reference_images:
            resolved = await storage.resolve_input(r, cfg=cfg, kind="image")
            refs.append((resolved.mime, resolved.data))
        images = await g.call_gemini_image(ctx.http, cfg.gemini_api_keys, prompt, refs)
        base = storage.safe_filename(filename or "edited")
        results = []
        for i, img in enumerate(images, 1):
            name = base if len(images) == 1 else f"{base}-{i}"
            results.append(await storage.save_result(img, project=project, subpath=subpath,
                                                     filename=name, ext="png", cfg=cfg))
        return {"count": len(results), "images": results}

    @mcp.tool()
    async def upscale_image(image: str, project: str, subpath: str | None = None,
                            filename: str | None = None) -> dict:
        """Upscale an image ~4x on a local GPU box (ESRGAN) — uncensored, free. image: an https URL or
        a '<Project>/<path>' workspace path. Routes across the pool (skips boxes being gamed-on / running
        chim) and frees the card after. Great for finishing a fast small gen at high resolution."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        chosen, sel = await g.select_comfy(ctx.http, backends)
        if not chosen:
            raise g.GenerationError(f"No GPU backend available to upscale ({sel})")
        up = await g.call_comfy_upscale(ctx.http, chosen, src.data)
        res = await storage.save_result(up, project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or "upscaled"), ext="png", cfg=cfg)
        res["engine"] = f"esrgan-4x@{sel}"
        return {"image": res, "engine": res["engine"]}

    @mcp.tool()
    async def edit_local(image: str, instruction: str, project: str, steps: int | None = None,
                         seed: int | None = None, guidance: float | None = None,
                         subpath: str | None = None, filename: str | None = None) -> dict:
        """Edit an image by a text INSTRUCTION, LOCALLY + uncensored, via Flux Kontext — e.g. "make the
        jacket red", "remove the person on the left", "change the background to a snowy mountain",
        "turn it into night". image: an https URL or a '<Project>/<path>' workspace path. Routes across
        the GPU pool (only boxes that have the Kontext model; skips boxes being gamed-on / running chim).
        guidance ~2.5 (higher = follow the instruction more literally). This is the local/uncensored
        counterpart to edit_image (which uses cloud Gemini)."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        chosen, sel = await g.select_comfy(ctx.http, backends, require_unets=[g.FLUX_KONTEXT["unet"]])
        if not chosen:
            raise g.GenerationError(f"No GPU backend with Flux Kontext available ({sel})")
        out = await g.call_comfy_kontext(ctx.http, chosen, src.data, instruction,
                                         steps=steps, seed=seed, guidance=guidance)
        res = await storage.save_result(out, project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or "edited"), ext="png", cfg=cfg)
        res["engine"] = f"kontext@{sel}"
        return {"image": res, "engine": res["engine"]}

    async def _resolve_reference(character, reference_image, mode, weight):
        """Returns (ref_images, mode, weight) from EITHER a saved character name (all its stored
        references, averaged) OR a one-off reference_image (URL/workspace path). Saved-character
        defaults fill in mode/weight when unset."""
        if character:
            ref_list, entry = chars.read_references(character)
            return ref_list, (mode or entry.get("mode") or "character"), \
                (weight if weight is not None else entry.get("weight", 0.8))
        if reference_image:
            src = await storage.resolve_input(reference_image, cfg=cfg, kind="image")
            return [src.data], (mode or "character"), (weight if weight is not None else 0.8)
        raise g.GenerationError("Provide either `character` (a saved name) or `reference_image`")

    @mcp.tool()
    async def generate_with_reference(prompt: str, project: str, reference_image: str | None = None,
                                      character: str | None = None, mode: str | None = None,
                                      weight: float | None = None, aspect_ratio: str = "3:4",
                                      negative_prompt: str | None = None, steps: int | None = None,
                                      seed: int | None = None, lora: str | None = None,
                                      lora_strength: float = 1.0, face_detail: bool = False,
                                      subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate an image that KEEPS the character / face / style of a reference (IPAdapter) — LOCAL
        on the GPU pool, uncensored. Use it for the SAME character across scenes/poses, a consistent
        mascot, or transferring an art style. SDXL/pony only.
          character: a SAVED character name (see save_character / list_characters) — its stored
            reference + default mode/weight are used. OR
          reference_image: a one-off https URL / '<Project>/<path>' workspace path.
          mode: 'character' (subject + style) | 'face' (portrait, face-locked) | 'style' (style only).
          weight: 0.4-1.2 — how strongly to follow the reference (lower = more prompt freedom).
          lora: optional SDXL LoRA (curated alias or installed filename; see list_models) applied on top.
          face_detail: ADetailer face-restore pass (sharper face, keeps the character) — recommended for portraits.
          aspect_ratio: 1:1, 3:4, 4:3, 9:16, 16:9."""
        ref_images, mode, weight = await _resolve_reference(character, reference_image, mode, weight)
        w, h = LOCAL_AR.get(aspect_ratio, (896, 1152))
        loras = [(g.resolve_lora(lora), lora_strength)] if lora else None
        preset, weight_type = _ipa_preset(mode)
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        chosen, sel = await g.select_comfy(ctx.http, backends)
        if not chosen:
            raise g.GenerationError(f"No GPU backend available for reference gen ({sel})")
        imgs = await g.call_comfy_ref(ctx.http, chosen, ref_images, prompt, model="pony", preset=preset,
                                      weight=weight, weight_type=weight_type, negative_prompt=negative_prompt,
                                      width=w, height=h, steps=steps, seed=seed, loras=loras,
                                      face_detail=face_detail)
        res = await storage.save_result(imgs[0], project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or prompt[:40]), ext="png", cfg=cfg)
        res["engine"] = f"ipadapter-{mode}{('+lora:' + lora) if lora else ''}{'+facedetail' if face_detail else ''}@{sel}"
        return {"image": res, "engine": res["engine"], "mode": mode, "character": character}

    @mcp.tool()
    async def generate_with_face(prompt: str, project: str, reference_image: str | None = None,
                                 character: str | None = None, model: str = "cyberrealistic",
                                 weight: float = 0.8, aspect_ratio: str = "3:4",
                                 negative_prompt: str | None = None, steps: int | None = None,
                                 seed: int | None = None, face_detail: bool = False,
                                 subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate an image with the EXACT FACE of a reference person (InstantID) — a far stronger identity
        lock than generate_with_reference's IPAdapter 'face' mode. Use for a recurring character who must look
        like the SAME person across every scene, pose, and outfit. LOCAL + uncensored; SDXL only (photoreal
        models like cyberrealistic / bigasp shine). Give a clear, mostly front-on face photo.
          character: a SAVED character name (its stored reference is used as the face) OR
          reference_image: a one-off https URL / '<Project>/<path>' workspace path.
          model: an SDXL alias/checkpoint (default cyberrealistic). weight: 0.6-1.0 identity strength
            (higher = closer to the reference face, less prompt freedom).
          face_detail: add an ADetailer pass for extra face sharpness (keeps the InstantID identity).
          aspect_ratio: 1:1, 3:4, 4:3, 9:16, 16:9."""
        if character:
            ref_list, _entry = chars.read_references(character)  # raises if unknown
            face_bytes = ref_list[0]
        elif reference_image:
            src = await storage.resolve_input(reference_image, cfg=cfg, kind="image")
            face_bytes = src.data
        else:
            raise g.GenerationError("Provide either `character` (a saved name) or `reference_image`")
        w, h = LOCAL_AR.get(aspect_ratio, (896, 1152))
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        chosen, sel = await g.select_comfy(ctx.http, backends)
        if not chosen:
            raise g.GenerationError(f"No GPU backend available for InstantID ({sel})")
        imgs = await g.call_comfy_instantid(ctx.http, chosen, face_bytes, prompt, model=model,
                                            negative_prompt=negative_prompt, ip_weight=weight,
                                            width=w, height=h, steps=steps, seed=seed, face_detail=face_detail)
        res = await storage.save_result(imgs[0], project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or prompt[:40]), ext="png", cfg=cfg)
        res["engine"] = f"instantid:{model}{'+facedetail' if face_detail else ''}@{sel}"
        return {"image": res, "engine": res["engine"], "character": character}

    @mcp.tool()
    async def save_character(name: str, reference_image: str, description: str | None = None,
                             mode: str = "character", weight: float = 0.8) -> dict:
        """Save a named CHARACTER from a reference image, so you can reuse it by name (consistent
        character/face/style) in generate_with_reference and generate_clip — no need to re-pass the
        image each time. reference_image: https URL or '<Project>/<path>'. Persists across restarts.
          mode: default IPAdapter mode for this character — character | face | style.
          weight: default reference strength (0.4-1.2). Re-saving the same name replaces it."""
        src = await storage.resolve_input(reference_image, cfg=cfg, kind="image")
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(src.mime, "png")
        dims = list(storage._image_dims(src.data))
        saved = chars.save(name=name, data=src.data, ext=ext, description=description, mode=mode,
                           weight=weight, source=reference_image, dims=dims if dims[0] else None)
        return {"saved": saved}

    @mcp.tool()
    async def add_character_reference(name: str, reference_image: str) -> dict:
        """Add ANOTHER reference image to an existing saved character (e.g. a second/third angle or
        expression). More references = a stronger, more robust likeness — the IPAdapter path averages
        their embeds. reference_image: https URL or '<Project>/<path>'."""
        src = await storage.resolve_input(reference_image, cfg=cfg, kind="image")
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(src.mime, "png")
        return {"character": chars.add_reference(name=name, data=src.data, ext=ext)}

    @mcp.tool()
    async def list_characters() -> dict:
        """List all saved characters (name, description, default mode/weight, ref size) — the named
        references usable via the `character` param of generate_with_reference / generate_clip."""
        cs = chars.list()
        return {"count": len(cs), "characters": cs}

    @mcp.tool()
    async def delete_character(name: str) -> dict:
        """Delete a saved character and its stored reference image."""
        return chars.delete(name)

    @mcp.tool()
    async def update_character(name: str, new_name: str | None = None, description: str | None = None,
                               mode: str | None = None, weight: float | None = None) -> dict:
        """Edit a saved character's metadata IN PLACE - rename it, or change its description,
        default mode (character|face|style) or default weight (0.4-1.2). Reference images are left
        untouched (use add_character_reference to add more, or save_character to replace the primary).
        Only the provided fields change."""
        return {"character": chars.update(name=name, new_name=new_name, description=description,
                                          mode=mode, weight=weight)}

    @mcp.tool()
    async def generate_icon(subject: str, project: str, style: str = "flat vector",
                            model: str = "flux", size: int = 512, transparent: bool = True,
                            seed: int | None = None, subpath: str | None = None,
                            filename: str | None = None) -> dict:
        """Generate a clean app/UI ICON in one shot: builds an icon-optimized prompt, renders the
        subject centered on a solid white background, cuts it out to TRANSPARENT (rembg clean-cutout
        recipe), and squares + pads to `size`. Great for connector/app icons, logos, game items.
          subject: what the icon depicts (e.g. "a purple robot head", "a shopping cart").
          style: 'flat vector' | '3D clay' | 'line art' | 'pixel art' | 'glassmorphism' | 'sticker' …
          model: 'flux' / 'pony' (LOCAL, free, uncensored) | 'imagen' (cloud).
          transparent: cut the background out to transparent (default true); false keeps the card.
          size: output square px (default 512)."""
        prompt = ICON_TEMPLATE.format(subject=subject, style=style)
        raster = None
        engine_str = None
        if model != "imagen":  # any local checkpoint (alias or installed filename); else cloud
            backends = [
                {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
                {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
            ]
            req_ckpt = None
            if model not in ("pony", "flux"):
                try:
                    req_ckpt = [g.resolve_model(model)[0]]
                except g.GenerationError:
                    req_ckpt = None
            chosen, sel = await g.select_comfy(ctx.http, backends, require_checkpoints=req_ckpt)
            if chosen:
                try:
                    raster = (await g.call_comfy(ctx.http, chosen, prompt, model=model,
                                                 width=1024, height=1024, seed=seed))[0]
                    engine_str = f"comfy:{model}@{sel}"
                except g.GenerationError:
                    raster = None
        if raster is None:  # cloud fallback (or model='imagen')
            _require_key()
            raster = (await g.call_imagen(ctx.http, cfg.gemini_api_keys,
                                          g.resolve_image_model("imagen-4"), prompt,
                                          sample_count=1, aspect_ratio="1:1"))[0]
            engine_str = "imagen-4"
        out = raster
        is_pixel_style = "pixel" in style.lower()
        refine_report = None
        if transparent:
            opts = PipelineOptions(model="isnet-general-use", alpha_matting=True, edge_smooth=True,
                                   auto_trim=True, color_remove=True, color_auto_detect=True)
            out = await engine.run_pipeline(raster, opts)
            # pixel-art styled icons get grid-refined to true low-res pixels; the
            # reconstruction gate keeps 1:1 when the render has no real grid
            if is_pixel_style:
                out, refine_report = await engine.pixel_refine(out)
            grid_found = bool(refine_report and refine_report["grid"]["detected"])
            out = _square_pad(out, size=size, pixel=grid_found)
        elif is_pixel_style:
            out, refine_report = await engine.pixel_refine(out, target_px=size)
        res = await storage.save_result(out, project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or subject[:40]), ext="png", cfg=cfg)
        res["engine"] = f"icon/{engine_str}"
        if refine_report is not None:
            res["refine"] = refine_report
        return {"image": res, "engine": res["engine"], "prompt": prompt}

    async def _poll_and_finish(job_id: str, op: str):
        job = jobs.get(job_id)
        sample = await g.poll_veo(ctx.http, cfg.gemini_api_key, op,
                                  on_progress=lambda m: jobs.update(job_id, message=m))
        jobs.update(job_id, message="downloading video")
        mp4 = await g.download_veo_video(ctx.http, sample, cfg.gemini_api_key)
        res = await storage.save_result(mp4, project=job["project"], subpath=job["subpath"],
                                        filename=job["filename"] or "veo", ext="mp4", cfg=cfg)
        jobs.update(job_id, status="done", message="complete", results=[res])

    ctx.poll_and_finish = _poll_and_finish  # server.py uses this to resume jobs on startup

    async def _run_veo_job(job_id: str, model_id: str, prompt: str, start_image,
                           aspect_ratio: str, duration_seconds: int, generate_audio: bool):
        try:
            op = await g.start_veo(ctx.http, cfg.gemini_api_key, model_id, prompt,
                                   start_image=start_image, aspect_ratio=aspect_ratio,
                                   duration_seconds=duration_seconds, generate_audio=generate_audio)
            jobs.update(job_id, operation_name=op, message="submitted to Veo")
            await _poll_and_finish(job_id, op)
        except Exception as e:  # job boundary: everything becomes a readable failed status
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def generate_video(prompt: str, project: str, model: str = "veo-3-fast",
                             start_image: str | None = None, aspect_ratio: str = "16:9",
                             duration_seconds: int = 8, generate_audio: bool = True,
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate a video with Veo (takes minutes — returns a job_id immediately; poll with
        job_status). model: veo-3 | veo-3-fast | veo-2. aspect_ratio: 16:9 or 9:16.
        start_image: optional image to animate (URL or workspace path). COSTS REAL MONEY per clip."""
        _require_key()
        if aspect_ratio not in g.VIDEO_ASPECTS:
            raise g.GenerationError(f"aspect_ratio must be one of {g.VIDEO_ASPECTS}")
        model_id = g.resolve_video_model(model)
        storage.validate_project(project, cfg=cfg)  # fail fast before paying Google
        img = None
        if start_image:
            r = await storage.resolve_input(start_image, cfg=cfg, kind="image")
            img = (r.mime, r.data)
        job = jobs.create(kind="veo", model=model_id, prompt=prompt, project=project,
                          subpath=subpath, filename=filename)
        asyncio.create_task(_run_veo_job(job["id"], model_id, prompt, img,
                                         aspect_ratio, duration_seconds, generate_audio))
        return {"job_id": job["id"], "status": "running",
                "note": "Video generation takes 1-6 minutes. Poll with job_status."}

    async def _run_local_video_job(job_id, prompt, neg, w, h, length, steps, seed, fps):
        try:
            backends = [
                {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
                {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
            ]
            spec = g.LOCAL_VIDEO_MODELS["wan"]
            chosen, sel = await g.select_comfy(ctx.http, backends, require_unets=[spec["high"], spec["low"]])
            if not chosen:
                raise g.GenerationError(f"No local video backend available ({sel}) — all busy/gaming or lacking Wan models")
            jobs.update(job_id, message=f"generating on {sel}…")
            mp4 = await g.call_comfy_video(ctx.http, chosen, prompt, model="wan", negative_prompt=neg,
                                           width=w, height=h, length=length, steps=steps, seed=seed, fps=fps)
            job = jobs.get(job_id)
            res = await storage.save_result(mp4, project=job["project"], subpath=job["subpath"],
                                            filename=job["filename"] or "wan", ext="mp4", cfg=cfg)
            jobs.update(job_id, status="done", message=f"complete (engine=wan@{sel})", results=[res])
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def generate_video_local(prompt: str, project: str, aspect_ratio: str = "16:9",
                                   seconds: float = 3.0, negative_prompt: str | None = None,
                                   steps: int | None = None, seed: int | None = None,
                                   subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate a video LOCALLY on a GPU box via ComfyUI + Wan 2.2 — FULLY UNCENSORED (no cloud
        filter), FREE (no per-clip cost). Routes across the gen-pool (laybackrig→maingamingrig),
        auto-skipping any box being gamed on / running chim, and only boxes that have the Wan models.
        Takes a few minutes — returns a job_id immediately; poll with job_status.
        aspect_ratio: 1:1, 16:9, 9:16, 4:3, 3:4. seconds: clip length (~2-5)."""
        w, h = g.VIDEO_AR.get(aspect_ratio, (832, 480))
        fps = g.LOCAL_VIDEO_MODELS["wan"]["fps"]
        length = max(17, int(round(seconds * fps / 4)) * 4 + 1)  # Wan wants 4n+1 frames
        storage.validate_project(project, cfg=cfg)
        job = jobs.create(kind="wan", model="wan", prompt=prompt, project=project,
                          subpath=subpath, filename=filename)
        asyncio.create_task(_run_local_video_job(job["id"], prompt, negative_prompt, w, h, length, steps, seed, fps))
        return {"job_id": job["id"], "status": "running",
                "note": "Local video (Wan 2.2) takes a few minutes. Poll with job_status."}

    async def _run_i2v_job(job_id, image_bytes, prompt, neg, w, h, length, steps, seed, fps,
                           model="wan-i2v", lora=None, lora_strength=1.0):
        try:
            backends = [
                {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
                {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
            ]
            spec = g.LOCAL_VIDEO_MODELS[model]
            chosen, sel = await g.select_comfy(ctx.http, backends, require_unets=[spec["high"], spec["low"]])
            if not chosen:
                raise g.GenerationError(f"No I2V backend available ({sel}) — all busy/gaming or lacking the Wan I2V models")
            jobs.update(job_id, message=f"animating on {sel}…")
            mp4 = await g.call_comfy_i2v(ctx.http, chosen, image_bytes, prompt, model=model, negative_prompt=neg,
                                         width=w, height=h, length=length, steps=steps, seed=seed, fps=fps,
                                         lora=lora, lora_strength=lora_strength)
            job = jobs.get(job_id)
            res = await storage.save_result(mp4, project=job["project"], subpath=job["subpath"],
                                            filename=job["filename"] or "i2v", ext="mp4", cfg=cfg)
            eng = f"{model}{('+lora:' + lora) if lora else ''}@{sel}"
            jobs.update(job_id, status="done", message=f"complete (engine={eng})", results=[res])
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def animate_image(image: str, project: str, prompt: str = "", aspect_ratio: str = "16:9",
                            seconds: float = 3.0, nsfw: bool = False, lora: str | None = None,
                            lora_strength: float = 1.0, negative_prompt: str | None = None,
                            steps: int | None = None, seed: int | None = None,
                            subpath: str | None = None, filename: str | None = None) -> dict:
        """Animate a still IMAGE into a video LOCALLY via ComfyUI + Wan 2.2 I2V — FULLY UNCENSORED,
        FREE. image: an https URL or a '<Project>/<path>' workspace path (e.g. a fresh generate_image
        output, or anything in Carbon Drive). prompt: optional motion/camera guidance ("slow zoom in,
        hair blowing"). nsfw=True adds the broad Wan NSFW-22 LoRA pair (= lora='general'). lora: pick a
        specific curated Wan I2V LoRA pair instead (see list_models -> local_loras.video_pairs), applied
        on the wan-i2v base experts. Routes across the gen-pool, skipping any box being gamed-on / running chim.
        Takes a few minutes — returns a job_id immediately; poll with job_status.
        aspect_ratio: 1:1, 16:9, 9:16, 4:3, 3:4. seconds: clip length (~2-5)."""
        src = await storage.resolve_input(image, cfg=cfg, kind="image")
        if lora:
            g.resolve_wan_i2v_lora(lora)  # validate now (the background job would otherwise swallow it)
            model = "wan-i2v"             # base experts; the selected pair is applied on top
        else:
            model = "wan-i2v-spicy" if nsfw else "wan-i2v"
        w, h = g.VIDEO_AR.get(aspect_ratio, (832, 480))
        fps = g.LOCAL_VIDEO_MODELS[model]["fps"]
        length = max(17, int(round(seconds * fps / 4)) * 4 + 1)  # Wan wants 4n+1 frames
        storage.validate_project(project, cfg=cfg)
        job = jobs.create(kind=model, model=model, prompt=prompt or "(animate)", project=project,
                          subpath=subpath, filename=filename)
        asyncio.create_task(_run_i2v_job(job["id"], src.data, prompt, negative_prompt, w, h, length,
                                         steps, seed, fps, model, lora, lora_strength))
        return {"job_id": job["id"], "status": "running",
                "note": "Local I2V (Wan 2.2) takes a few minutes. Poll with job_status."}

    async def _run_clip_job(job_id, prompt, img_model, motion_prompt, iw, ih, vw, vh,
                            length, steps, seed, fps, neg, do_upscale,
                            char_refs=None, char_mode=None, char_weight=None):
        try:
            backends = [
                {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
                {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
            ]
            job = jobs.get(job_id)
            base = storage.safe_filename(job["filename"] or prompt[:40])
            results = []
            # 1) still — IPAdapter (consistent character) if a reference was given, else plain gen
            jobs.update(job_id, message="generating still…")
            chosen, sel = await g.select_comfy(ctx.http, backends)
            if not chosen:
                raise g.GenerationError(f"No GPU backend for the still ({sel})")
            if char_refs:
                preset, wt = _ipa_preset(char_mode)
                stills = await g.call_comfy_ref(ctx.http, chosen, char_refs, prompt, model="pony",
                                                preset=preset, weight=char_weight, weight_type=wt,
                                                negative_prompt=neg, width=iw, height=ih, seed=seed)
                sel = f"{sel}/ipadapter-{char_mode}"
            else:
                stills = await g.call_comfy(ctx.http, chosen, prompt, model=img_model,
                                            negative_prompt=neg, width=iw, height=ih, seed=seed)
            still_bytes = stills[0]
            still_res = await storage.save_result(still_bytes, project=job["project"], subpath=job["subpath"],
                                                  filename=base, ext="png", cfg=cfg)
            still_res["engine"] = f"comfy:{img_model}@{sel}"; still_res["kind"] = "still"
            results.append(still_res)
            # 2) optional 4x upscale -> hi-res STILL deliverable
            if do_upscale:
                jobs.update(job_id, message="upscaling still…", results=results)
                uc, usel = await g.select_comfy(ctx.http, backends)
                if uc:
                    up = await g.call_comfy_upscale(ctx.http, uc, still_bytes)
                    up_res = await storage.save_result(up, project=job["project"], subpath=job["subpath"],
                                                       filename=f"{base}_4x", ext="png", cfg=cfg)
                    up_res["engine"] = f"esrgan-4x@{usel}"; up_res["kind"] = "upscaled"
                    results.append(up_res)
            # 3) animate the (target-res) still -> video
            jobs.update(job_id, message="animating…", results=results)
            spec = g.LOCAL_VIDEO_MODELS["wan-i2v"]
            vc, vsel = await g.select_comfy(ctx.http, backends, require_unets=[spec["high"], spec["low"]])
            if not vc:
                raise g.GenerationError(f"No I2V backend for the clip ({vsel})")
            mp4 = await g.call_comfy_i2v(ctx.http, vc, still_bytes, motion_prompt, negative_prompt=neg,
                                         width=vw, height=vh, length=length, steps=steps, seed=seed, fps=fps)
            vid_res = await storage.save_result(mp4, project=job["project"], subpath=job["subpath"],
                                                filename=base, ext="mp4", cfg=cfg)
            vid_res["engine"] = f"wan-i2v@{vsel}"; vid_res["kind"] = "video"
            results.append(vid_res)
            jobs.update(job_id, status="done", message=f"complete (still@{sel} -> video@{vsel})", results=results)
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def generate_clip(prompt: str, project: str, img_model: str = "pony",
                            aspect_ratio: str = "16:9", seconds: float = 3.0,
                            motion_prompt: str | None = None, upscale: bool = False,
                            character: str | None = None, character_mode: str | None = None,
                            character_weight: float | None = None,
                            negative_prompt: str | None = None, steps: int | None = None,
                            seed: int | None = None, subpath: str | None = None,
                            filename: str | None = None) -> dict:
        """One-shot ASSET PIPELINE: generate a still from `prompt`, optionally 4x-upscale it, then
        ANIMATE it into a video (Wan 2.2 I2V) — all LOCAL on the GPU pool, uncensored & free. Returns
        a job_id immediately; poll job_status (results carry the still, the upscaled still if
        upscale=True, and the video, each tagged with `kind`).
          character: a SAVED character name (see save_character) — the still is generated as THAT
            character (IPAdapter, forces the pony model), then animated. character_mode/character_weight
            override its saved defaults.
          img_model: pony (fast) | flux (best quality) — used only when no character is given.
          aspect_ratio: 1:1,3:4,4:3,9:16,16:9. seconds: ~2-5. motion_prompt: motion/camera guidance.
        Note: upscale gives a hi-res STILL; the clip's resolution is set by aspect_ratio (Wan resizes the
        start frame), so upscaling doesn't raise the video's resolution."""
        char_refs = char_mode = char_weight = None
        if character:
            char_refs, entry = chars.read_references(character)  # raises if unknown
            char_mode = character_mode or entry.get("mode") or "character"
            char_weight = character_weight if character_weight is not None else entry.get("weight", 0.8)
        iw, ih = LOCAL_AR.get(aspect_ratio, (896, 1152))
        vw, vh = g.VIDEO_AR.get(aspect_ratio, (832, 480))
        fps = g.LOCAL_VIDEO_MODELS["wan-i2v"]["fps"]
        length = max(17, int(round(seconds * fps / 4)) * 4 + 1)
        storage.validate_project(project, cfg=cfg)
        job = jobs.create(kind="clip", model="wan-i2v", prompt=prompt, project=project,
                          subpath=subpath, filename=filename)
        asyncio.create_task(_run_clip_job(job["id"], prompt, img_model, motion_prompt or prompt,
                                          iw, ih, vw, vh, length, steps, seed, fps, negative_prompt, upscale,
                                          char_refs=char_refs, char_mode=char_mode, char_weight=char_weight))
        return {"job_id": job["id"], "status": "running",
                "note": "Asset pipeline (" + (f"character:{character} -> " if character else "")
                + "still -> " + ("upscale -> " if upscale else "") + "animate). Poll job_status."}

    async def _run_montage_job(job_id, shots, neg, w, h, length, steps, fps):
        try:
            spec = g.LOCAL_VIDEO_MODELS["wan"]
            backends = [
                {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
                {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
            ]
            elig = await g.eligible_comfy_backends(ctx.http, backends, require_unets=[spec["high"], spec["low"]])
            if not elig:
                raise g.GenerationError("No local video backend available for montage (busy/gaming/chim or no Wan models)")
            jobs.update(job_id, message=f"rendering {len(shots)} shots across {len(elig)} box(es)…")

            async def _seg(i, shot):
                b = elig[i % len(elig)]
                return await g.call_comfy_video(ctx.http, b["url"], shot, model="wan", negative_prompt=neg,
                                                width=w, height=h, length=length, steps=steps, fps=fps,
                                                free_after=False)  # free once after all segments
            segs = await asyncio.gather(*[_seg(i, s) for i, s in enumerate(shots)])  # ordered → preserves shot order
            for b in elig:
                await g.comfy_free(ctx.http, b["url"])  # hand the cards back to chim/games
            jobs.update(job_id, message="stitching segments…")
            final = await video.concat(list(segs))
            job = jobs.get(job_id)
            res = await storage.save_result(final, project=job["project"], subpath=job["subpath"],
                                            filename=job["filename"] or "montage", ext="mp4", cfg=cfg)
            jobs.update(job_id, status="done", message=f"complete ({len(shots)} shots across {len(elig)} box(es))", results=[res])
        except Exception as e:
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def generate_video_montage(shots: list[str], project: str, aspect_ratio: str = "16:9",
                                     seconds_per_shot: float = 3.0, negative_prompt: str | None = None,
                                     steps: int | None = None, subpath: str | None = None,
                                     filename: str | None = None) -> dict:
        """Make a MULTI-SHOT video: render each shot-prompt as a Wan 2.2 segment IN PARALLEL across the
        GPU pool (laybackrig + maingamingrig), then ffmpeg-concat them into one clip — leverages every
        free box at once, so a 4-shot clip on 2 boxes finishes in ~2 shots' time. Skips boxes being
        gamed-on / running chim. shots: 1-8 prompts (in order). Takes minutes — returns a job_id; poll
        job_status. aspect_ratio: 1:1,16:9,9:16,4:3,3:4; seconds_per_shot ~2-5."""
        if not shots or len(shots) > 8:
            raise g.GenerationError("Provide 1-8 shot prompts")
        w, h = g.VIDEO_AR.get(aspect_ratio, (832, 480))
        fps = g.LOCAL_VIDEO_MODELS["wan"]["fps"]
        length = max(17, int(round(seconds_per_shot * fps / 4)) * 4 + 1)
        storage.validate_project(project, cfg=cfg)
        job = jobs.create(kind="montage", model="wan", prompt=" | ".join(shots)[:200],
                          project=project, subpath=subpath, filename=filename)
        asyncio.create_task(_run_montage_job(job["id"], shots, negative_prompt, w, h, length, steps, fps))
        return {"job_id": job["id"], "status": "running",
                "note": f"{len(shots)}-shot montage rendering in parallel across the pool. Poll job_status."}

    @mcp.tool()
    async def job_status(job_id: str) -> dict:
        """Status of a generate_video / generate_video_local / generate_video_montage job: running, done, or failed."""
        job = jobs.get(job_id)
        if not job:
            raise g.GenerationError(f"No job '{job_id}'")
        return {k: job[k] for k in ("id", "status", "message", "results", "error", "created_at", "updated_at")}
