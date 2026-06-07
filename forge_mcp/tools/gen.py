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


def _square_pad(data: bytes, size: int = 512, pad_frac: float = 0.12) -> bytes:
    """Center a (already cut-out) image on a transparent square canvas with padding, sized to `size`."""
    from PIL import Image
    im = Image.open(BytesIO(data)).convert("RGBA")
    inner = max(1, int(size * (1 - 2 * pad_frac)))
    w, h = im.size
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
        """Generate image(s) from a text prompt with Imagen 4.
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
        images = await g.call_imagen(ctx.http, cfg.gemini_api_key, model_id, prompt,
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
    async def generate_local(prompt: str, project: str, model: str = "pony",
                             aspect_ratio: str = "3:4", steps: int | None = None,
                             cfg_scale: float | None = None, negative_prompt: str | None = None,
                             seed: int | None = None, lora: str | None = None,
                             lora_strength: float = 1.0, fallback: bool = True,
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
        list_models -> local_loras); lora_strength ~0.6-1.0. SDXL models only; ignored on cloud fallback."""
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
                engine = f"comfy:{model}{('+lora:' + lora) if lora else ''}@{sel}"
                try:
                    images = await g.call_comfy(ctx.http, chosen_url, prompt, model=model,
                                                negative_prompt=negative_prompt, width=w, height=h,
                                                steps=steps, cfg=cfg_scale, seed=seed, loras=loras)
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
            images = await g.call_imagen(ctx.http, cfg.gemini_api_key,
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
        images = await g.call_gemini_image(ctx.http, cfg.gemini_api_key, prompt, refs)
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
                                      lora_strength: float = 1.0, subpath: str | None = None,
                                      filename: str | None = None) -> dict:
        """Generate an image that KEEPS the character / face / style of a reference (IPAdapter) — LOCAL
        on the GPU pool, uncensored. Use it for the SAME character across scenes/poses, a consistent
        mascot, or transferring an art style. SDXL/pony only.
          character: a SAVED character name (see save_character / list_characters) — its stored
            reference + default mode/weight are used. OR
          reference_image: a one-off https URL / '<Project>/<path>' workspace path.
          mode: 'character' (subject + style) | 'face' (portrait, face-locked) | 'style' (style only).
          weight: 0.4-1.2 — how strongly to follow the reference (lower = more prompt freedom).
          lora: optional SDXL LoRA (curated alias or installed filename; see list_models) applied on top.
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
                                      width=w, height=h, steps=steps, seed=seed, loras=loras)
        res = await storage.save_result(imgs[0], project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or prompt[:40]), ext="png", cfg=cfg)
        res["engine"] = f"ipadapter-{mode}{('+lora:' + lora) if lora else ''}@{sel}"
        return {"image": res, "engine": res["engine"], "mode": mode, "character": character}

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
            raster = (await g.call_imagen(ctx.http, cfg.gemini_api_key,
                                          g.resolve_image_model("imagen-4"), prompt,
                                          sample_count=1, aspect_ratio="1:1"))[0]
            engine_str = "imagen-4"
        out = raster
        if transparent:
            opts = PipelineOptions(model="isnet-general-use", alpha_matting=True, edge_smooth=True,
                                   auto_trim=True, color_remove=True, color_auto_detect=True)
            out = await engine.run_pipeline(raster, opts)
            out = _square_pad(out, size=size)
        res = await storage.save_result(out, project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or subject[:40]), ext="png", cfg=cfg)
        res["engine"] = f"icon/{engine_str}"
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
