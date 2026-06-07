"""Generation tools — Imagen, Gemini edit, Veo (async jobs)."""
import asyncio

from forge_mcp import generation as g
from forge_mcp import storage

# Aspect ratio -> (width, height) at SDXL/Flux-friendly resolutions for local generation.
LOCAL_AR = {
    "1:1": (1024, 1024), "3:4": (896, 1152), "4:3": (1152, 896),
    "9:16": (768, 1344), "16:9": (1344, 768),
}


def register(mcp, ctx):
    cfg, jobs = ctx.cfg, ctx.jobs

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
                             seed: int | None = None, fallback: bool = True,
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Generate an image LOCALLY on the GPU box via ComfyUI — FULLY UNCENSORED (dark/gore/adult
        OK; no cloud content filter). Saves into the project's workspace folder.
          model: 'pony' (fast SDXL ~5-8s, maximally uncensored) | 'flux' (best quality/coherence ~15-20s).
          aspect_ratio: 1:1, 3:4, 4:3, 9:16, 16:9.
        Pony score-tags + a sane negative prompt are applied automatically (override via negative_prompt).
        If the GPU box is unreachable/busy and fallback=True, falls back to cloud Imagen 4 (filtered)."""
        w, h = LOCAL_AR.get(aspect_ratio, (896, 1152))
        engine = f"comfy:{model}"
        images, reason = None, None
        # Routing policy: laybackrig's ComfyUI first; overflow to maingamingrig when laybackrig
        # is busy (a gen already running) — and never a box being gamed on (presence-gated).
        backends = [
            {"url": cfg.comfy_url, "presence_url": cfg.comfy_presence_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "presence_url": cfg.comfy_overflow_presence_url, "label": "maingamingrig"},
        ]
        if any(b["url"] for b in backends):
            chosen_url, sel = await g.select_comfy(ctx.http, backends)
            if chosen_url:
                engine = f"comfy:{model}@{sel}"
                try:
                    images = await g.call_comfy(ctx.http, chosen_url, prompt, model=model,
                                                negative_prompt=negative_prompt, width=w, height=h,
                                                steps=steps, cfg=cfg_scale, seed=seed)
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
                                      negative_prompt=negative_prompt, width=w, height=h)  # fresh seed each call
            res = await storage.save_result(imgs[0], project=project, subpath=subpath,
                                            filename=f"{base}-{i + 1}", ext="png", cfg=cfg)
            res["engine"] = f"comfy:{model}@{backend['label']}"
            return res

        settled = await asyncio.gather(*[_one(i, elig[i % len(elig)]) for i in range(count)],
                                       return_exceptions=True)
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

    @mcp.tool()
    async def job_status(job_id: str) -> dict:
        """Status of a generate_video / generate_video_local job: running (with progress), done (results), or failed."""
        job = jobs.get(job_id)
        if not job:
            raise g.GenerationError(f"No job '{job_id}'")
        return {k: job[k] for k in ("id", "status", "message", "results", "error", "created_at", "updated_at")}
