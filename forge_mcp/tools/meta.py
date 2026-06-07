"""Status + discovery tools."""
import os
import shutil

from backend.processing import AVAILABLE_MODELS
from forge_mcp import engine
from forge_mcp import generation as g
from forge_mcp.generation import IMAGE_MODEL_ALIASES, VIDEO_MODEL_ALIASES


def register(mcp, ctx):
    cfg, jobs = ctx.cfg, ctx.jobs

    @mcp.tool()
    async def forge_status() -> dict:
        """Health of the forge service: engine models, ffmpeg, workspace mount, jobs, config."""
        ws_ok, ws_err = True, None
        try:
            probe = os.path.join(cfg.workspace_root, ".forge-write-test")
            with open(probe, "w") as f:
                f.write("ok")
            os.unlink(probe)
        except OSError as e:
            ws_ok, ws_err = False, str(e)
        cache_files = 0
        files_root = os.path.join(cfg.results_root, "files")
        if os.path.isdir(files_root):
            cache_files = len(os.listdir(files_root))
        return {
            "engine": engine.status(),
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "workspace_writable": ws_ok,
            "workspace_error": ws_err,
            "gemini_key_configured": bool(cfg.gemini_api_key),
            "results_cached": cache_files,
            "jobs_running": jobs.running_count(),
            "public_url": cfg.public_url,
        }

    @mcp.tool()
    async def list_models() -> dict:
        """Generation models. `local_aliases` are curated names; `installed_checkpoints` is the LIVE
        list of checkpoints actually on each GPU box (auto-discovered — drop a .safetensors in
        ComfyUI's models/checkpoints + sync and it appears here, usable by filename, no code change).
        The `model` param of generate_local / generate_icon / generate_clip / generate_with_reference
        accepts a local alias OR any installed checkpoint filename. Plus bg-removal + cloud aliases."""
        backends = [
            {"url": cfg.comfy_url, "label": "laybackrig"},
            {"url": cfg.comfy_overflow_url, "label": "maingamingrig"},
        ]
        installed = {}
        for b in backends:
            if b["url"]:
                cks = await g.comfy_checkpoints(ctx.http, b["url"])
                if cks:
                    installed[b["label"]] = cks
        return {
            "local_aliases": {a: {"checkpoint": c, "family": f, "style": s} for a, (c, f, s) in g.LOCAL_MODELS.items()},
            "installed_checkpoints": installed,
            "local_video": list(g.LOCAL_VIDEO_MODELS.keys()),
            "background_removal": AVAILABLE_MODELS,
            "image_generation_cloud": list(IMAGE_MODEL_ALIASES.keys()),
            "image_edit": ["gemini-2.5-flash-image (default — used by edit_image)"],
            "video_generation_cloud": list(VIDEO_MODEL_ALIASES.keys()),
        }

    @mcp.tool()
    async def list_projects() -> dict:
        """Workspace project folders that tools can save into (the 'project' parameter)."""
        try:
            dirs = sorted(d for d in os.listdir(cfg.workspace_root)
                          if os.path.isdir(os.path.join(cfg.workspace_root, d)) and not d.startswith("."))
        except OSError as e:
            return {"error": f"workspace not reachable: {e}", "projects": []}
        return {"projects": dirs}
