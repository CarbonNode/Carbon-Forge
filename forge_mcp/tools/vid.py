"""ffmpeg-backed video tools."""
from forge_mcp import storage, video


def _ext_for(mime: str) -> str:
    return {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov"}.get(mime, "mp4")


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def video_trim(video_input: str, start: str, end: str, project: str,
                         subpath: str | None = None, filename: str | None = None) -> dict:
        """Trim a video to [start, end]. Times as seconds ('12.5') or '[HH:]MM:SS[.ms]'.
        video_input: https URL or workspace path '<Project>/<relative path>'."""
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        out = await video.trim(src.data, start, end, in_ext=_ext_for(src.mime))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "trimmed", ext="mp4", cfg=cfg)

    @mcp.tool()
    async def video_extract_frames(video_input: str, timestamps: list[str], project: str,
                                   image_format: str = "png", subpath: str | None = None,
                                   filename: str | None = None) -> dict:
        """Extract still frames at the given timestamps (max 20). image_format: png or jpg."""
        if not timestamps or len(timestamps) > 20:
            raise video.VideoError("Provide 1-20 timestamps")
        if image_format not in ("png", "jpg"):
            raise video.VideoError("image_format must be png or jpg")
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        frames = await video.extract_frames(src.data, timestamps, fmt=image_format,
                                            in_ext=_ext_for(src.mime))
        base = filename or "frame"
        results = []
        for ts, frame in zip(timestamps, frames):
            safe_ts = str(ts).replace(":", "-").replace(".", "_")
            results.append(await storage.save_result(
                frame, project=project, subpath=subpath,
                filename=f"{base}-{safe_ts}", ext=image_format, cfg=cfg))
        return {"count": len(results), "frames": results}

    @mcp.tool()
    async def video_convert(video_input: str, output_format: str, project: str,
                            crf: int | None = None, scale: int | None = None,
                            subpath: str | None = None, filename: str | None = None) -> dict:
        """Convert/compress a video. output_format: mp4, webm, or gif. crf: quality
        (lower=better; mp4 default 23, webm 32). scale: target height in px (e.g. 720)."""
        src = await storage.resolve_input(video_input, cfg=cfg, kind="video")
        out = await video.convert(src.data, output_format, crf=crf, scale=scale,
                                  in_ext=_ext_for(src.mime))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "converted", ext=output_format, cfg=cfg)
