"""Quick utility conversions — audio, image format, Draco GLB, media info.

The 'just convert this' tools: inputs are https URLs or workspace paths
'<Project>/<relative path>' (files uploaded in Conduit chat land at
'<Project>/.conduit/uploads/<name>'); outputs come back as a public URL plus a
workspace copy — so a chat upload is one tool call away from a converted file."""
import asyncio

from forge_mcp import assets3d, imaging, storage, video

_AUDIO_EXT = {"audio/mpeg": "mp3", "audio/wav": "wav", "audio/ogg": "ogg",
              "audio/flac": "flac", "audio/mp4": "m4a"}
_VIDEO_EXT = {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov",
              "video/x-msvideo": "avi"}


def _media_ext(mime: str) -> str:
    return _AUDIO_EXT.get(mime) or _VIDEO_EXT.get(mime, "bin")


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def audio_convert(audio_input: str, output_format: str, project: str,
                            bitrate_kbps: int | None = None, sample_rate_hz: int | None = None,
                            channels: int | None = None, subpath: str | None = None,
                            filename: str | None = None) -> dict:
        """Convert audio to mp3, wav, ogg, opus, flac, or m4a. Accepts audio OR video input
        (a video's audio track is extracted). audio_input: https URL or workspace path
        '<Project>/<relative path>' (chat uploads: '<Project>/.conduit/uploads/<name>').
        Optional: bitrate_kbps (lossy formats), sample_rate_hz (e.g. 44100), channels
        (1=mono, 2=stereo)."""
        src = await storage.resolve_input(audio_input, cfg=cfg, kind="audio")
        out = await video.audio_convert(src.data, output_format, bitrate_kbps=bitrate_kbps,
                                        sample_rate_hz=sample_rate_hz, channels=channels,
                                        in_ext=_media_ext(src.mime))
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "converted",
                                         ext=output_format, cfg=cfg)

    @mcp.tool()
    async def audio_trim(audio_input: str, start: str, end: str, project: str,
                         subpath: str | None = None, filename: str | None = None) -> dict:
        """Trim audio to [start, end]. Times as seconds ('12.5') or '[HH:]MM:SS[.ms]'.
        Keeps the input format (lossless stream-copy when possible); a video input yields
        a trimmed mp3 of its audio track."""
        src = await storage.resolve_input(audio_input, cfg=cfg, kind="audio")
        in_ext = _media_ext(src.mime)
        out_ext = in_ext if src.mime in _AUDIO_EXT else "mp3"
        out = await video.audio_trim(src.data, start, end, in_ext=in_ext, out_ext=out_ext)
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "trimmed", ext=out_ext, cfg=cfg)

    @mcp.tool()
    async def image_convert(image_input: str, output_format: str, project: str,
                            quality: int | None = None, max_dimension: int | None = None,
                            background: str = "#ffffff", subpath: str | None = None,
                            filename: str | None = None) -> dict:
        """Convert/compress an image: png, jpg, webp, gif, bmp, ico, or avif. quality 1-100
        (jpg/webp/avif), max_dimension shrinks the long edge, background fills transparency
        for jpg. Plain format/size conversion — for AI edits use edit_image."""
        src = await storage.resolve_input(image_input, cfg=cfg, kind="image")
        out = await asyncio.to_thread(imaging.convert_image, src.data, output_format,
                                      quality=quality, max_dimension=max_dimension,
                                      background=background)
        ext = "jpg" if output_format.lower() == "jpeg" else output_format.lower()
        return await storage.save_result(out, project=project, subpath=subpath,
                                         filename=filename or "converted", ext=ext, cfg=cfg)

    @mcp.tool()
    async def draco_compress(model_input: str, project: str,
                             quantize_position: int | None = None,
                             quantize_normal: int | None = None,
                             quantize_texcoord: int | None = None,
                             subpath: str | None = None, filename: str | None = None) -> dict:
        """Draco-compress a 3D model (.glb binary glTF) — typically shrinks geometry 3-10x.
        Optional quantization bits (position/normal/texcoord; lower = smaller but lossier,
        omit for sensible defaults). Returns the compressed .glb + before/after sizes."""
        src = await storage.resolve_input(model_input, cfg=cfg, kind="model")
        before = assets3d.glb_stats(src.data)
        if before["draco_compressed"]:
            raise assets3d.ModelError("Model is already Draco-compressed")
        out = await assets3d.draco_compress(src.data, quantize_position=quantize_position,
                                            quantize_normal=quantize_normal,
                                            quantize_texcoord=quantize_texcoord)
        result = await storage.save_result(out, project=project, subpath=subpath,
                                           filename=filename or "draco", ext="glb", cfg=cfg)
        result["bytes_before"] = len(src.data)
        result["compression_ratio"] = round(len(src.data) / max(1, len(out)), 2)
        return result

    @mcp.tool()
    async def media_info(file_input: str) -> dict:
        """Inspect any media file: image (dimensions/mode), audio/video (duration, codecs,
        bitrate, resolution, fps — ffprobe), or .glb 3D model (meshes/materials/textures).
        file_input: https URL or '<Project>/<relative path>'."""
        src = await storage.resolve_input(file_input, cfg=cfg, kind="any")
        info = {"mime": src.mime or "unknown", "bytes": len(src.data)}
        if storage.is_image_mime(src.mime):
            info["image"] = await asyncio.to_thread(imaging.image_info, src.data)
        elif storage.is_model_mime(src.mime):
            info["model"] = assets3d.glb_stats(src.data)
        else:
            raw = await video.probe(src.data, in_ext=_media_ext(src.mime))
            info["media"] = video.summarize_probe(raw)
        return info
