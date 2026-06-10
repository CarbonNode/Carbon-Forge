"""ffmpeg wrappers — video trim/frames/convert, audio convert/trim, ffprobe.
Bytes in -> bytes out via temp files."""
import asyncio
import json
import os
import re
import tempfile

FFMPEG_TIMEOUT_S = 600


class VideoError(Exception):
    """Readable, user-facing video-processing failure."""


def to_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", s)
    if not m:
        raise VideoError(f"Bad time '{value}' — use seconds or [HH:]MM:SS[.ms]")
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def build_trim_cmd(inp, outp, start_s, end_s, *, reencode):
    cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-to", str(end_s), "-i", inp]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    return cmd + [outp]


def build_convert_cmd(inp, outp, fmt, *, crf, scale):
    vf = []
    if scale:
        vf.append(f"scale=-2:{scale}")
    cmd = ["ffmpeg", "-y", "-i", inp]
    if fmt == "mp4":
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf or 23), "-c:a", "aac"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
    elif fmt == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-crf", str(crf or 32), "-b:v", "0", "-c:a", "libopus"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
    elif fmt == "gif":
        scale_f = f"scale=-2:{scale or 480}:flags=lanczos"
        cmd += ["-vf", f"fps=12,{scale_f},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"]
    else:
        raise VideoError(f"Unsupported format '{fmt}' (mp4, webm, gif)")
    return cmd + [outp]


def build_frame_cmd(inp, outp, ts_s, fmt):
    return ["ffmpeg", "-y", "-ss", str(ts_s), "-i", inp, "-frames:v", "1", outp]


async def _run(cmd) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        raise VideoError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s")
    if proc.returncode != 0:
        tail = (stderr or b"").decode(errors="replace")[-800:]
        raise VideoError(f"ffmpeg failed (exit {proc.returncode}): {tail}")


class _Tmp:
    """Temp file that survives Windows handle semantics (close before ffmpeg reads it)."""

    def __init__(self, suffix, data=None):
        fd, self.path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            if data:
                f.write(data)

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def cleanup(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass


async def trim(data: bytes, start, end, in_ext="mp4") -> bytes:
    start_s, end_s = to_seconds(start), to_seconds(end)
    if end_s <= start_s:
        raise VideoError(f"end ({end_s}s) must be after start ({start_s}s)")
    src, dst = _Tmp(f".{in_ext}", data), _Tmp(".mp4")
    try:
        await _run(build_trim_cmd(src.path, dst.path, start_s, end_s, reencode=False))
        out = dst.read()
        if len(out) < 1024:  # stream-copy produced garbage (keyframe issues) -> re-encode
            await _run(build_trim_cmd(src.path, dst.path, start_s, end_s, reencode=True))
            out = dst.read()
        if len(out) < 1024:
            raise VideoError("Trim produced an empty file")
        return out
    finally:
        src.cleanup()
        dst.cleanup()


async def extract_frames(data: bytes, timestamps, fmt="png", in_ext="mp4") -> list:
    src = _Tmp(f".{in_ext}", data)
    out = []
    try:
        for ts in timestamps:
            dst = _Tmp(f".{fmt}")
            try:
                await _run(build_frame_cmd(src.path, dst.path, to_seconds(ts), fmt))
                frame = dst.read()
                if not frame:
                    raise VideoError(f"No frame at {ts} (past end of video?)")
                out.append(frame)
            finally:
                dst.cleanup()
        return out
    finally:
        src.cleanup()


def build_concat_cmd(list_path, outp, *, reencode):
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c", "copy"]
    return cmd + [outp]


async def concat(segments, in_ext="mp4") -> bytes:
    """Concatenate video segments (bytes) into one clip via the ffmpeg concat demuxer. Tries
    stream-copy first (fast; segments share codec/res when from the same Wan workflow), falls
    back to re-encode if that produces garbage. Used to stitch parallel multi-shot segments."""
    if not segments:
        raise VideoError("No segments to concat")
    if len(segments) == 1:
        return segments[0]
    tmps = [_Tmp(f".{in_ext}", s) for s in segments]
    listing = "\n".join(f"file '{t.path}'" for t in tmps)
    lst = _Tmp(".txt", listing.encode())
    dst = _Tmp(".mp4")
    try:
        out = b""
        try:
            await _run(build_concat_cmd(lst.path, dst.path, reencode=False))
            out = dst.read()
        except VideoError:
            out = b""
        if len(out) < 1024:  # stream-copy mismatch -> re-encode
            await _run(build_concat_cmd(lst.path, dst.path, reencode=True))
            out = dst.read()
        if len(out) < 1024:
            raise VideoError("Concat produced an empty file")
        return out
    finally:
        for t in tmps:
            t.cleanup()
        lst.cleanup()
        dst.cleanup()


async def convert(data: bytes, fmt, *, crf=None, scale=None, in_ext="mp4") -> bytes:
    src, dst = _Tmp(f".{in_ext}", data), _Tmp(f".{fmt}")
    try:
        await _run(build_convert_cmd(src.path, dst.path, fmt, crf=crf, scale=scale))
        out = dst.read()
        if not out:
            raise VideoError("Conversion produced an empty file")
        return out
    finally:
        src.cleanup()
        dst.cleanup()


# --- audio ---------------------------------------------------------------

# format -> (ffmpeg codec, default bitrate kbps for lossy; None = lossless/uncompressed)
AUDIO_CODECS = {
    "mp3": ("libmp3lame", 192),
    "wav": ("pcm_s16le", None),
    "ogg": ("libvorbis", 160),
    "opus": ("libopus", 96),
    "flac": ("flac", None),
    "m4a": ("aac", 192),
}


def build_audio_convert_cmd(inp, outp, fmt, *, bitrate_kbps=None, sample_rate_hz=None, channels=None):
    if fmt not in AUDIO_CODECS:
        raise VideoError(f"Unsupported audio format '{fmt}' ({', '.join(AUDIO_CODECS)})")
    codec, default_kbps = AUDIO_CODECS[fmt]
    cmd = ["ffmpeg", "-y", "-i", inp, "-vn", "-c:a", codec]
    kbps = bitrate_kbps or default_kbps
    if kbps and default_kbps is not None:  # bitrate only makes sense for lossy codecs
        cmd += ["-b:a", f"{int(kbps)}k"]
    if sample_rate_hz:
        cmd += ["-ar", str(int(sample_rate_hz))]
    if channels:
        cmd += ["-ac", str(int(channels))]
    return cmd + [outp]


def build_audio_trim_cmd(inp, outp, start_s, end_s, *, codec):
    cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-to", str(end_s), "-i", inp, "-vn"]
    cmd += ["-c:a", "copy"] if codec is None else ["-c:a", codec]
    return cmd + [outp]


async def audio_convert(data: bytes, fmt, *, bitrate_kbps=None, sample_rate_hz=None,
                        channels=None, in_ext="mp3") -> bytes:
    src, dst = _Tmp(f".{in_ext}", data), _Tmp(f".{fmt}")
    try:
        await _run(build_audio_convert_cmd(src.path, dst.path, fmt, bitrate_kbps=bitrate_kbps,
                                           sample_rate_hz=sample_rate_hz, channels=channels))
        out = dst.read()
        if not out:
            raise VideoError("Audio conversion produced an empty file")
        return out
    finally:
        src.cleanup()
        dst.cleanup()


async def audio_trim(data: bytes, start, end, *, in_ext="mp3", out_ext=None) -> bytes:
    start_s, end_s = to_seconds(start), to_seconds(end)
    if end_s <= start_s:
        raise VideoError(f"end ({end_s}s) must be after start ({start_s}s)")
    out_ext = out_ext or in_ext
    src, dst = _Tmp(f".{in_ext}", data), _Tmp(f".{out_ext}")
    try:
        out = b""
        if out_ext == in_ext:  # same container: try lossless stream copy first
            try:
                await _run(build_audio_trim_cmd(src.path, dst.path, start_s, end_s, codec=None))
                out = dst.read()
            except VideoError:
                out = b""
        if len(out) < 256:  # copy failed or container change -> re-encode
            codec, _ = AUDIO_CODECS.get(out_ext, AUDIO_CODECS["mp3"])
            await _run(build_audio_trim_cmd(src.path, dst.path, start_s, end_s, codec=codec))
            out = dst.read()
        if len(out) < 256:
            raise VideoError("Trim produced an empty file")
        return out
    finally:
        src.cleanup()
        dst.cleanup()


# --- ffprobe -------------------------------------------------------------

async def probe(data: bytes, in_ext="bin") -> dict:
    """Raw ffprobe JSON (format + streams) for any audio/video file."""
    src = _Tmp(f".{in_ext}", data)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", src.path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            raise VideoError("ffprobe timed out after 60s")
        if proc.returncode != 0:
            tail = (stderr or b"").decode(errors="replace")[-400:]
            raise VideoError(f"ffprobe could not read this file: {tail}")
        return json.loads(stdout.decode(errors="replace") or "{}")
    finally:
        src.cleanup()


def summarize_probe(raw: dict) -> dict:
    """Curate ffprobe output down to what a chat answer needs."""
    fmt = raw.get("format", {})
    out = {
        "container": fmt.get("format_name"),
        "duration_s": round(float(fmt["duration"]), 2) if fmt.get("duration") else None,
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "streams": [],
    }
    for s in raw.get("streams", []):
        entry = {"type": s.get("codec_type"), "codec": s.get("codec_name")}
        if s.get("codec_type") == "video":
            entry["width"], entry["height"] = s.get("width"), s.get("height")
            fr = s.get("avg_frame_rate") or ""
            if "/" in fr:
                num, den = fr.split("/", 1)
                if num.isdigit() and den.isdigit() and int(den):
                    entry["fps"] = round(int(num) / int(den), 2)
        elif s.get("codec_type") == "audio":
            entry["sample_rate"] = int(s["sample_rate"]) if s.get("sample_rate") else None
            entry["channels"] = s.get("channels")
        out["streams"].append(entry)
    return out
