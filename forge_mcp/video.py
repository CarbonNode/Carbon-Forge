"""ffmpeg wrappers — trim, frame extraction, conversion. Bytes in -> bytes out via temp files."""
import asyncio
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
