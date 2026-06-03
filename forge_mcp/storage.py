"""Input resolution (URL / workspace path), result writing (workspace + cache), URL minting, janitor."""
import asyncio
import difflib
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from io import BytesIO

import httpx

from forge_mcp.config import Config

URL_FETCH_TIMEOUT_S = 60


class StorageError(Exception):
    """Readable, user-facing storage failure."""


@dataclass
class ResolvedInput:
    data: bytes
    mime: str
    source: str  # 'url' | 'workspace'


_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),       # checked further below
    (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"),
    (b"\x1a\x45\xdf\xa3", "video/webm"),
]


def sniff_mime(data: bytes):
    if len(data) < 12:
        return None
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    if data[4:8] == b"ftyp":  # mp4 / mov family
        brand = data[8:12]
        return "video/quicktime" if brand in (b"qt  ",) else "video/mp4"
    return None


def is_image_mime(mime):
    return bool(mime) and mime.startswith("image/")


def is_video_mime(mime):
    return bool(mime) and mime.startswith("video/")


def _workspace_abs(rel: str, cfg: Config) -> str:
    """Resolve '<Project>/<relpath>' under workspace_root, rejecting traversal."""
    norm = os.path.normpath(rel.replace("\\", "/")).replace("\\", "/")
    if norm.startswith("..") or "/../" in norm or norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise StorageError(f"Invalid workspace path (traversal not allowed): {rel}")
    abs_path = os.path.normpath(os.path.join(cfg.workspace_root, norm))
    root = os.path.normpath(cfg.workspace_root)
    if not abs_path.startswith(root + os.sep) and abs_path != root:
        raise StorageError(f"Invalid workspace path (escapes workspace): {rel}")
    return abs_path


def _read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _write_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _makedirs(path: str):
    os.makedirs(path, exist_ok=True)


async def resolve_input(ref: str, *, cfg: Config, kind: str = "image") -> ResolvedInput:
    """ref is an https URL or '<Project>/<relpath>' inside the workspace."""
    max_bytes = (cfg.max_image_mb if kind == "image" else cfg.max_video_mb) * 1024 * 1024
    if ref.startswith("http://") or ref.startswith("https://"):
        async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT_S, follow_redirects=True) as client:
            try:
                resp = await client.get(ref)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise StorageError(f"Could not fetch {ref}: {e}") from e
            data = resp.content
        source = "url"
    else:
        path = _workspace_abs(ref, cfg)
        if not os.path.isfile(path):
            raise StorageError(
                f"No file at '{ref}' in the workspace. Check the project name and path "
                f"(input paths are '<Project>/<relative path>')."
            )
        data = await asyncio.to_thread(_read_file, path)
        source = "workspace"

    if len(data) > max_bytes:
        raise StorageError(f"Input too large: {len(data) // (1024 * 1024)} MB (limit {max_bytes // (1024 * 1024)} MB)")
    mime = sniff_mime(data)
    if kind == "image" and not is_image_mime(mime):
        raise StorageError(f"Input from '{ref}' is not a recognized image (png/jpg/webp/gif/bmp)")
    if kind == "video" and not is_video_mime(mime):
        raise StorageError(f"Input from '{ref}' is not a recognized video (mp4/webm/mov)")
    return ResolvedInput(data=data, mime=mime, source=source)


def validate_project(project: str, *, cfg: Config) -> str:
    if not project or "/" in project or "\\" in project or project.startswith("."):
        raise StorageError(f"Invalid project name: {project!r}")
    path = os.path.join(cfg.workspace_root, project)
    if os.path.isdir(path):
        return project
    try:
        existing = [d for d in os.listdir(cfg.workspace_root)
                    if os.path.isdir(os.path.join(cfg.workspace_root, d)) and not d.startswith(".")]
    except OSError as e:
        raise StorageError(f"Workspace not reachable ({e}); cannot validate project '{project}'") from e
    close = difflib.get_close_matches(project, existing, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise StorageError(f"No workspace project named '{project}'.{hint} (Use list_projects to see all.)")


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name).strip(" .")
    return name[:120] or "output"


def _unique_path(directory: str, base: str, ext: str) -> str:
    candidate = os.path.join(directory, f"{base}.{ext}")
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}-{n}.{ext}")
        n += 1
    return candidate


def _image_dims(data: bytes):
    try:
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            return im.width, im.height
    except Exception:
        return None, None


async def save_result(data: bytes, *, project, subpath, filename, ext, cfg: Config) -> dict:
    """Write to results cache (always) + workspace (best effort). Returns the tool result dict."""
    base = safe_filename(filename or "output")
    file_id = secrets.token_urlsafe(18)
    cache_dir = os.path.join(cfg.results_root, "files", file_id)
    await asyncio.to_thread(_makedirs, cache_dir)
    cache_name = f"{base}.{ext}"
    await asyncio.to_thread(_write_file, os.path.join(cache_dir, cache_name), data)

    result = {
        "url": f"{cfg.public_url}/files/{file_id}/{cache_name}",
        "bytes": len(data),
    }
    if ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp"):
        w, h = _image_dims(data)
        if w:
            result["width"], result["height"] = w, h

    if project:
        sub = (subpath or "assets/forge").strip("/\\")
        try:
            validate_project(project, cfg=cfg)
            rel_dir = "/".join([project] + sub.replace("\\", "/").split("/"))
            target_dir = _workspace_abs(rel_dir, cfg)
            await asyncio.to_thread(_makedirs, target_dir)
            target = await asyncio.to_thread(_unique_path, target_dir, base, ext)
            await asyncio.to_thread(_write_file, target, data)
            # Display path is always Windows-style (the workspace lives on carbonserver),
            # regardless of the container OS this runs on.
            rel_to_root = os.path.relpath(target, cfg.workspace_root).replace("/", "\\")
            result["workspace_path"] = cfg.workspace_display_root.rstrip("\\") + "\\" + rel_to_root
        except (StorageError, OSError) as e:
            result["workspace_write_error"] = (
                f"Result NOT written to the workspace ({e}). It is still available at the url above."
            )
    return result


def prune_cache_once(cfg: Config):
    """Delete /results/files/<id> dirs older than the TTL."""
    files_root = os.path.join(cfg.results_root, "files")
    if not os.path.isdir(files_root):
        return
    cutoff = time.time() - cfg.cache_ttl_days * 86400
    for entry in os.listdir(files_root):
        path = os.path.join(files_root, entry)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


async def janitor_loop(cfg: Config):
    while True:
        await asyncio.to_thread(prune_cache_once, cfg)
        await asyncio.sleep(24 * 3600)
