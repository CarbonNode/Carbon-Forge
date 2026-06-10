"""GLB (binary glTF) helpers — Draco mesh compression via gltf-transform, header/JSON stats."""
import asyncio
import json
import shutil
import struct

from forge_mcp.video import _Tmp

GLTF_TIMEOUT_S = 300


class ModelError(Exception):
    """Readable, user-facing 3D-asset failure."""


def glb_stats(data: bytes) -> dict:
    """Parse the GLB header + JSON chunk (pure python, no node needed)."""
    if len(data) < 20 or data[:4] != b"glTF":
        raise ModelError("Not a binary glTF (.glb) file")
    version, _total = struct.unpack_from("<II", data, 4)
    chunk_len, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON" or 20 + chunk_len > len(data):
        raise ModelError("Malformed GLB (first chunk is not valid JSON)")
    try:
        doc = json.loads(data[20:20 + chunk_len].decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise ModelError(f"Malformed GLB JSON chunk: {e}") from e
    extensions = doc.get("extensionsUsed", [])
    return {
        "gltf_version": version,
        "bytes": len(data),
        "meshes": len(doc.get("meshes", [])),
        "primitives": sum(len(m.get("primitives", [])) for m in doc.get("meshes", [])),
        "materials": len(doc.get("materials", [])),
        "textures": len(doc.get("textures", [])),
        "images": len(doc.get("images", [])),
        "animations": len(doc.get("animations", [])),
        "nodes": len(doc.get("nodes", [])),
        "extensions_used": extensions,
        "draco_compressed": "KHR_draco_mesh_compression" in extensions,
    }


async def draco_compress(data: bytes, *, quantize_position=None, quantize_normal=None,
                         quantize_texcoord=None) -> bytes:
    if shutil.which("gltf-transform") is None:
        raise ModelError("gltf-transform is not installed in this image (image rebuild needed)")
    src, dst = _Tmp(".glb", data), _Tmp(".glb")
    cmd = ["gltf-transform", "draco", src.path, dst.path]
    if quantize_position:
        cmd += ["--quantize-position", str(int(quantize_position))]
    if quantize_normal:
        cmd += ["--quantize-normal", str(int(quantize_normal))]
    if quantize_texcoord:
        cmd += ["--quantize-texcoord", str(int(quantize_texcoord))]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=GLTF_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            raise ModelError(f"gltf-transform timed out after {GLTF_TIMEOUT_S}s")
        if proc.returncode != 0:
            tail = (stderr or b"").decode(errors="replace")[-800:]
            raise ModelError(f"gltf-transform failed (exit {proc.returncode}): {tail}")
        out = dst.read()
        if len(out) < 20 or out[:4] != b"glTF":
            raise ModelError("Draco compression produced an invalid GLB")
        return out
    finally:
        src.cleanup()
        dst.cleanup()
