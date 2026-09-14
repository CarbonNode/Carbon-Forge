"""3D → pixel sprite bake: an animated GLB in, a palette-locked, direction-tagged
sprite sheet bundle out (the Warcraft II / Diablo pre-rendered-sprite recipe).

Why this beats 2D generation for animated units: the frames come from a real
rig, so silhouettes, proportions and the ground line are consistent across
every frame and every facing by construction. Nothing "boils", nobody's feet
slide. The pixel-art look is applied AFTER the render, on ONE locked grid and
ONE locked palette shared by every frame (the same discipline as
backend.sprite_anim — per-frame quantization is what makes AI sheets flicker).

Two halves:
  run_blender(glb_bytes, spec)  — writes the GLB to a temp dir, runs
      backend/blender_bake.py in a SEPARATE Blender process (the `blender`
      binary, or a Python that has the `bpy` wheel), returns the manifest +
      the rendered PNG frames. Blender never runs inside the service process.
  compose(frames_by_row, ...)   — pure NumPy: supersampled renders → medoid-
      downsampled logical pixels → k-means palette over ALL frames → optional
      outline → ONE shared crop box (ground line stays put) → sprite sheet +
      Aseprite atlas (one frameTag per action_direction row) + GIFs.

bake(glb_bytes, ...) chains both. Deterministic given the renders.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from backend import bake_spec as bs
from backend import pixel_art as pa
from backend import sprite_anim as sa

HERE = os.path.dirname(os.path.abspath(__file__))
BAKE_SCRIPT = os.path.join(HERE, "blender_bake.py")
DEFAULT_TIMEOUT_S = 1800


class BakeError(Exception):
    """Readable, user-facing bake failure."""


# ---------------------------------------------------------------------------
# Blender process
# ---------------------------------------------------------------------------

def find_blender(blender_bin=None, bpy_python=None):
    """How to run the bake script: ('blender', path) for a Blender binary,
    ('python', path) for an interpreter that can `import bpy`, or None."""
    cand = blender_bin or os.environ.get("FORGE_BLENDER_BIN") or shutil.which("blender")
    if cand and os.path.isfile(cand):
        return ("blender", cand)
    py = bpy_python or os.environ.get("FORGE_BPY_PYTHON")
    if py and os.path.isfile(py):
        return ("python", py)
    try:
        import bpy  # noqa: F401  (the service's own interpreter has the wheel)
        return ("python", sys.executable)
    except Exception:  # noqa: BLE001
        return None


def bake_command(runner, spec_path):
    kind, exe = runner
    if kind == "blender":
        return [exe, "-b", "--python", BAKE_SCRIPT, "--", spec_path]
    return [exe, BAKE_SCRIPT, spec_path]


def parse_bake_result(stdout):
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("BAKE_RESULT "):
            return json.loads(line[len("BAKE_RESULT "):])
    return None


def run_blender_sync(glb_bytes, spec, *, runner=None, timeout_s=DEFAULT_TIMEOUT_S, keep_dir=None):
    """Render the frames. Returns (manifest, {filename: png_bytes}). Blocking."""
    runner = runner or find_blender()
    if runner is None:
        raise BakeError("No Blender available: set FORGE_BLENDER_BIN (a blender binary) or "
                        "FORGE_BPY_PYTHON (a Python with the `bpy` wheel) on the forge service")
    spec = bs.normalize_spec(spec)
    work = keep_dir or tempfile.mkdtemp(prefix="forge-bake-")
    try:
        glb_path = os.path.join(work, "model.glb")
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        out_dir = os.path.join(work, "frames")
        full = dict(spec, glb=glb_path, out_dir=out_dir)
        spec_path = os.path.join(work, "spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(full, f)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            proc = subprocess.run(bake_command(runner, spec_path), capture_output=True, text=True,
                                  timeout=timeout_s, env=env, cwd=work)
        except subprocess.TimeoutExpired as e:
            raise BakeError(f"Blender bake timed out after {timeout_s}s") from e
        result = parse_bake_result(proc.stdout)
        if proc.returncode != 0 or result is None:
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-25:])
            raise BakeError(f"Blender bake failed (exit {proc.returncode}): {tail[-1500:]}")
        with open(os.path.join(out_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        frames = {}
        for row in manifest["rows"]:
            for name in row["files"]:
                with open(os.path.join(out_dir, name), "rb") as f:
                    frames[name] = f.read()
        return manifest, frames
    finally:
        if keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)


async def run_blender(glb_bytes, spec, **kw):
    return await asyncio.to_thread(run_blender_sync, glb_bytes, spec, **kw)


# ---------------------------------------------------------------------------
# Compose: renders → pixel art bundle
# ---------------------------------------------------------------------------

def downsample(png_bytes, supersample, sampling="medoid"):
    """A render at cell*supersample px → cell px logical pixels (Oklab medoid per cell:
    crisp, no anti-aliased mush). Alpha is hardened so edges are pixel-clean."""
    rgba = sa._load_rgba(png_bytes)
    ss = max(1, int(supersample))
    if ss == 1:
        small = rgba.copy()
    else:
        h, w = rgba.shape[:2]
        grid = {"cell_w": ss, "cell_h": ss, "offset_x": 0, "offset_y": 0,
                "out_w": max(1, w // ss), "out_h": max(1, h // ss)}
        small = pa.sample_cells(rgba, grid, mode=sampling)
    a = small[..., 3]
    small[..., 3] = np.where(a >= 128, 255, 0).astype(np.uint8)
    small[small[..., 3] == 0, :3] = 0
    return small


def shared_palette(frames, max_colors=16, palette=None, palette_colors=None):
    """ONE palette for every frame of every row (list of '#rrggbb') or None."""
    if palette_colors:
        return [sa._hex(pa.parse_hex_color(c)) for c in palette_colors]
    if palette:
        spec = pa.RETRO_PALETTES.get(palette)
        if spec is None:
            raise ValueError(f"Unknown palette '{palette}'. Available: {', '.join(sorted(pa.RETRO_PALETTES))}")
        if spec.get("kmeans"):
            pal = pa.kmeans_palette(_stack(frames), spec["kmeans"])
            pal = pa.rgb555_round(np.concatenate([pal, np.full((pal.shape[0], 1), 255, np.uint8)], axis=1))[:, :3]
            return [sa._hex(c) for c in pal]
        return list(spec["colors"])
    if max_colors and int(max_colors) > 0:
        pal = pa.kmeans_palette(_stack(frames), int(max_colors))
        return [sa._hex(c) for c in pal]
    return None


def _stack(frames):
    """All frames side by side (one wide RGBA) so k-means sees the whole animation."""
    h = max(f.shape[0] for f in frames)
    w = sum(f.shape[1] for f in frames)
    out = np.zeros((h, w, 4), np.uint8)
    x = 0
    for f in frames:
        out[:f.shape[0], x:x + f.shape[1]] = f
        x += f.shape[1]
    return out


def compose(rows, *, supersample=4, max_colors=16, palette=None, palette_colors=None,
            dither="none", outline="none", outline_color="#000000", margin=0,
            fps=10, columns=0, padding=0, scale=1, name="sprite", per_row_gifs=True):
    """rows: [{"tag": "walk_s", "frames": [png_bytes, ...]}, ...] in sheet order.
    Returns the same bundle shape as sprite_anim.build_from_sheet:
    {sheet, atlas, atlas_json, gif, frames, report, row_gifs}."""
    if not rows or not any(r["frames"] for r in rows):
        raise ValueError("no frames to compose")
    small_rows = [[downsample(f, supersample) for f in r["frames"]] for r in rows]
    flat = [f for r in small_rows for f in r]
    pal_hex = shared_palette(flat, max_colors=max_colors, palette=palette, palette_colors=palette_colors)
    if pal_hex:
        pal = np.stack([pa.parse_hex_color(c) for c in pal_hex])
        flat = [pa.apply_palette(f, pal, dither=dither) for f in flat]
    if outline and outline != "none":
        oc = pa.parse_hex_color(outline_color)
        flat = [pa.add_outline(f, oc, style=outline) for f in flat]
    # ONE crop box across every action and facing: the sprite's feet land on the same
    # row in every frame, so a walk never bobs and a death collapse stays in place.
    flat, box = sa.crop_union(flat, margin=margin)
    tags, i = [], 0
    for r, sr in zip(rows, small_rows):
        n = len(sr)
        if n:
            tags.append({"name": r["tag"], "from": i, "to": i + n - 1})
            i += n
    n_total = len(flat)
    pack_cols = int(columns) if columns and int(columns) > 0 else max(len(r) for r in small_rows)
    sheet, atlas = sa.pack_sheet(flat, columns=pack_cols, padding=padding, scale=scale,
                                 name=name, fps=fps, tags=tags)
    gif = sa.gif_bytes(flat, fps=fps, scale=scale)
    row_gifs = {}
    if per_row_gifs and len(tags) > 1:
        for t in tags:
            row_gifs[t["name"]] = sa.gif_bytes(flat[t["from"]:t["to"] + 1], fps=fps, scale=scale)
    frames_out = [sa._png_bytes(pa.scale_nearest(f, scale) if scale > 1 else f) for f in flat]
    colors = set()
    for f in flat:
        c, _, _ = pa._unique_weighted_colors(f)
        colors.update(map(tuple, c.tolist()))
    report = {
        "supersample": int(supersample), "palette_colors": pal_hex,
        "palette": (f"kmeans-{max_colors} (all frames)" if pal_hex and not palette and not palette_colors
                    else palette or ("custom" if palette_colors else None)),
        "frames_out": n_total, "rows": len(tags),
        "crop_box": list(map(int, box)) if box else None,
        "frame_size": [int(flat[0].shape[1]), int(flat[0].shape[0])],
        "unique_colors": len(colors), "export_scale": max(1, min(32, int(scale))),
        "tags": tags, "outline": outline, "fps": int(fps),
    }
    return {"sheet": sheet, "atlas": atlas, "atlas_json": json.dumps(atlas, indent=1),
            "gif": gif, "frames": frames_out, "report": report, "row_gifs": row_gifs}


def rows_from_manifest(manifest, frame_files, action_prefix=None):
    """Turn a bake manifest + its PNGs into compose() rows. action_prefix renames the
    action part of every tag (e.g. Meshy's 'Armature|Casual_Walk' → 'walk')."""
    rows = []
    for row in manifest["rows"]:
        tag = row["tag"]
        if action_prefix is not None:
            tag = bs.row_tag(action_prefix, row["direction"], single_action=(action_prefix == ""))
        rows.append({"tag": tag, "action": row["action"], "direction": row["direction"],
                     "frames": [frame_files[n] for n in row["files"]]})
    return rows


def render_spec(*, cell=64, supersample=4, directions=8, frames=8, elevation_deg=35.0, actions=None,
                loop=True, engine="cycles", samples=16, clip_actions=None):
    """The Blender-side spec for one GLB. `cell` is the logical sprite size the whole
    animation's union bounds are fit into; the render is cell*supersample px."""
    return {"directions": directions, "frames": frames, "render_px": int(cell) * int(supersample),
            "elevation_deg": elevation_deg, "actions": actions, "loop": loop, "engine": engine,
            "samples": samples, "clip_actions": clip_actions or {}}


def render_rows_sync(glb_bytes, spec, *, action_prefix=None, runner=None, timeout_s=DEFAULT_TIMEOUT_S):
    """One GLB → (manifest, compose() rows). Several GLBs (one Meshy clip each) can be
    rendered this way and composed TOGETHER so they share one palette and one crop box."""
    manifest, files = run_blender_sync(glb_bytes, spec, runner=runner, timeout_s=timeout_s)
    return manifest, rows_from_manifest(manifest, files, action_prefix=action_prefix)


async def render_rows(glb_bytes, spec, **kw):
    return await asyncio.to_thread(render_rows_sync, glb_bytes, spec, **kw)


def bake_report(manifests):
    keys = ("directions", "direction_tags", "actions_available", "actions_baked", "frames_per_action",
            "loop", "engine", "elevation_deg", "render_px", "frames_rendered", "blender")
    if len(manifests) == 1:
        return {k: manifests[0][k] for k in keys if k in manifests[0]}
    return [{k: m[k] for k in keys if k in m} for m in manifests]


def bake_sync(glb_bytes, *, cell=64, supersample=4, directions=8, frames=8, elevation_deg=35.0,
              actions=None, loop=True, engine="cycles", samples=16, action_prefix=None,
              max_colors=16, palette=None, palette_colors=None, dither="none", outline="none",
              outline_color="#000000", margin=0, fps=10, columns=0, padding=0, scale=1,
              name="sprite", runner=None, timeout_s=DEFAULT_TIMEOUT_S, clip_actions=None):
    """GLB → bundle, blocking (render + compose)."""
    spec = render_spec(cell=cell, supersample=supersample, directions=directions, frames=frames,
                       elevation_deg=elevation_deg, actions=actions, loop=loop, engine=engine,
                       samples=samples, clip_actions=clip_actions)
    manifest, rows = render_rows_sync(glb_bytes, spec, action_prefix=action_prefix, runner=runner,
                                      timeout_s=timeout_s)
    res = compose(rows, supersample=supersample, max_colors=max_colors, palette=palette,
                  palette_colors=palette_colors, dither=dither, outline=outline,
                  outline_color=outline_color, margin=margin, fps=fps, columns=columns,
                  padding=padding, scale=scale, name=name)
    res["manifest"] = manifest
    res["report"]["bake"] = bake_report([manifest])
    return res


async def bake(glb_bytes, **kw):
    return await asyncio.to_thread(bake_sync, glb_bytes, **kw)
