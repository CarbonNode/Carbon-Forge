"""Sprite animation engine — turn a VIDEO of a pixel sprite into a game-ready
animation (sprite sheet + GIF + frame atlas JSON).

The "pixel animation is solved" recipe: draw ONE sprite, let a video model
(Wan 2.2 I2V locally, or any hosted model) move it, then bring the frames
back to true pixel art. The hard part is not the video — it is making the
frames belong to the SAME sprite afterwards. AI video output is blurry,
color-shifted and softly anti-aliased, so refining each frame on its own
gives every frame its own pixel grid, its own palette and its own bounding
box: the animation "boils". This module locks all three across frames:

  1. lock_style      — one pixel grid (cell size) + one palette, taken from
                       the SOURCE sprite (or the first frame when no source)
  2. refine_frames   — every frame is resampled on THAT grid (Oklab medoid),
                       background-keyed at the low resolution (clean edges
                       even after video compression), then snapped to THAT
                       palette — no frame invents a color
  3. crop_union      — one bounding box across all frames so nothing jitters
  4. dedupe_frames   — drop consecutive identical frames (held poses)
  5. pack_sheet      — grid sprite sheet + Aseprite/Phaser-compatible atlas
                       JSON + a looping GIF preview

Pure functions, bytes in / bytes out, no AI models, deterministic — same
contract as backend.pixel_art (which this builds on). Shared by the hosted
MCP tools (forge_mcp/tools/sprite.py); the desktop app can import it too.
"""
import io
import json
import math

import numpy as np
from PIL import Image

from backend import pixel_art as pa

# Video compression smears flat colors — keying at low-res needs more slack
# than the 24 used for clean AI stills.
DEFAULT_KEY_TOLERANCE = 48
# Default chroma key when a transparent sprite must be composited before I2V.
DEFAULT_KEY_COLOR = "#FF00FF"
DEFAULT_MAX_COLORS = 16
# Sheets are clean stills (no compression fringe) — the pixel_refine default.
SHEET_KEY_TOLERANCE = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rgba(data):
    """PNG/JPG bytes (or an array) -> uint8 RGBA array."""
    if isinstance(data, np.ndarray):
        return np.asarray(data, dtype=np.uint8)
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def _png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _hex(rgb):
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def frame_timestamps(duration_s, count, start_s=0.0, end_s=None, loop=True):
    """Evenly spaced sample times inside [start, end).

    loop=True leaves the last sample one step short of `end` so the final
    frame does not duplicate the first when the clip is played on repeat
    (the way a walk cycle is sampled). loop=False includes both endpoints."""
    count = max(1, int(count))
    if end_s is None or end_s <= 0 or end_s > duration_s:
        end_s = float(duration_s)
    start_s = max(0.0, float(start_s))
    if end_s <= start_s:
        raise ValueError(f"end ({end_s}s) must be after start ({start_s}s)")
    span = end_s - start_s
    if count == 1:
        return [round(start_s, 4)]
    step = span / count if loop else span / (count - 1)
    times = [start_s + i * step for i in range(count)]
    # ffmpeg's -ss on the very last instant can land past the final frame;
    # keep a hair of headroom.
    times = [min(t, max(start_s, end_s - 0.02)) for t in times]
    return [round(t, 4) for t in times]


def detect_border_color(data):
    """Dominant color along the four edges of an image, as '#rrggbb' — the
    background a video model kept around the sprite (key_color='auto')."""
    rgba = _load_rgba(data)
    edges = np.concatenate([rgba[0, :, :3], rgba[-1, :, :3],
                            rgba[:, 0, :3], rgba[:, -1, :3]]).astype(np.int64)
    keys = (edges[:, 0] << 16) | (edges[:, 1] << 8) | edges[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    k = int(uniq[np.argmax(counts)])
    return f"#{(k >> 16) & 0xFF:02x}{(k >> 8) & 0xFF:02x}{k & 0xFF:02x}"


def prepare_reference(data, size, key_color=DEFAULT_KEY_COLOR):
    """Make the opaque start frame a video model should animate: flatten the
    sprite onto the chroma key, nearest-neighbor upscale by the largest
    integer that fits `size` (crisp blocks, no resampling blur), and pad to
    exactly `size` with the key so the model never stretches the sprite.
    Returns (png_bytes, info)."""
    rgba = _load_rgba(data)
    W, H = int(size[0]), int(size[1])
    h, w = rgba.shape[:2]
    factor = max(1, min(W // max(1, w), H // max(1, h)))
    big = pa.scale_nearest(rgba, factor) if factor > 1 else rgba
    bh, bw = big.shape[:2]
    if bw > W or bh > H:  # source larger than the frame: fit it with LANCZOS
        img = Image.fromarray(big, "RGBA")
        img.thumbnail((W, H), Image.LANCZOS)
        big = np.array(img, dtype=np.uint8)
        bh, bw = big.shape[:2]
    key = np.asarray(pa.parse_hex_color(key_color), dtype=np.uint8)
    canvas = np.empty((H, W, 4), dtype=np.uint8)
    canvas[..., :3] = key
    canvas[..., 3] = 255
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    a = big[..., 3:4].astype(np.float64) / 255.0
    region = canvas[y0:y0 + bh, x0:x0 + bw, :3].astype(np.float64)
    region = big[..., :3].astype(np.float64) * a + region * (1.0 - a)
    canvas[y0:y0 + bh, x0:x0 + bw, :3] = np.clip(np.round(region), 0, 255).astype(np.uint8)
    info = {"source_size": [w, h], "upscale": factor, "placed": [x0, y0, bw, bh],
            "frame_size": [W, H], "key_color": key_color}
    return _png_bytes(canvas), info


def prepare_native_frame(data, min_size, max_size, pad_to_multiple=1):
    """The exact frame an API that animates YOUR sprite should receive: the
    sprite at its logical size (a true pixel grid is collapsed to 1 px per
    logical pixel), integer-upscaled (nearest) if smaller than `min_size`,
    LANCZOS-fitted only if larger than `max_size`, on a transparent canvas
    whose sides are at least min_size. Returns (png_bytes, (w, h), info)."""
    rgba = _load_rgba(data)
    info = {"source_size": [int(rgba.shape[1]), int(rgba.shape[0])]}
    _, g, small, grep = pa.resolve_grid(rgba)
    if g["detected"] and (g["cell_w"] > 1 or g["cell_h"] > 1):
        rgba = small
        info["grid"] = {"cell": [int(g["cell_w"]), int(g["cell_h"])],
                        "logical_size": [int(rgba.shape[1]), int(rgba.shape[0])]}
        if grep.get("fractional"):
            info["grid"]["fractional"] = grep["fractional"]
    rgba, box = crop_union([rgba])
    rgba = rgba[0]
    h, w = rgba.shape[:2]
    factor = 1
    while max(w, h) * (factor + 1) <= max_size and min(w, h) * factor < min_size:
        factor += 1
    if factor > 1:
        rgba = pa.scale_nearest(rgba, factor)
        h, w = rgba.shape[:2]
    if max(w, h) > max_size:
        img = Image.fromarray(rgba, "RGBA")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        rgba = np.array(img, dtype=np.uint8)
        h, w = rgba.shape[:2]
        info["fitted"] = True
    W, H = max(w, min_size), max(h, min_size)
    m = max(1, int(pad_to_multiple))
    W, H = -(-W // m) * m, -(-H // m) * m
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    x0, y0 = (W - w) // 2, H - h  # feet on the floor, centered
    canvas[y0:y0 + h, x0:x0 + w] = rgba
    info.update({"upscale": factor, "placed": [int(x0), int(y0), int(w), int(h)],
                 "frame_size": [int(W), int(H)]})
    return _png_bytes(canvas), (int(W), int(H)), info


def composite_on_key(data, key_color=DEFAULT_KEY_COLOR):
    """Flatten a transparent sprite onto a solid chroma key so a video model
    gets an opaque start frame that can be keyed back out. Returns (png_bytes,
    had_alpha)."""
    rgba = _load_rgba(data)
    had_alpha = bool((rgba[..., 3] < 255).any())
    if not had_alpha:
        return _png_bytes(rgba), False
    key = np.asarray(pa.parse_hex_color(key_color), dtype=np.float64)
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    rgb = rgba[..., :3].astype(np.float64) * a + key * (1.0 - a)
    out = np.empty_like(rgba)
    out[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return _png_bytes(out), True


# ---------------------------------------------------------------------------
# 1. Style lock — one grid + one palette for the whole animation
# ---------------------------------------------------------------------------

def lock_style(reference, frame_size, *, cell_size=0, max_colors=DEFAULT_MAX_COLORS,
               palette=None, palette_colors=None, key_color=None,
               key_tolerance=DEFAULT_KEY_TOLERANCE):
    """Derive the animation's fixed pixel grid + palette.

    reference: PNG bytes of the source sprite (the still that was animated) or
      the first video frame when there is no source.
    frame_size: (w, h) of the video frames. The reference's logical width is
      carried over: cell = frame_w / logical_w, so frames land on the same
      number of logical pixels as the sprite even though the video model
      resized the image.
    cell_size > 0 forces the frame cell size (skips detection).
    Palette priority: palette_colors (custom hex list) > palette (retro name)
      > k-means of the refined reference (max_colors; 0 = no palette lock).

    Returns {cell_size, logical_size, palette (list of hex) or None, report}.
    """
    ref = _load_rgba(reference)
    fw, fh = int(frame_size[0]), int(frame_size[1])
    report = {"reference_size": [int(ref.shape[1]), int(ref.shape[0])]}

    if key_color:
        ref = pa.key_background(ref, color=pa.parse_hex_color(key_color),
                                tolerance=key_tolerance)

    if cell_size and int(cell_size) > 1:
        cs = int(cell_size)
        report["grid"] = {"mode": "manual", "cell": cs}
        logical_w, logical_h = max(1, fw // cs), max(1, fh // cs)
        small_ref = None
    else:
        _, g, small_ref, grep = pa.resolve_grid(ref)
        report["grid"] = {"mode": "auto", **{k: grep.get(k) for k in
                          ("cell_w", "cell_h", "detected", "score", "reconstruction_error",
                           "fractional", "harmonic_rescue", "rejected") if grep.get(k) is not None}}
        if g["detected"] and (g["cell_w"] > 1 or g["cell_h"] > 1):
            logical_w, logical_h = int(small_ref.shape[1]), int(small_ref.shape[0])
        else:
            # Not real pixel art (vector/flat) — treat every source pixel
            # as logical so the frames are simply downsampled to it.
            small_ref = ref.copy()
            logical_w, logical_h = ref.shape[1], ref.shape[0]
        # The video model resized the reference to the frame size; map the
        # reference's logical grid onto the frame. Use the axis that scaled
        # least so the sprite never gets MORE logical pixels than it had.
        cs = max(1, int(round(min(fw / max(1, logical_w), fh / max(1, logical_h)))))
        logical_w, logical_h = max(1, fw // cs), max(1, fh // cs)
        report["grid"]["cell"] = cs

    pal_hex = None
    if palette_colors:
        pal_hex = [_hex(pa.parse_hex_color(c)) for c in palette_colors]
        report["palette"] = "custom"
    elif palette:
        spec = pa.RETRO_PALETTES.get(palette)
        if spec is None:
            raise ValueError(f"Unknown palette '{palette}'. Available: "
                             f"{', '.join(sorted(pa.RETRO_PALETTES))}")
        if spec.get("kmeans"):
            # SFC-style: k-means over the reference, rounded to RGB555
            src = small_ref if small_ref is not None else ref
            pal = pa.kmeans_palette(src, spec["kmeans"])
            pal = pa.rgb555_round(np.concatenate(
                [pal, np.full((pal.shape[0], 1), 255, np.uint8)], axis=1))[:, :3]
            pal_hex = [_hex(c) for c in pal]
        else:
            pal_hex = list(spec["colors"])
        report["palette"] = palette
    elif max_colors and int(max_colors) > 0:
        src = small_ref if small_ref is not None else ref
        pal = pa.kmeans_palette(src, int(max_colors))
        pal_hex = [_hex(c) for c in pal]
        report["palette"] = f"kmeans-{int(max_colors)} (from reference)"
    else:
        report["palette"] = None

    return {"cell_size": cs, "logical_size": [int(logical_w), int(logical_h)],
            "palette": pal_hex, "report": report}


# ---------------------------------------------------------------------------
# 2. Per-frame refinement on the locked grid + palette
# ---------------------------------------------------------------------------

def refine_frame(frame, cell_size, *, palette=None, sampling="medoid",
                 key_color=None, key_tolerance=DEFAULT_KEY_TOLERANCE,
                 dither="none", dither_strength=1.0, hard_alpha=True):
    """One video frame -> logical-resolution RGBA array on a FIXED grid.

    Background keying happens AFTER cell sampling: the medoid already threw
    away the compression fringe inside each cell, so a plain tolerance key at
    logical resolution gives clean silhouettes; keying the blurry full-res
    frame first leaves a halo of half-keyed edge cells."""
    rgba = _load_rgba(frame)
    h, w = rgba.shape[:2]
    cs = max(1, int(cell_size))
    grid = {"cell_w": cs, "cell_h": cs, "offset_x": 0, "offset_y": 0,
            "out_w": max(1, w // cs), "out_h": max(1, h // cs)}
    small = pa.sample_cells(rgba, grid, mode=sampling)
    if key_color:
        small = pa.key_background(small, color=pa.parse_hex_color(key_color),
                                  tolerance=key_tolerance)
    if hard_alpha:
        a = small[..., 3]
        small[..., 3] = np.where(a >= 128, 255, 0).astype(np.uint8)
    if palette:
        pal = np.stack([pa.parse_hex_color(c) for c in palette])
        small = pa.apply_palette(small, pal, dither=dither, strength=dither_strength)
    return small


def refine_frames(frames, cell_size, **kwargs):
    return [refine_frame(f, cell_size, **kwargs) for f in frames]


# ---------------------------------------------------------------------------
# 3/4. Alignment + dedupe
# ---------------------------------------------------------------------------

def union_bbox(frames, threshold=16, margin=0):
    """Bounding box (x0, y0, x1, y1) covering the opaque pixels of ALL frames.
    Cropping every frame to it keeps the sprite's position stable across the
    animation (per-frame trimming would make it jump)."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for f in frames:
        fh, fw = f.shape[:2]
        mask[:min(h, fh), :min(w, fw)] |= f[:min(h, fh), :min(w, fw), 3] >= threshold
    if not mask.any():
        return (0, 0, w, h)
    ys, xs = np.nonzero(mask)
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(w, xs.max() + 1 + margin), min(h, ys.max() + 1 + margin))


def crop_union(frames, threshold=16, margin=0):
    box = union_bbox(frames, threshold, margin)
    if box is None:
        return [], None
    x0, y0, x1, y1 = box
    return [f[y0:y1, x0:x1].copy() for f in frames], box


def dedupe_frames(frames, times=None, max_diff_px=0):
    """Drop a frame when it is (near-)identical to the one kept before it.
    max_diff_px: how many logical pixels may differ and still count as the
    same pose. Returns (frames, kept_indices)."""
    kept, idx = [], []
    for i, f in enumerate(frames):
        if kept:
            prev = kept[-1]
            if prev.shape == f.shape:
                diff = (prev != f).any(axis=-1).sum()
                if diff <= max_diff_px:
                    continue
        kept.append(f)
        idx.append(i)
    return kept, idx


# ---------------------------------------------------------------------------
# 5. Sheet packing + atlas + GIF
# ---------------------------------------------------------------------------

def _layout(n, columns):
    if columns and columns > 0:
        cols = max(1, min(n, int(columns)))
    else:
        cols = n  # one horizontal strip — the format every engine imports
    rows = math.ceil(n / cols)
    return cols, rows


def _frame_tags(tags, n, name):
    """Aseprite frameTags: a plain string tags the whole animation; a dict
    {name, from, to[, direction]} tags a frame range (e.g. one row of a
    4-direction sheet). Ranges are clamped to the packed frames."""
    out = []
    for t in (tags or [name]):
        if isinstance(t, dict):
            a = max(0, min(n - 1, int(t.get("from", 0))))
            b = max(a, min(n - 1, int(t.get("to", n - 1))))
            out.append({"name": str(t.get("name") or name), "from": a, "to": b,
                        "direction": t.get("direction", "forward")})
        else:
            out.append({"name": str(t), "from": 0, "to": n - 1, "direction": "forward"})
    return out


def pack_sheet(frames, *, columns=0, padding=0, scale=1, name="sprite", fps=12,
               tags=None, source_times=None):
    """Pack same-size RGBA frames into a sheet.

    Returns (sheet_png_bytes, atlas_dict). The atlas follows Aseprite's JSON
    "array" export (frames[], meta.frameTags) — Phaser (`this.load.aseprite`),
    Godot importers, Unity/Unreal plugins and most engines read it directly.
    scale: integer nearest-neighbor export scale (frame size in the atlas is
    the EXPORTED size so the coordinates match the PNG)."""
    if not frames:
        raise ValueError("no frames to pack")
    fh, fw = frames[0].shape[:2]
    for f in frames:
        if f.shape[:2] != (fh, fw):
            raise ValueError("all frames must share one size (run crop_union first)")
    scale = max(1, min(32, int(scale)))
    padding = max(0, int(padding))
    n = len(frames)
    cols, rows = _layout(n, columns)
    ew, eh = fw * scale, fh * scale
    sheet_w = cols * ew + (cols - 1) * padding
    sheet_h = rows * eh + (rows - 1) * padding
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    dur_ms = int(round(1000.0 / max(1, fps)))
    atlas_frames = []
    for i, f in enumerate(frames):
        c, r = i % cols, i // cols
        x, y = c * (ew + padding), r * (eh + padding)
        big = pa.scale_nearest(f, scale) if scale > 1 else f
        sheet[y:y + eh, x:x + ew] = big
        entry = {
            "filename": f"{name} {i}.png",
            "frame": {"x": int(x), "y": int(y), "w": int(ew), "h": int(eh)},
            "rotated": False, "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": int(ew), "h": int(eh)},
            "sourceSize": {"w": int(ew), "h": int(eh)},
            "duration": dur_ms,
        }
        if source_times is not None and i < len(source_times):
            entry["source_time_s"] = source_times[i]
        atlas_frames.append(entry)
    atlas = {
        "frames": atlas_frames,
        "meta": {
            "app": "carbon-forge", "version": "1",
            "image": f"{name}.png", "format": "RGBA8888",
            "size": {"w": int(sheet_w), "h": int(sheet_h)},
            "scale": str(scale),
            "frameTags": _frame_tags(tags, n, name),
            "layout": {"columns": cols, "rows": rows, "frame_w": int(ew),
                       "frame_h": int(eh), "padding": padding, "count": n,
                       "fps": int(fps), "logical_frame": [int(fw), int(fh)]},
        },
    }
    return _png_bytes(sheet), atlas


def gif_bytes(frames, fps=12, scale=1, loop=0):
    """Looping GIF preview of the frames (nearest-neighbor scaled). Transparent
    pixels stay transparent; every frame shares one palette so it does not
    flicker."""
    if not frames:
        raise ValueError("no frames")
    scale = max(1, min(32, int(scale)))
    dur = int(round(1000.0 / max(1, fps)))
    imgs = []
    for f in frames:
        big = pa.scale_nearest(f, scale) if scale > 1 else f
        imgs.append(Image.fromarray(np.asarray(big, dtype=np.uint8), "RGBA"))
    # One shared palette: quantize a strip of all frames, then remap each.
    strip = Image.new("RGBA", (imgs[0].width * len(imgs), imgs[0].height))
    for i, im in enumerate(imgs):
        strip.paste(im, (i * imgs[0].width, 0))
    alpha_mask = np.array(strip)[..., 3] < 128
    pal_img = strip.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                            dither=Image.Dither.NONE)
    palette = pal_img.getpalette()[:255 * 3] + [0, 0, 0]  # index 255 = transparent
    out = []
    w = imgs[0].width
    for i, im in enumerate(imgs):
        q = im.convert("RGB").quantize(palette=pal_img, dither=Image.Dither.NONE)
        arr = np.array(q, dtype=np.uint8)
        arr[alpha_mask[:, i * w:(i + 1) * w]] = 255
        p = Image.fromarray(arr, "P")
        p.putpalette(palette)
        out.append(p)
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:], duration=dur,
                loop=loop, transparency=255, disposal=2, optimize=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Top-level: frames (PNG bytes) -> sheet bundle
# ---------------------------------------------------------------------------

def build_animation(frame_pngs, *, reference=None, cell_size=0, max_colors=DEFAULT_MAX_COLORS,
                    palette=None, palette_colors=None, key_color=None,
                    key_tolerance=DEFAULT_KEY_TOLERANCE, sampling="medoid",
                    dither="none", dither_strength=1.0, outline="none",
                    outline_color="#000000", dedupe=True, margin=0, columns=0,
                    padding=0, scale=1, fps=12, name="sprite", source_times=None):
    """Full pipeline: video frames -> {sheet, gif, atlas, frames, report}.

    reference: PNG bytes of the source sprite the video was made from (best —
      locks grid + palette to the real sprite); None = lock to the first frame.
    key_color: '#rrggbb' background to key out of every frame (and of the
      reference, which is harmless when it has none), or 'auto' to sample the
      first frame's border; None = keep the frames opaque.
    All per-frame outputs are logical-resolution RGBA arrays; `scale` is only
    applied to the exported sheet/GIF/frames.
    """
    if not frame_pngs:
        raise ValueError("no frames")
    first = _load_rgba(frame_pngs[0])
    fh, fw = first.shape[:2]
    if key_color == "auto":
        key_color = detect_border_color(first)
    style = lock_style(reference if reference is not None else frame_pngs[0], (fw, fh),
                       cell_size=cell_size, max_colors=max_colors, palette=palette,
                       palette_colors=palette_colors,
                       key_color=key_color, key_tolerance=key_tolerance)
    cs, pal = style["cell_size"], style["palette"]
    small = refine_frames(frame_pngs, cs, palette=pal, sampling=sampling, key_color=key_color,
                          key_tolerance=key_tolerance, dither=dither,
                          dither_strength=dither_strength)
    if outline and outline != "none":
        oc = pa.parse_hex_color(outline_color)
        small = [pa.add_outline(f, oc, style=outline) for f in small]
    kept_idx = list(range(len(small)))
    if dedupe:
        small, kept_idx = dedupe_frames(small)
    small, box = crop_union(small, margin=margin)
    times = [source_times[i] for i in kept_idx] if source_times else None
    sheet, atlas = pack_sheet(small, columns=columns, padding=padding, scale=scale,
                              name=name, fps=fps, source_times=times)
    gif = gif_bytes(small, fps=fps, scale=scale)
    frames_out = [_png_bytes(pa.scale_nearest(f, scale) if scale > 1 else f) for f in small]
    colors = set()
    for f in small:
        c, _, _ = pa._unique_weighted_colors(f)
        colors.update(map(tuple, c.tolist()))
    report = {
        **style["report"],
        "key_color": key_color,
        "cell_size": cs,
        "palette_colors": pal,
        "frames_in": len(frame_pngs),
        "frames_out": len(small),
        "dropped_duplicates": len(frame_pngs) - len(small),
        "kept_frame_indices": kept_idx,
        "crop_box": list(map(int, box)) if box else None,
        "frame_size": [int(small[0].shape[1]), int(small[0].shape[0])],
        "unique_colors": len(colors),
        "export_scale": scale,
    }
    return {"sheet": sheet, "atlas": atlas, "atlas_json": json.dumps(atlas, indent=1),
            "gif": gif, "frames": frames_out, "report": report}


def pack_existing(frame_pngs, *, dedupe=False, columns=0, padding=0, scale=1, fps=12,
                  name="sprite"):
    """Frames that are ALREADY pixel art (split_sprites output, hand-drawn
    frames) -> the same bundle as build_animation, without refining. Frames of
    different sizes are placed on one shared canvas (centered on the bottom
    edge — the way a character stands on the ground) and cropped to one box."""
    if not frame_pngs:
        raise ValueError("no frames")
    arrs = [_load_rgba(f) for f in frame_pngs]
    H = max(a.shape[0] for a in arrs)
    W = max(a.shape[1] for a in arrs)
    placed = []
    for a in arrs:
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        h, w = a.shape[:2]
        x0, y0 = (W - w) // 2, H - h
        canvas[y0:y0 + h, x0:x0 + w] = a
        placed.append(canvas)
    kept_idx = list(range(len(placed)))
    if dedupe:
        placed, kept_idx = dedupe_frames(placed)
    placed, box = crop_union(placed)
    sheet, atlas = pack_sheet(placed, columns=columns, padding=padding, scale=scale, name=name, fps=fps)
    gif = gif_bytes(placed, fps=fps, scale=scale)
    frames_out = [_png_bytes(pa.scale_nearest(f, scale) if scale > 1 else f) for f in placed]
    report = {
        "frames_in": len(frame_pngs), "frames_out": len(placed),
        "dropped_duplicates": len(frame_pngs) - len(placed), "kept_frame_indices": kept_idx,
        "crop_box": list(map(int, box)) if box else None,
        "frame_size": [int(placed[0].shape[1]), int(placed[0].shape[0])],
        "export_scale": max(1, min(32, int(scale))), "refined": False,
    }
    return {"sheet": sheet, "atlas": atlas, "atlas_json": json.dumps(atlas, indent=1),
            "gif": gif, "frames": frames_out, "report": report}


# ---------------------------------------------------------------------------
# Sprite SHEETS as input — Retro Diffusion (Astropulse), PixelLab, Aseprite
# exports, any fixed-cell grid
# ---------------------------------------------------------------------------

# Retro Diffusion's animation presets return a fixed-cell grid whose ROWS are
# the four facings, in this order (verified on live output 2026-09-08).
RD_DIRECTIONS = ("down", "right", "up", "left")


def slice_sheet(sheet_png, frame_w, frame_h, *, columns=0, rows=0, drop_empty_tail=True):
    """Cut a fixed-cell sprite sheet into frames, row-major.

    frame_w/frame_h: the cell size in sheet pixels. columns/rows: 0 = as many
    as fit (sheet size // cell). Trailing cells with no opaque pixel are
    dropped (a partial last row); interior empty cells are kept so the grid
    indices stay meaningful. Returns (frames, columns, rows_used)."""
    sheet = _load_rgba(sheet_png)
    H, W = sheet.shape[:2]
    fw, fh = max(1, int(frame_w)), max(1, int(frame_h))
    cols = int(columns) if columns and int(columns) > 0 else max(1, W // fw)
    nrows = int(rows) if rows and int(rows) > 0 else max(1, H // fh)
    if fw > W or fh > H:
        raise ValueError(f"frame {fw}x{fh} is larger than the sheet {W}x{H}")
    frames = []
    for r in range(nrows):
        for c in range(cols):
            y0, x0 = r * fh, c * fw
            cell = np.zeros((fh, fw, 4), dtype=np.uint8)
            src = sheet[y0:min(H, y0 + fh), x0:min(W, x0 + fw)]
            cell[:src.shape[0], :src.shape[1]] = src
            frames.append(cell)
    if drop_empty_tail:
        while len(frames) > 1 and not (frames[-1][..., 3] >= 16).any():
            frames.pop()
    rows_used = math.ceil(len(frames) / cols)
    return frames, cols, rows_used


def build_from_sheet(sheet_png, frame_w, frame_h, *, columns=0, rows=0, row_tags=None,
                     reference=None, max_colors=0, palette=None, palette_colors=None,
                     key_color=None, key_tolerance=SHEET_KEY_TOLERANCE,
                     outline="none", outline_color="#000000", margin=0, out_columns=0,
                     padding=0, scale=1, fps=8, name="sprite", per_row_gifs=True):
    """A sprite SHEET that is already pixel art -> the same bundle as
    build_animation, with one Aseprite frameTag per ROW (row_tags, e.g.
    RD_DIRECTIONS) so engines can play "walk-down" / "walk-left" directly.

    Frames are not resampled (cell size 1) — the sheet's pixels ARE the
    logical pixels. Optional: key_color keys an opaque background out;
    palette / palette_colors / max_colors>0 snap every frame to ONE palette
    (max_colors is k-means over `reference` — the source sprite the sheet was
    generated from — or over the first frame). Frames are cropped to ONE box
    shared by every frame of every row, so all directions align in the atlas.
    out_columns: 0 keeps the source column count (rows stay rows); N repacks.
    Returns {sheet, atlas, atlas_json, gif, frames, report, row_gifs}."""
    frames, cols, nrows = slice_sheet(sheet_png, frame_w, frame_h, columns=columns, rows=rows)
    report = {"source_layout": {"columns": cols, "rows": nrows, "frame_w": int(frame_w),
                                "frame_h": int(frame_h), "count": len(frames)}}
    pal_hex = None
    if key_color == "auto":
        key_color = detect_border_color(_png_bytes(frames[0]))
    if palette_colors or palette or (max_colors and int(max_colors) > 0):
        ref_png = reference if reference is not None else _png_bytes(frames[0])
        style = lock_style(ref_png, (frame_w, frame_h), cell_size=1, max_colors=max_colors,
                           palette=palette, palette_colors=palette_colors,
                           key_color=key_color, key_tolerance=key_tolerance)
        pal_hex = style["palette"]
        report["palette"] = style["report"].get("palette")
    else:
        report["palette"] = None
    small = [refine_frame(_png_bytes(f), 1, palette=pal_hex, key_color=key_color,
                          key_tolerance=key_tolerance) for f in frames]
    if outline and outline != "none":
        oc = pa.parse_hex_color(outline_color)
        small = [pa.add_outline(f, oc, style=outline) for f in small]
    small, box = crop_union(small, margin=margin)
    n = len(small)
    tags = []
    names = list(row_tags or [])
    for r in range(nrows):
        a, b = r * cols, min(n - 1, r * cols + cols - 1)
        if a > b:
            break
        tags.append({"name": names[r] if r < len(names) else f"row{r}", "from": a, "to": b})
    if nrows == 1 and not names:
        tags = [name]
    pack_cols = int(out_columns) if out_columns and int(out_columns) > 0 else cols
    sheet, atlas = pack_sheet(small, columns=pack_cols, padding=padding, scale=scale,
                              name=name, fps=fps, tags=tags)
    gif = gif_bytes(small, fps=fps, scale=scale)
    row_gifs = {}
    if per_row_gifs and nrows > 1:
        for t in tags:
            if isinstance(t, dict):
                row_gifs[t["name"]] = gif_bytes(small[t["from"]:t["to"] + 1], fps=fps, scale=scale)
    frames_out = [_png_bytes(pa.scale_nearest(f, scale) if scale > 1 else f) for f in small]
    colors = set()
    for f in small:
        c, _, _ = pa._unique_weighted_colors(f)
        colors.update(map(tuple, c.tolist()))
    report.update({
        "key_color": key_color, "cell_size": 1, "palette_colors": pal_hex,
        "frames_in": len(frames), "frames_out": n, "dropped_duplicates": 0,
        "kept_frame_indices": list(range(n)),
        "crop_box": list(map(int, box)) if box else None,
        "frame_size": [int(small[0].shape[1]), int(small[0].shape[0])],
        "unique_colors": len(colors), "export_scale": max(1, min(32, int(scale))),
        "tags": [t if isinstance(t, dict) else {"name": t, "from": 0, "to": n - 1} for t in tags],
    })
    return {"sheet": sheet, "atlas": atlas, "atlas_json": json.dumps(atlas, indent=1),
            "gif": gif, "frames": frames_out, "report": report, "row_gifs": row_gifs}
