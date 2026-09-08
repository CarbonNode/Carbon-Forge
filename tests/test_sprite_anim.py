"""Tests for backend/sprite_anim.py — video frames -> locked-style sprite sheet."""
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageSequence

from backend import sprite_anim as sa

SCRATCH_PAL = np.array([[200, 40, 40], [40, 200, 40], [40, 40, 220],
                        [240, 220, 60], [30, 30, 30]], np.uint8)


def png(rgba):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def load(data):
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))


def logical_sprite(seed=1, size=24, pad=4):
    """A 5-color random sprite on transparent, padded to (size+2*pad)^2."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(SCRATCH_PAL), size=(size, size))
    full = size + 2 * pad
    spr = np.zeros((full, full, 4), np.uint8)
    spr[pad:pad + size, pad:pad + size, :3] = SCRATCH_PAL[idx]
    spr[pad:pad + size, pad:pad + size, 3] = 255
    return spr


def upscale(rgba, factor, blur=0.0):
    big = np.repeat(np.repeat(rgba, factor, 0), factor, 1)
    if blur:
        big = np.array(Image.fromarray(big, "RGBA").filter(ImageFilter.GaussianBlur(blur)))
    return big


def fake_video_frames(spr, n=6, video=480, blur=0.6, noise=6, seed=2):
    """Simulate an I2V clip: the sprite shifts 0/1/2 px per frame, composited on
    magenta, resized to the video size with bilinear blur + compression noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = np.roll(spr, i % 3, axis=1)
        fk, _ = sa.composite_on_key(png(upscale(f, 16, blur)))
        im = Image.open(io.BytesIO(fk)).resize((video, video), Image.BILINEAR)
        arr = np.array(im).astype(np.int16)
        arr[..., :3] += rng.integers(-noise, noise + 1, size=arr[..., :3].shape)
        frames.append(png(np.clip(arr, 0, 255).astype(np.uint8)))
    return frames


# --- timestamps ---------------------------------------------------------------

def test_frame_timestamps_loop_excludes_end():
    t = sa.frame_timestamps(2.0, 4)
    assert t == [0.0, 0.5, 1.0, 1.5]
    t2 = sa.frame_timestamps(2.0, 3, loop=False)
    assert t2[0] == 0.0 and t2[-1] == pytest.approx(1.98, abs=0.01)  # headroom before the end
    assert sa.frame_timestamps(2.0, 1) == [0.0]
    with pytest.raises(ValueError):
        sa.frame_timestamps(2.0, 4, start_s=1.5, end_s=1.0)


# --- reference preparation ---------------------------------------------------

def test_prepare_reference_integer_upscales_and_pads_with_key():
    spr = logical_sprite(size=24, pad=4)  # 32x32
    out, info = sa.prepare_reference(png(spr), (512, 512), "#ff00ff")
    arr = load(out)
    assert arr.shape == (512, 512, 4) and info["upscale"] == 16
    assert (arr[..., 3] == 255).all()
    assert tuple(arr[0, 0, :3]) == (255, 0, 255)  # padded corner = key
    # nearest-neighbor: a 16x16 block is perfectly flat
    x0, y0 = info["placed"][0] + 4 * 16, info["placed"][1] + 4 * 16
    block = arr[y0:y0 + 16, x0:x0 + 16, :3]
    assert (block == block[0, 0]).all()


def test_prepare_reference_shrinks_oversized_source():
    big = np.zeros((900, 700, 4), np.uint8)
    big[..., 3] = 255
    out, info = sa.prepare_reference(png(big), (512, 512))
    arr = load(out)
    assert arr.shape[:2] == (512, 512) and info["upscale"] == 1
    assert info["placed"][3] <= 512 and info["placed"][2] <= 512


def test_composite_on_key_and_border_detect():
    spr = logical_sprite()
    flat, had = sa.composite_on_key(png(spr), "#00ff00")
    assert had is True
    arr = load(flat)
    assert (arr[..., 3] == 255).all() and tuple(arr[0, 0, :3]) == (0, 255, 0)
    assert sa.detect_border_color(flat) == "#00ff00"
    again, had2 = sa.composite_on_key(flat)
    assert had2 is False


# --- style lock ----------------------------------------------------------------

def test_lock_style_carries_reference_grid_onto_frame_size():
    spr = logical_sprite(size=24, pad=4)              # 32 logical px
    ai_ref = png(upscale(spr, 16, 0.8))               # 512px "AI" render
    style = sa.lock_style(ai_ref, (480, 480), max_colors=16)
    assert style["cell_size"] == 15                   # 480 / 32
    assert style["logical_size"] == [32, 32]
    assert style["report"]["grid"]["detected"] is True
    assert style["palette"] and len(style["palette"]) <= 16
    # every source color survives the k-means lock
    got = {tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in style["palette"]}
    for col in SCRATCH_PAL:
        assert any(max(abs(int(a) - int(b)) for a, b in zip(tuple(col), gcol)) <= 8 for gcol in got)


def test_lock_style_manual_cell_and_retro_palette():
    spr = logical_sprite()
    style = sa.lock_style(png(spr), (512, 512), cell_size=16, palette="gb_pocket")
    assert style["cell_size"] == 16 and style["logical_size"] == [32, 32]
    assert style["palette"] == ["#000000", "#545454", "#a8a8a8", "#ffffff"]
    with pytest.raises(ValueError):
        sa.lock_style(png(spr), (512, 512), palette="nope")


# --- frames ---------------------------------------------------------------------

def test_refine_frame_keys_background_after_sampling():
    spr = logical_sprite()
    frames = fake_video_frames(spr, n=1)
    small = sa.refine_frame(frames[0], 15, key_color="#ff00ff", palette=[sa._hex(c) for c in SCRATCH_PAL])
    assert small.shape == (32, 32, 4)
    assert small[0, 0, 3] == 0                        # background keyed
    assert small[4:28, 4:28, 3].mean() > 250          # sprite body solid
    # no fringe: every opaque color is one of the palette colors
    opaque = small[small[..., 3] > 0][:, :3]
    pal = {tuple(c) for c in SCRATCH_PAL.tolist()}
    assert all(tuple(c) in pal for c in opaque.tolist())


def test_crop_union_is_shared_and_dedupe_drops_repeats():
    a = np.zeros((10, 10, 4), np.uint8); a[2:4, 2:4] = 255
    b = np.zeros((10, 10, 4), np.uint8); b[6:8, 6:8] = 255
    cropped, box = sa.crop_union([a, b])
    assert box == (2, 2, 8, 8)
    assert all(f.shape == (6, 6, 4) for f in cropped)
    kept, idx = sa.dedupe_frames([a, a, b, b, a])
    assert idx == [0, 2, 4]


# --- packing ----------------------------------------------------------------------

def test_pack_sheet_strip_and_grid_atlas():
    frames = [np.full((8, 6, 4), v, np.uint8) for v in (40, 80, 120, 160, 200)]
    sheet, atlas = sa.pack_sheet(frames, fps=10, name="run", scale=2)
    arr = load(sheet)
    assert arr.shape == (16, 60, 4)
    assert atlas["meta"]["layout"] == {"columns": 5, "rows": 1, "frame_w": 12, "frame_h": 16,
                                        "padding": 0, "count": 5, "fps": 10, "logical_frame": [6, 8]}
    assert atlas["frames"][3]["frame"] == {"x": 36, "y": 0, "w": 12, "h": 16}
    assert atlas["frames"][0]["duration"] == 100
    assert atlas["meta"]["frameTags"][0] == {"name": "run", "from": 0, "to": 4, "direction": "forward"}
    sheet2, atlas2 = sa.pack_sheet(frames, columns=2, padding=1)
    assert load(sheet2).shape == (3 * 8 + 2, 2 * 6 + 1, 4)
    assert atlas2["frames"][2]["frame"]["y"] == 9
    with pytest.raises(ValueError):
        sa.pack_sheet([frames[0], np.zeros((3, 3, 4), np.uint8)])


def test_gif_is_animated_and_transparent():
    frames = []
    for i in range(3):
        f = np.zeros((8, 8, 4), np.uint8)
        f[i:i + 3, 2:6, :3] = SCRATCH_PAL[i]
        f[i:i + 3, 2:6, 3] = 255
        frames.append(f)
    data = sa.gif_bytes(frames, fps=5, scale=2)
    im = Image.open(io.BytesIO(data))
    assert im.format == "GIF" and im.n_frames == 3 and im.size == (16, 16)
    assert im.info.get("duration") == 200
    assert "transparency" in im.info


# --- end to end -------------------------------------------------------------------

def test_build_animation_end_to_end_locks_style_and_aligns():
    spr = logical_sprite(size=24, pad=4)
    ai_ref = png(upscale(spr, 16, 0.8))
    frames = fake_video_frames(spr, n=6)
    times = sa.frame_timestamps(3.0, 6)
    res = sa.build_animation(frames, reference=ai_ref, key_color="#ff00ff", fps=8, scale=3, name="walk",
                             source_times=times)
    rep = res["report"]
    assert rep["cell_size"] == 15 and rep["frames_out"] == 6 and rep["dropped_duplicates"] == 0
    assert rep["frame_size"] == [26, 24]              # 24 wide + 2px of shift, shared box
    assert rep["unique_colors"] <= 5                  # never more colors than the sprite
    assert rep["key_color"] == "#ff00ff"
    sheet = load(res["sheet"])
    assert sheet.shape == (24 * 3, 26 * 3 * 6, 4)
    atlas = json.loads(res["atlas_json"])
    assert atlas["meta"]["size"] == {"w": 26 * 3 * 6, "h": 24 * 3}
    assert len(res["frames"]) == 6 and load(res["frames"][0]).shape == (72, 78, 4)
    assert [f["source_time_s"] for f in atlas["frames"]] == times


def test_build_animation_auto_key_and_dedupe_without_reference():
    spr = logical_sprite()
    frames = fake_video_frames(spr, n=4, blur=0.0, noise=0)
    frames = [frames[0], frames[0], frames[1], frames[2]]
    res = sa.build_animation(frames, key_color="auto", max_colors=8)
    assert res["report"]["key_color"] == "#ff00ff"
    assert res["report"]["frames_out"] == 3 and res["report"]["kept_frame_indices"] == [0, 2, 3]


def test_pack_existing_aligns_mixed_sizes():
    a = np.zeros((10, 12, 4), np.uint8); a[1:5, 1:5] = 255
    b = np.zeros((9, 7, 4), np.uint8); b[3:8, 2:6] = 200
    res = sa.pack_existing([png(a), png(b)], fps=6, name="mix")
    # canvas 12x10, bottom-centered: a's box x1..5/y1..5, b's x4..8/y4..9 -> union 7x8
    assert res["report"]["frame_size"] == [7, 8]
    assert res["report"]["crop_box"] == [1, 1, 8, 9]
    assert res["atlas"]["meta"]["layout"]["count"] == 2
    assert load(res["sheet"]).shape == (8, 14, 4)


# --- sheets as input (Retro Diffusion / PixelLab / Aseprite exports) ----------

def four_dir_sheet(cell=16, cols=4, rows=4):
    """A cols x rows grid of `cell` px cells; each cell holds a distinct
    5-color blob whose position differs per row so the shared crop matters."""
    rng = np.random.default_rng(5)
    sheet = np.zeros((rows * cell, cols * cell, 4), np.uint8)
    for r in range(rows):
        for c in range(cols):
            idx = rng.integers(0, len(SCRATCH_PAL), size=(6, 6))
            y0, x0 = r * cell + 2 + r, c * cell + 3 + c  # drifts per row/col
            sheet[y0:y0 + 6, x0:x0 + 6, :3] = SCRATCH_PAL[idx]
            sheet[y0:y0 + 6, x0:x0 + 6, 3] = 255
    return sheet


def test_slice_sheet_row_major_and_drops_empty_tail():
    sheet = four_dir_sheet()
    sheet[48:, 32:] = 0  # last row: only 2 of 4 cells drawn
    frames, cols, rows = sa.slice_sheet(png(sheet), 16, 16)
    assert (cols, rows) == (4, 4)
    assert len(frames) == 14  # two trailing empty cells dropped
    assert frames[0].shape == (16, 16, 4)
    assert (frames[5] == sheet[16:32, 16:32]).all()  # row 1, col 1


def test_frame_tags_accept_ranges_and_clamp():
    tags = sa._frame_tags([{"name": "down", "from": 0, "to": 3},
                           {"name": "left", "from": 12, "to": 99}, "all"], 16, "x")
    assert tags[0] == {"name": "down", "from": 0, "to": 3, "direction": "forward"}
    assert tags[1]["to"] == 15
    assert tags[2] == {"name": "all", "from": 0, "to": 15, "direction": "forward"}


def test_build_from_sheet_tags_rows_and_shares_one_box():
    sheet = four_dir_sheet()
    res = sa.build_from_sheet(png(sheet), 16, 16, row_tags=sa.RD_DIRECTIONS, name="k", fps=8)
    rep, atlas = res["report"], res["atlas"]
    assert rep["source_layout"] == {"columns": 4, "rows": 4, "frame_w": 16, "frame_h": 16, "count": 16}
    names = [t["name"] for t in atlas["meta"]["frameTags"]]
    assert names == ["down", "right", "up", "left"]
    assert atlas["meta"]["frameTags"][2] == {"name": "up", "from": 8, "to": 11, "direction": "forward"}
    # one shared crop: the blobs drift 3px across rows/cols -> box spans the drift + 6px
    assert rep["frame_size"] == [9, 9]
    assert atlas["meta"]["layout"]["columns"] == 4 and atlas["meta"]["layout"]["rows"] == 4
    assert set(res["row_gifs"]) == {"down", "right", "up", "left"}
    assert Image.open(io.BytesIO(res["row_gifs"]["left"])).n_frames == 4
    assert Image.open(io.BytesIO(res["gif"])).n_frames == 16
    assert rep["unique_colors"] <= len(SCRATCH_PAL)
    # frames are kept 1:1 — no resampling
    assert (load(res["frames"][0])[..., 3] == 255).sum() == 36


def test_build_from_sheet_palette_lock_and_key():
    sheet = four_dir_sheet()
    opaque = sheet.copy()
    opaque[..., :3][opaque[..., 3] == 0] = (255, 0, 255)
    opaque[..., 3] = 255
    res = sa.build_from_sheet(png(opaque), 16, 16, key_color="#FF00FF", max_colors=3,
                              row_tags=sa.RD_DIRECTIONS)
    assert res["report"]["key_color"] == "#FF00FF"
    assert len(res["report"]["palette_colors"]) == 3
    assert res["report"]["unique_colors"] <= 3
    assert (load(res["frames"][0])[..., 3] == 0).any()  # background keyed out


# --- native frame for "animate YOUR sprite" APIs ---------------------------------

def test_prepare_native_frame_collapses_grid_and_upscales_to_min():
    spr = logical_sprite(size=20, pad=2)            # 24x24 logical, 20x20 opaque
    big = upscale(spr, 8)                           # 192x192 fake-upscaled pixel art
    png_bytes, (w, h), info = sa.prepare_native_frame(png(big), 32, 256)
    assert info["grid"]["cell"] == [8, 8]
    assert (w, h) == (40, 40)                        # 20 logical px x2 = 40 >= 32
    assert info["upscale"] == 2
    arr = load(png_bytes)
    assert arr.shape[:2] == (40, 40)
    assert (arr[..., 3] == 255).sum() == 20 * 20 * 4
    assert arr[-1, :, 3].any()                       # feet on the floor


def test_prepare_native_frame_fits_oversized_and_pads_to_multiple():
    spr = logical_sprite(size=300, pad=0)
    png_bytes, (w, h), info = sa.prepare_native_frame(png(spr), 32, 256, pad_to_multiple=8)
    assert max(w, h) == 256 and info.get("fitted")
    assert w % 8 == 0 and h % 8 == 0


def test_rd_snap_frames_and_payload():
    from forge_mcp import retrodiffusion_api as RD
    assert [RD.snap_frames(n) for n in (1, 5, 7, 9, 11, 14, 99)] == [4, 4, 6, 8, 10, 12, 16]
    body = RD.advanced_payload("slow steps", "walking", b"\x89PNG", 64, 64, frames=7, seed=3)
    assert body["prompt_style"] == "rd_advanced_animation__walking"
    assert body["frames_duration"] == 6 and body["num_images"] == 1 and body["return_spritesheet"] is True
    assert body["input_image"] == "iVBORw=="
    with pytest.raises(Exception):
        RD.advanced_payload("x", "fly", b"", 64, 64)
    with pytest.raises(Exception):
        RD.advanced_payload("x", "idle", b"", 20, 20)


# --- the pixel-art FEEL: despill, key-free palette, pose selection, hold timing ---

def test_despill_removes_magenta_fringe_but_keeps_other_colors():
    f = np.zeros((2, 2, 4), np.uint8)
    f[..., 3] = 255
    f[0, 0, :3] = (200, 40, 190)   # magenta-tinted edge pixel
    f[0, 1, :3] = (40, 200, 40)    # green stays
    f[1, 0, :3] = (120, 120, 120)  # grey stays
    out = sa.despill_key(f, "#FF00FF")
    assert tuple(out[0, 0, :3]) == (50, 40, 40)          # excess 150 pulled out of R and B
    assert tuple(out[0, 1, :3]) == (40, 200, 40)
    assert tuple(out[1, 0, :3]) == (120, 120, 120)
    assert sa.is_key_like((250, 10, 240), "#FF00FF") and sa.is_key_like((180, 40, 170), "#FF00FF")
    assert not sa.is_key_like((120, 40, 40), "#FF00FF")


def test_lock_style_palette_excludes_key_fringe():
    spr = logical_sprite(size=20, pad=4)
    # paint a magenta-tinted rim inside the sprite, as video compression does
    spr[4, 4:24, :3] = (230, 30, 220)
    style = sa.lock_style(png(upscale(spr, 8)), (224, 224), max_colors=6, key_color="#FF00FF")
    assert not any(sa.is_key_like(sa.pa.parse_hex_color(c), "#FF00FF") for c in style["palette"])


def test_select_poses_spacing_and_distinctness():
    base = logical_sprite(size=16, pad=2)
    frames = []
    for i in range(24):
        f = base.copy()
        if 8 <= i < 12:      # a big pose in the middle
            f = np.roll(f, 6, axis=0)
        elif i >= 18:        # another late
            f = np.roll(f, 5, axis=1)
        frames.append(f)
    picks = sa.select_poses(frames, 4, min_gap=3)
    assert picks[0] == 0 and len(picks) == 4
    assert any(8 <= p < 12 for p in picks) and any(p >= 18 for p in picks)
    assert all(b - a >= 3 for a, b in zip(picks, picks[1:]))


def test_build_animation_poses_and_hold_timing():
    spr = logical_sprite(size=20, pad=4)
    dense = fake_video_frames(spr, n=24)
    times = [i * 0.1 for i in range(24)]
    res = sa.build_animation(dense, reference=png(upscale(spr, 16)), key_color="#FF00FF",
                             frame_select="poses", pose_count=6, dedupe=False, hold_timing=True,
                             source_times=times, clip_seconds=2.4, fps=8)
    rep = res["report"]
    assert rep["frame_select"] == "poses" and rep["despill"] is True
    assert rep["frames_out"] <= 6 and rep["kept_frame_indices"][0] == 0
    assert len(rep["durations_ms"]) == rep["frames_out"] and sum(rep["durations_ms"]) >= 2000
    atlas_durs = [f["duration"] for f in res["atlas"]["frames"]]
    assert atlas_durs == rep["durations_ms"]
    g = Image.open(io.BytesIO(res["gif"]))
    # Pillow merges identical consecutive GIF frames (and adds their durations)
    assert 1 < g.n_frames <= rep["frames_out"]
    total = sum(fr.info.get("duration", 0) for fr in ImageSequence.Iterator(g))
    assert abs(total - sum(rep["durations_ms"])) <= 20
