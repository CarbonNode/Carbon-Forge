"""Tests for backend/pixel_art.py — the PixelRefiner port."""
import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from backend import pixel_art as pa


def make_logical(seed=0, w=24, h=24, colors=6):
    """A deterministic random logical pixel image (RGBA, fully opaque)."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 3), dtype=np.uint8)
    # keep palette entries far apart so blur can't merge them
    palette = (palette // 64) * 64 + 32
    idx = rng.integers(0, colors, size=(h, w))
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = palette[idx]
    rgba[..., 3] = 255
    return rgba


def upscale(rgba, factor, blur=0.0):
    big = np.repeat(np.repeat(rgba, factor, axis=0), factor, axis=1)
    if blur > 0:
        img = Image.fromarray(big, "RGBA").filter(
            ImageFilter.GaussianBlur(radius=blur))
        big = np.array(img)
    return big


def to_png(rgba):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), "RGBA").save(
        buf, format="PNG")
    return buf.getvalue()


# --- grid detection ---------------------------------------------------------

def test_detect_grid_exact_upscale():
    logical = make_logical(seed=1)
    big = upscale(logical, 8)
    g = pa.detect_grid(big)
    assert g["detected"]
    assert g["cell_w"] == 8 and g["cell_h"] == 8
    assert g["offset_x"] == 0 and g["offset_y"] == 0
    assert g["out_w"] == 24 and g["out_h"] == 24


def test_detect_grid_with_blur():
    logical = make_logical(seed=2)
    big = upscale(logical, 8, blur=1.0)
    g = pa.detect_grid(big)
    assert g["detected"]
    assert g["cell_w"] == 8 and g["cell_h"] == 8


def test_detect_grid_blob_with_blur_and_noise():
    # regression: a round sprite on transparency, blurred + noisy — coarse
    # periods with only a handful of boundary lines used to out-score the
    # true cell size via max-over-offsets cherry-picking (picked 30x20)
    rng = np.random.default_rng(99)
    logical = make_logical(seed=99, w=48, h=48, colors=8)
    yy, xx = np.mgrid[0:48, 0:48]
    outside = ((yy - 24) ** 2 + (xx - 24) ** 2) >= 20 ** 2
    logical[outside] = 0
    big = upscale(logical, 10, blur=1.5)
    noise = rng.normal(0, 5, big[..., :3].shape)
    big[..., :3] = np.clip(big[..., :3].astype(float) + noise,
                           0, 255).astype(np.uint8)
    g = pa.detect_grid(big)
    assert g["detected"]
    assert g["cell_w"] == 10 and g["cell_h"] == 10


def test_detect_grid_survives_transparent_background():
    logical = make_logical(seed=3, w=16, h=16)
    logical[:4, :, 3] = 0  # transparent band
    big = upscale(logical, 6)
    g = pa.detect_grid(big)
    assert g["detected"]
    assert g["cell_w"] == 6 and g["cell_h"] == 6


# --- cell sampling ----------------------------------------------------------

def test_sample_cells_recovers_exact_colors():
    logical = make_logical(seed=4)
    big = upscale(logical, 8)
    g = pa.detect_grid(big)
    small = pa.sample_cells(big, g, mode="medoid")
    assert small.shape == logical.shape
    assert np.array_equal(small, logical)


def test_sample_cells_recovers_colors_through_blur():
    logical = make_logical(seed=5)
    big = upscale(logical, 8, blur=1.0)
    g = pa.detect_grid(big)
    small = pa.sample_cells(big, g, mode="medoid")
    # medoid picks real source pixels from the cell core, so nearly all cells
    # should land exactly on a logical color despite the blur
    match = (small[..., :3] == logical[..., :3]).all(axis=-1).mean()
    assert match > 0.9


def test_sample_cells_mean_mode():
    logical = make_logical(seed=6)
    big = upscale(logical, 4)
    g = {"cell_w": 4, "cell_h": 4, "offset_x": 0, "offset_y": 0,
         "out_w": 24, "out_h": 24}
    small = pa.sample_cells(big, g, mode="mean")
    assert small.shape == logical.shape
    assert np.array_equal(small[..., :3], logical[..., :3])


def test_sample_cells_hard_alpha():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4] = [255, 0, 0, 255]  # left half opaque red
    g = {"cell_w": 4, "cell_h": 4, "offset_x": 0, "offset_y": 0,
         "out_w": 2, "out_h": 2}
    small = pa.sample_cells(rgba, g, mode="hard")
    assert set(np.unique(small[..., 3]).tolist()) <= {0, 255}
    assert small[0, 0, 3] == 255 and small[0, 1, 3] == 0


# --- quantization -----------------------------------------------------------

def test_kmeans_reduces_and_is_deterministic():
    rng = np.random.default_rng(7)
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., :3] = rng.integers(0, 256, size=(32, 32, 3))
    rgba[..., 3] = 255
    out1 = pa.quantize(rgba, 8)
    out2 = pa.quantize(rgba, 8)
    assert np.array_equal(out1, out2)
    colors, _, _ = pa._unique_weighted_colors(out1)
    assert colors.shape[0] <= 8


def test_kmeans_no_op_when_few_colors():
    logical = make_logical(seed=8, colors=4)
    out = pa.quantize(logical, 16)
    assert np.array_equal(out, logical)


def test_apply_palette_maps_into_palette():
    logical = make_logical(seed=9)
    pal = np.stack([pa.parse_hex_color(c)
                    for c in pa.RETRO_PALETTES["gb_legacy"]["colors"]])
    out = pa.apply_palette(logical, pal)
    colors, _, _ = pa._unique_weighted_colors(out)
    pal_set = {tuple(c) for c in pal.tolist()}
    assert all(tuple(c) in pal_set for c in colors.tolist())


def test_dither_modes_run():
    logical = make_logical(seed=10)
    pal = np.stack([pa.parse_hex_color(c)
                    for c in pa.RETRO_PALETTES["gb_pocket"]["colors"]])
    pal_set = {tuple(c) for c in pal.tolist()}
    for mode in ("floyd-steinberg", "bayer-2x2", "bayer-4x4", "bayer-8x8",
                 "ordered"):
        out = pa.apply_palette(logical, pal, dither=mode, strength=0.8)
        colors, _, _ = pa._unique_weighted_colors(out)
        assert all(tuple(c) in pal_set for c in colors.tolist()), mode


# --- outline / trim / scale -------------------------------------------------

def test_outline_rounded():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[2, 2] = [255, 0, 0, 255]
    out = pa.add_outline(rgba, pa.parse_hex_color("#00ff00"), style="rounded")
    assert out.shape == (7, 7, 4)
    # 8 neighbors painted
    painted = (out[..., 3] == 255).sum()
    assert painted == 9
    assert tuple(out[3, 2, :3]) == (0, 255, 0)


def test_outline_sharp_only_4way():
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[2, 2] = [255, 0, 0, 255]
    out = pa.add_outline(rgba, pa.parse_hex_color("#0000ff"), style="sharp")
    assert (out[..., 3] == 255).sum() == 5  # center + 4 neighbors


def test_auto_trim_and_scale():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[3:6, 4:8] = [1, 2, 3, 255]
    trimmed = pa.auto_trim(rgba)
    assert trimmed.shape == (3, 4, 4)
    scaled = pa.scale_nearest(trimmed, 3)
    assert scaled.shape == (9, 12, 4)
    assert np.array_equal(scaled[::3, ::3], trimmed)


def test_rgb555_round():
    rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    rgba[0, 0] = [130, 7, 250, 255]
    out = pa.rgb555_round(rgba)
    for v in out[0, 0, :3]:
        assert int(round(v / 255 * 31)) * 255 // 31 in range(v - 5, v + 6)


# --- background keying ------------------------------------------------------

def test_key_background_auto_border():
    rgba = np.full((12, 12, 4), 255, dtype=np.uint8)  # white bg
    rgba[4:8, 4:8, :3] = [200, 0, 0]
    out = pa.key_background(rgba, tolerance=10)
    assert out[0, 0, 3] == 0
    assert out[5, 5, 3] == 255


# --- end-to-end -------------------------------------------------------------

def test_refine_pixel_art_end_to_end():
    logical = make_logical(seed=11)
    big = upscale(logical, 8, blur=0.8)
    out_bytes, report = pa.refine_pixel_art(to_png(big), max_colors=8)
    img = Image.open(io.BytesIO(out_bytes))
    assert img.size == (24, 24)
    assert report["grid"]["cell_w"] == 8
    assert report["colors_after"] <= 8
    assert report["output_size"] == [24, 24]


def test_refine_manual_cell_size_and_scale():
    logical = make_logical(seed=12, w=16, h=16)
    big = upscale(logical, 4)
    out_bytes, report = pa.refine_pixel_art(
        to_png(big), cell_size=4, palette="pico8", dither="bayer-4x4",
        scale=2)
    img = Image.open(io.BytesIO(out_bytes))
    assert img.size == (32, 32)
    assert report["palette"] == "pico8"
    assert report["export_scale"] == 2


def test_refine_grid_off_preserves_size():
    logical = make_logical(seed=13, w=10, h=10)
    out_bytes, report = pa.refine_pixel_art(to_png(logical), grid="off",
                                            trim=False)
    img = Image.open(io.BytesIO(out_bytes))
    assert img.size == (10, 10)
    assert report["grid"]["mode"] == "off"


def test_refine_target_px_picks_integer_scale():
    logical = make_logical(seed=15, w=20, h=20)
    big = upscale(logical, 6)
    out_bytes, report = pa.refine_pixel_art(to_png(big), target_px=512,
                                            trim=False)
    img = Image.open(io.BytesIO(out_bytes))
    # 20px logical -> largest integer scale under 512 is x25, capped at x32 -> 25
    assert report["export_scale"] == 25
    assert img.size == (500, 500)


def test_refine_explicit_scale_beats_target_px():
    logical = make_logical(seed=16, w=16, h=16)
    out_bytes, report = pa.refine_pixel_art(to_png(logical), grid="off",
                                            scale=2, target_px=512, trim=False)
    img = Image.open(io.BytesIO(out_bytes))
    assert img.size == (32, 32)
    assert report["export_scale"] == 2


def test_refine_split_sprite_shape():
    # the split_sprites integration path: a small already-transparent sprite
    # (as run_split_pipeline emits) refined with auto grid
    logical = make_logical(seed=17, w=12, h=12, colors=4)
    logical[:2] = 0
    logical[:, :2] = 0
    big = upscale(logical, 8, blur=0.6)
    out_bytes, report = pa.refine_pixel_art(to_png(big), max_colors=4)
    assert report["grid"]["cell_w"] == 8 and report["grid"]["cell_h"] == 8
    assert report["colors_after"] <= 4


def test_refine_unknown_palette_raises():
    logical = make_logical(seed=14, w=8, h=8)
    with pytest.raises(ValueError, match="Unknown palette"):
        pa.refine_pixel_art(to_png(logical), palette="nope")
