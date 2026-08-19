"""Pixel-art refinement engine — a NumPy port of PixelRefiner
(https://github.com/HappyOnigiri/PixelRefiner, MIT).

Turns AI-generated "fake" pixel art (large images with soft ~NxN blocks,
anti-aliased edges and thousands of near-duplicate colors) into true
low-resolution pixel art:

  1. detect_grid      — find the logical pixel-cell size + offset per axis
  2. sample_cells     — collapse each cell to one color (Oklab medoid or mean)
  3. quantize/palette — deterministic weighted k-means in Oklab, retro palettes,
                        Floyd-Steinberg / Bayer dithering
  4. outline / trim / nearest-neighbor scale for export

Pure functions, bytes in / bytes out at the top level (refine_pixel_art), same
contract as backend.processing. No AI models involved — safe to run anywhere.

Port notes: constants and scoring follow PixelRefiner's detector.ts,
cell-sampler.ts, oklab-kmeans.ts and palette-dithering.ts. The thin-feature
continuity heuristic of the medoid sampler is intentionally not ported (it
needs per-cell neighbor probing; the core-region medoid already handles the
common cases). Everything is deterministic — no RNG anywhere.
"""
import io

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (mirrors PixelRefiner shared/config.ts)
# ---------------------------------------------------------------------------

DETECTION_STRIP_COUNT = 12
DETECTION_QUANT_STEP = 64
AUTO_MAX_CELLS = 512
MAX_CELL_PX = 256

# Soft-alpha "bleed only" cell rejection
SOFT_MIN_CELL_SIZE = 2
SOFT_MIN_RAMP_SPAN = 16
SOFT_RAMP_PEAK_TO_FLOOR = 2
SOFT_MAX_BLEED_PEAK = 192
SOFT_MAX_BLEED_COVERAGE = 128
HARD_EDGE_COVERAGE_THRESHOLD = 128

# Core region of a cell used for medoid candidates (excludes boundary blends)
CORE_MARGIN_RATIO = 0.375
CORE_MIN_SPAN = 2
CORE_MAX_MARGIN_PX = 6

CELL_ALPHA_THRESHOLD = 16
MAX_SAMPLES_PER_AXIS = 5  # ≤ 5x5 = 25 stratified samples per cell

RETRO_PALETTES = {
    "gb_legacy": {"name": "Game Boy (Legacy)",
                  "colors": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]},
    "gb_pocket": {"name": "Game Boy (Pocket)",
                  "colors": ["#000000", "#545454", "#a8a8a8", "#ffffff"]},
    "gb_light": {"name": "Game Boy (Light)",
                 "colors": ["#004040", "#15605d", "#308880", "#00e0e0"]},
    "pico8": {"name": "PICO-8", "colors": [
        "#000000", "#1D2B53", "#7E2553", "#008751", "#AB5236", "#5F574F",
        "#C2C3C7", "#FFF1E8", "#FF004D", "#FFA300", "#FFEC27", "#00E436",
        "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"]},
    "nes": {"name": "NES", "colors": [
        "#7C7C7C", "#0000FC", "#0000BC", "#4428BC", "#940084", "#A80020",
        "#A81000", "#881400", "#503000", "#007800", "#006800", "#005800",
        "#004058", "#000000", "#BCBCBC", "#0078F8", "#0058F8", "#6844FC",
        "#D800CC", "#E40058", "#F83800", "#E45C10", "#AC7C00", "#00B800",
        "#00A800", "#00A844", "#008888", "#F8F8F8", "#3CBCFC", "#6888FC",
        "#9878F8", "#F878F8", "#F85898", "#F87858", "#FCA044", "#F8B800",
        "#B8F818", "#58D854", "#58F898", "#00E8D8", "#787878", "#FCFCFC",
        "#A4E4FC", "#B8B8F8", "#D8B8F8", "#F8B8F8", "#F8A4C0", "#F0D0B0",
        "#FCE0A8", "#F8D878", "#D8F878", "#B8F8B8", "#B8F8D8", "#00FCFC",
        "#F8D8F8"]},
    "mono": {"name": "Monochrome", "colors": ["#000000", "#FFFFFF"]},
    "pc98": {"name": "PC-9801", "colors": [
        "#000000", "#0000F8", "#F80000", "#F800F8", "#00F800", "#00F8F8",
        "#F8F800", "#F8F8F8", "#888888", "#000088", "#880000", "#880088",
        "#008800", "#008888", "#888800", "#C0C0C0"]},
    "msx": {"name": "MSX1", "colors": [
        "#000000", "#3EB849", "#74D07D", "#5955E0", "#8076F1", "#B95E51",
        "#65DBEF", "#DB6559", "#FF897D", "#CCC35E", "#DED087", "#3AA241",
        "#B766B5", "#CCCCCC", "#FFFFFF"]},
    "c64": {"name": "Commodore 64", "colors": [
        "#000000", "#FFFFFF", "#813338", "#75CEC8", "#8E3C97", "#56AC4D",
        "#2E2C9B", "#EDF171", "#8E5029", "#553800", "#C46C71", "#4A4A4A",
        "#7B7B7B", "#A9FF9F", "#706DEB", "#B2B2B2"]},
    "arne16": {"name": "Arne 16", "colors": [
        "#000000", "#9D9D9D", "#FFFFFF", "#BE2633", "#E06F8B", "#493C2B",
        "#A46422", "#EB8931", "#F7E26B", "#2F484E", "#44891A", "#A3CE27",
        "#1B2632", "#005784", "#31A2F2", "#B2DCEF"]},
    # SFC-style: k-means reduce then round to RGB555
    "sfc_sprite": {"name": "SFC Style (16 colors)", "colors": [], "kmeans": 16,
                   "rgb555": True},
    "sfc_bg": {"name": "SFC Style (256 colors)", "colors": [], "kmeans": 256,
               "rgb555": True},
}

BAYER_MATRICES = {
    "bayer-2x2": (np.array([0, 2, 3, 1], dtype=np.float64).reshape(2, 2) + 0.5) / 4,
    "bayer-4x4": (np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5],
                           dtype=np.float64).reshape(4, 4) + 0.5) / 16,
    "bayer-8x8": (np.array([
        0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
        12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
        3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
        15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21],
        dtype=np.float64).reshape(8, 8) + 0.5) / 64,
    "ordered": (np.array([1, 9, 3, 11, 13, 5, 15, 7, 4, 12, 2, 10, 16, 8, 14, 6],
                         dtype=np.float64).reshape(4, 4) - 0.5) / 16,
}

DITHER_MODES = ("none", "floyd-steinberg") + tuple(BAYER_MATRICES.keys())


# ---------------------------------------------------------------------------
# Color space (sRGB <-> Oklab), vectorized
# ---------------------------------------------------------------------------

def _srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    out = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


_LMS_FROM_LINEAR = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188501, 0.6299787005]])
_OKLAB_FROM_LMS = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]])
_LMS_FROM_OKLAB = np.linalg.inv(_OKLAB_FROM_LMS)
_LINEAR_FROM_LMS = np.linalg.inv(_LMS_FROM_LINEAR)


def rgb_to_oklab(rgb, alpha=None):
    """rgb: (..., 3) uint8 -> oklab (..., 3) float64. Optional alpha (0..255)
    premultiplies in linear light (matches PixelRefiner's cell sampler)."""
    lin = _srgb_to_linear(rgb)
    if alpha is not None:
        lin = lin * (np.asarray(alpha, dtype=np.float64) / 255.0)[..., None]
    lms = lin @ _LMS_FROM_LINEAR.T
    return np.cbrt(lms) @ _OKLAB_FROM_LMS.T


def oklab_to_rgb(lab):
    lms = (np.asarray(lab, dtype=np.float64) @ _LMS_FROM_OKLAB.T) ** 3
    return _linear_to_srgb(lms @ _LINEAR_FROM_LMS.T)


def parse_hex_color(s):
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)],
                    dtype=np.uint8)


# ---------------------------------------------------------------------------
# Grid detection — boundary-contrast evidence
#
# PixelRefiner's shipping pipeline arbitrates grid candidates by how strongly
# image gradients concentrate on the predicted cell boundaries (its
# BOUNDARY_CONTRAST_LIMITS machinery); the legacy run-length detector alone
# over-favors tiny cells (every even boundary fits s=2 perfectly). We use the
# contrast evidence directly: for every candidate cell size, fold the per-axis
# gradient profile by the period, take the best phase, and score
#   evidence = mean gradient on predicted boundaries / mean gradient overall.
# A flat interior dilutes wrong periods; harmonics of the true cell keep full
# evidence, so we pick the FINEST size within tolerance of the best evidence
# (coarser multiples merge real logical pixels).
# ---------------------------------------------------------------------------

MIN_GRID_EVIDENCE = 1.1        # BOUNDARY_CONTRAST_LIMITS.minEvidence
HARMONIC_EVIDENCE_RATIO = 0.85  # keep the finest size within 85% of the peak


def _gradient_profile(rgba, axis):
    """Alpha-weighted mean color step between adjacent columns (axis=1) or
    rows (axis=0). Returns (g, w): edge strength and attainable alpha weight
    per boundary position i+1 along that axis. Normalizing by w keeps fully
    transparent bands from diluting the evidence of the true period."""
    rgb = rgba[..., :3].astype(np.float64)
    a = rgba[..., 3].astype(np.float64) / 255.0
    if axis == 1:
        d = np.abs(rgb[:, 1:] - rgb[:, :-1]).mean(axis=-1)
        wgt = np.minimum(a[:, 1:], a[:, :-1])
        return (d * wgt).sum(axis=0), wgt.sum(axis=0)
    d = np.abs(rgb[1:, :] - rgb[:-1, :]).mean(axis=-1)
    wgt = np.minimum(a[1:, :], a[:-1, :])
    return (d * wgt).sum(axis=1), wgt.sum(axis=1)


def _detect_axis(profile, length, max_cells):
    """Best (cell_size, offset, evidence) along one axis, or None."""
    g, wprof = profile
    total_g, total_w = g.sum(), wprof.sum()
    if total_g <= 0 or total_w <= 0 or length < 4:
        return None
    overall = total_g / total_w
    min_cell = max(2, int(np.ceil(length / max(2, max_cells))))
    max_cell = min(MAX_CELL_PX, length // 3)  # need >= 3 cells of evidence
    if max_cell < min_cell:
        return None

    # g index i == boundary between i and i+1, i.e. position i+1
    positions = np.arange(1, length)

    def evidence(s):
        folded_g = np.bincount(positions % s, weights=g, minlength=s)
        folded_w = np.bincount(positions % s, weights=wprof, minlength=s)
        means = np.where(folded_w > 0, folded_g / np.maximum(folded_w, 1e-12),
                         0.0)
        off = int(np.argmax(means))
        return off, float(means[off] / overall)

    best = None
    for s in range(min_cell, max_cell + 1):
        off, ev = evidence(s)
        if best is None or ev > best[2]:
            best = (s, off, ev)
    if best is None or best[2] < MIN_GRID_EVIDENCE:
        return None

    # prefer the finest cell size whose evidence is close to the peak —
    # coarser harmonics score the same but merge real logical pixels
    peak_ev = best[2]
    for s in range(min_cell, best[0]):
        off, ev = evidence(s)
        if ev >= peak_ev * HARMONIC_EVIDENCE_RATIO:
            return (s, off, ev)
    return best


def detect_grid(rgba, max_cells_w=AUTO_MAX_CELLS, max_cells_h=AUTO_MAX_CELLS):
    """Detect the logical pixel grid of an upscaled/AI pixel-art image.

    Returns a dict: cell_w, cell_h, offset_x, offset_y, out_w, out_h,
    score (boundary-contrast evidence, higher = more confident; ~1.0 means no
    grid), detected (False when detection failed and the 1:1 preserve fallback
    was used)."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    est_x = _detect_axis(_gradient_profile(rgba, 1), w, max_cells_w)
    est_y = _detect_axis(_gradient_profile(rgba, 0), h, max_cells_h)

    if est_x is None and est_y is None:
        return {"cell_w": 1, "cell_h": 1, "offset_x": 0, "offset_y": 0,
                "out_w": w, "out_h": h, "score": None, "detected": False}
    if est_x is None or est_y is None:
        # borrow the detected axis's cell size for the failed one
        known = est_x or est_y
        est_x = est_x or (known[0], 0, known[2])
        est_y = est_y or (known[0], 0, known[2])

    cell_w, cell_h = max(1, est_x[0]), max(1, est_y[0])
    # the boundary phase means grid lines sit at off, off+s, ... — the first
    # full cell starts at off (anything before it is a partial edge cell)
    off_x = est_x[1] % cell_w
    off_y = est_y[1] % cell_h
    return {
        "cell_w": cell_w, "cell_h": cell_h,
        "offset_x": off_x, "offset_y": off_y,
        "out_w": max(1, (w - off_x) // cell_w),
        "out_h": max(1, (h - off_y) // cell_h),
        "score": round((est_x[2] + est_y[2]) / 2, 3),
        "detected": True,
    }


# ---------------------------------------------------------------------------
# Cell sampling (port of cell-sampler.ts; fully vectorized)
# ---------------------------------------------------------------------------

def _core_margin(span):
    return min(span * CORE_MARGIN_RATIO, CORE_MAX_MARGIN_PX,
               max(0.0, (span - CORE_MIN_SPAN) / 2))


def sample_cells(rgba, grid, mode="medoid", alpha_threshold=CELL_ALPHA_THRESHOLD):
    """Collapse each grid cell to a single RGBA color.

    mode: 'medoid' (alpha-aware Oklab medoid — crisp, default),
          'mean'   (area-weighted premultiplied average — smoother),
          'hard'   (medoid with binary alpha — hard sprite edges).
    Returns uint8 array (out_h, out_w, 4).
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    cw, ch = grid["cell_w"], grid["cell_h"]
    ox, oy = grid["offset_x"], grid["offset_y"]
    out_w, out_h = grid["out_w"], grid["out_h"]
    if cw == 1 and ch == 1 and ox == 0 and oy == 0:
        return rgba[:out_h, :out_w].copy()

    rows = max(1, min(ch, MAX_SAMPLES_PER_AXIS))
    cols = max(1, min(cw, MAX_SAMPLES_PER_AXIS))

    # stratified sample coordinates, shared across all cells
    ry = np.floor((np.arange(rows) + 0.5) * ch / rows).astype(np.int64)  # (R,)
    rx = np.floor((np.arange(cols) + 0.5) * cw / cols).astype(np.int64)  # (C,)
    cell_y0 = oy + np.arange(out_h, dtype=np.int64) * ch                 # (H,)
    cell_x0 = ox + np.arange(out_w, dtype=np.int64) * cw                 # (W,)
    yy = np.minimum(cell_y0[:, None] + ry[None, :], h - 1)               # (H,R)
    xx = np.minimum(cell_x0[:, None] + rx[None, :], w - 1)               # (W,C)

    # gather: (H, W, R, C, 4)
    samples = rgba[yy[:, None, :, None], xx[None, :, None, :]]
    n = out_h * out_w
    s = rows * cols
    flat = samples.reshape(n, s, 4).astype(np.float64)
    rgb, a = flat[..., :3], flat[..., 3]

    coverage = a.mean(axis=1)                          # (N,) uniform strata
    peak, floor = a.max(axis=1), a.min(axis=1)
    bleed = np.zeros(n, dtype=bool)
    if cw >= SOFT_MIN_CELL_SIZE and ch >= SOFT_MIN_CELL_SIZE:
        bleed = ((peak - floor >= SOFT_MIN_RAMP_SPAN)
                 & (floor * SOFT_RAMP_PEAK_TO_FLOOR < peak)
                 & (peak < SOFT_MAX_BLEED_PEAK)
                 & (coverage < SOFT_MAX_BLEED_COVERAGE))

    out = np.zeros((n, 4), dtype=np.float64)

    if mode == "mean":
        af = a / 255.0
        asum = af.sum(axis=1)
        safe = np.maximum(asum, 1e-12)
        out[:, :3] = (rgb * af[..., None]).sum(axis=1) / safe[:, None]
        out[:, 3] = np.round(af.mean(axis=1) * 255.0)
        out[asum <= 0, :3] = 0
    else:
        # medoid in premultiplied Oklab, restricted to the cell core
        lab = rgb_to_oklab(flat[..., :3], flat[..., 3])          # (N,S,3)
        eligible = a >= alpha_threshold                          # (N,S)
        allow_all = ~eligible.any(axis=1)                        # (N,)
        eligible = eligible | allow_all[:, None]

        mx, my = _core_margin(cw), _core_margin(ch)
        core_x = (rx + 0.5 >= mx) & (rx + 0.5 <= cw - mx)        # (C,)
        core_y = (ry + 0.5 >= my) & (ry + 0.5 <= ch - my)        # (R,)
        core = (core_y[:, None] & core_x[None, :]).reshape(1, s)  # (1,S)
        pop = eligible & core
        has_core = pop.any(axis=1)
        pop = np.where(has_core[:, None], pop, eligible)         # (N,S)

        weights = np.where(allow_all[:, None], 1.0, a / 255.0) * pop  # (N,S)

        best = np.zeros(n, dtype=np.int64)
        chunk = max(1, 4_000_000 // (s * s))  # bound the (N,S,S) workspace
        for i in range(0, n, chunk):
            lab_c = lab[i:i + chunk]                              # (M,S,3)
            d2 = ((lab_c[:, :, None, :] - lab_c[:, None, :, :]) ** 2).sum(-1)
            score = (d2 * weights[i:i + chunk, None, :]).sum(-1)  # (M,S)
            score = np.where(pop[i:i + chunk], score, np.inf)
            best[i:i + chunk] = np.argmin(score, axis=1)
        idx = np.arange(n)
        out[:, :3] = flat[idx, best, :3]
        if mode == "hard":
            out[:, 3] = np.where(coverage >= HARD_EDGE_COVERAGE_THRESHOLD, 255, 0)
        else:
            out[:, 3] = np.round(coverage)

    out[bleed] = 0
    return np.clip(out, 0, 255).astype(np.uint8).reshape(out_h, out_w, 4)


# ---------------------------------------------------------------------------
# Deterministic weighted k-means in Oklab (port of oklab-kmeans.ts)
# ---------------------------------------------------------------------------

def _unique_weighted_colors(rgba):
    """Unique opaque colors sorted by packed key. Returns (rgb (U,3) uint8,
    counts (U,), labs (U,3))."""
    px = rgba.reshape(-1, 4)
    opaque = px[:, 3] > 0
    if not opaque.any():
        return (np.zeros((0, 3), np.uint8), np.zeros(0, np.int64),
                np.zeros((0, 3)))
    rgb = px[opaque, :3].astype(np.int64)
    keys = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    colors = np.stack([(uniq >> 16) & 0xFF, (uniq >> 8) & 0xFF, uniq & 0xFF],
                      axis=1).astype(np.uint8)
    return colors, counts, rgb_to_oklab(colors)


def kmeans_palette(rgba, max_colors, max_iterations=20, tolerance=1e-3):
    """Deterministic weighted k-means palette (Oklab). Returns (K,3) uint8."""
    colors, counts, labs = _unique_weighted_colors(rgba)
    u = colors.shape[0]
    if u == 0:
        return np.zeros((0, 3), np.uint8)
    if u <= max_colors:
        return colors

    # -- farthest-point weighted seeding (deterministic k-means++ analogue) --
    k = max_colors
    centroids = np.zeros((k, 3))
    used = np.zeros(u, dtype=bool)
    first = int(np.lexsort((np.arange(u), -counts))[0])
    centroids[0] = labs[first]
    used[first] = True
    min_d = ((labs - centroids[0]) ** 2).sum(-1)
    for ci in range(1, k):
        score = np.where(used, -1.0, min_d * counts)
        pick = int(np.argmax(score))
        centroids[ci] = labs[pick]
        used[pick] = True
        d = ((labs - centroids[ci]) ** 2).sum(-1)
        min_d = np.minimum(min_d, d)

    for _ in range(max_iterations):
        d2 = ((labs[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)  # (U,K)
        assign = np.argmin(d2, axis=1)
        w = counts.astype(np.float64)
        sums = np.zeros((k, 3))
        totals = np.zeros(k)
        np.add.at(sums, assign, labs * w[:, None])
        np.add.at(totals, assign, w)
        new_centroids = centroids.copy()
        nonempty = totals > 0
        new_centroids[nonempty] = sums[nonempty] / totals[nonempty, None]
        # reseed empty clusters to the farthest unclaimed color
        if (~nonempty).any():
            taken = np.zeros(u, dtype=bool)
            for ci in np.nonzero(~nonempty)[0]:
                dmin = ((labs[:, None, :] - new_centroids[None, :, :]) ** 2) \
                    .sum(-1).min(axis=1)
                dmin[taken] = -1.0
                pick = int(np.argmax(dmin))
                taken[pick] = True
                new_centroids[ci] = labs[pick]
        movement = ((new_centroids - centroids) ** 2).sum(-1).max()
        centroids = new_centroids
        if movement < tolerance ** 2:
            break

    rgbs = oklab_to_rgb(centroids)
    keys = (rgbs[:, 0].astype(np.int64) << 16) | \
           (rgbs[:, 1].astype(np.int64) << 8) | rgbs[:, 2].astype(np.int64)
    _, first_idx = np.unique(keys, return_index=True)
    return rgbs[np.sort(first_idx)]


# ---------------------------------------------------------------------------
# Palette mapping + dithering (port of palette-dithering.ts)
# ---------------------------------------------------------------------------

def _nearest_index(rgb_float, palette_labs):
    """rgb_float (...,3) 0..255 -> index of nearest palette color in Oklab."""
    lab = rgb_to_oklab(np.clip(rgb_float, 0, 255))
    d2 = ((lab[..., None, :] - palette_labs) ** 2).sum(-1)
    return np.argmin(d2, axis=-1)


def apply_palette(rgba, palette, dither="none", strength=1.0):
    """Map every opaque pixel to the nearest palette color (Oklab metric),
    optionally dithering. palette: (K,3) uint8. Returns new rgba uint8."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.shape[0] == 0:
        return rgba.copy()
    h, w = rgba.shape[:2]
    out = rgba.copy()
    opaque = rgba[..., 3] > 0
    palette_labs = rgb_to_oklab(palette)

    if dither == "floyd-steinberg" and strength > 0:
        work = rgba[..., :3].astype(np.float64)
        alpha = rgba[..., 3]
        res = np.zeros((h, w, 3), dtype=np.uint8)
        neighbors = ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16),
                     (1, 1, 1 / 16))
        pal_f = palette.astype(np.float64)
        for y in range(h):
            # nearest lookup row-by-row keeps the error feedback exact while
            # amortizing the Oklab conversion across the row
            for x in range(w):
                if alpha[y, x] == 0:
                    continue
                px = np.clip(work[y, x], 0, 255)
                ci = int(_nearest_index(px, palette_labs))
                chosen = pal_f[ci]
                err = (px - chosen) * strength
                res[y, x] = chosen
                for dx, dy, wgt in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and alpha[ny, nx] > 0:
                        work[ny, nx] = np.clip(work[ny, nx] + err * wgt, 0, 255)
        out[..., :3] = np.where(opaque[..., None], res, out[..., :3])
        return out

    work = rgba[..., :3].astype(np.float64)
    if dither in BAYER_MATRICES and strength > 0:
        m = BAYER_MATRICES[dither]
        tile = np.tile(m, (h // m.shape[0] + 1, w // m.shape[1] + 1))[:h, :w]
        bias = (tile - 0.5) * strength * 255.0
        work = np.clip(work + bias[..., None], 0, 255)
    idx = _nearest_index(work, palette_labs)
    mapped = palette[idx]
    out[..., :3] = np.where(opaque[..., None], mapped, out[..., :3])
    return out


def quantize(rgba, max_colors, dither="none", strength=1.0):
    """K-means color reduction to <= max_colors (skips if already fewer)."""
    colors, _, _ = _unique_weighted_colors(rgba)
    if colors.shape[0] <= max_colors:
        return np.asarray(rgba, dtype=np.uint8).copy()
    palette = kmeans_palette(rgba, max_colors)
    return apply_palette(rgba, palette, dither=dither, strength=strength)


# ---------------------------------------------------------------------------
# Outline / trim / scale (ports of outline.ts + export helpers)
# ---------------------------------------------------------------------------

def add_outline(rgba, color, style="rounded"):
    """1px outline around opaque regions. style: 'rounded' (8-way) or
    'sharp' (4-way). Expands the canvas by 1px on every side."""
    if style == "none":
        return np.asarray(rgba, dtype=np.uint8).copy()
    rgba = np.asarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    padded = np.zeros((h + 2, w + 2, 4), dtype=np.uint8)
    padded[1:-1, 1:-1] = rgba
    alpha = padded[..., 3] > 0
    if style == "sharp":
        shifts = ((0, -1), (0, 1), (-1, 0), (1, 0))
    else:
        shifts = ((0, -1), (0, 1), (-1, 0), (1, 0),
                  (-1, -1), (1, -1), (-1, 1), (1, 1))
    neighbor = np.zeros_like(alpha)
    for dy, dx in shifts:
        neighbor |= np.roll(np.roll(alpha, dy, axis=0), dx, axis=1)
    edge = neighbor & ~alpha
    out = padded.copy()
    out[edge, :3] = np.asarray(color, dtype=np.uint8)
    out[edge, 3] = 255
    return out


def auto_trim(rgba, threshold=16, margin=0):
    """Crop to the bounding box of pixels with alpha >= threshold."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    mask = rgba[..., 3] >= threshold
    if not mask.any():
        return rgba.copy()
    ys, xs = np.nonzero(mask)
    y0 = max(0, ys.min() - margin)
    y1 = min(rgba.shape[0], ys.max() + 1 + margin)
    x0 = max(0, xs.min() - margin)
    x1 = min(rgba.shape[1], xs.max() + 1 + margin)
    return rgba[y0:y1, x0:x1].copy()


def scale_nearest(rgba, factor):
    """Integer nearest-neighbor upscale for export (1..32)."""
    factor = int(factor)
    if factor <= 1:
        return np.asarray(rgba, dtype=np.uint8).copy()
    return np.repeat(np.repeat(rgba, factor, axis=0), factor, axis=1)


def rgb555_round(rgba):
    """Round color channels to the SNES 15-bit gamut."""
    rgba = np.asarray(rgba, dtype=np.uint8).copy()
    c = rgba[..., :3].astype(np.float64)
    rgba[..., :3] = np.clip(np.round(np.round(c / 255 * 31) * 255 / 31),
                            0, 255).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Background keying (simple flat-color removal, pre-grid)
# ---------------------------------------------------------------------------

def key_background(rgba, color=None, tolerance=24):
    """Make pixels near the background color transparent. When color is None
    the dominant border color is used (samples all four edges)."""
    rgba = np.asarray(rgba, dtype=np.uint8).copy()
    if color is None:
        edges = np.concatenate([
            rgba[0, :, :3], rgba[-1, :, :3],
            rgba[:, 0, :3], rgba[:, -1, :3]]).astype(np.int64)
        keys = (edges[:, 0] << 16) | (edges[:, 1] << 8) | edges[:, 2]
        uniq, counts = np.unique(keys, return_counts=True)
        k = int(uniq[np.argmax(counts)])
        color = np.array([(k >> 16) & 0xFF, (k >> 8) & 0xFF, k & 0xFF])
    color = np.asarray(color, dtype=np.int64)
    dist = np.abs(rgba[..., :3].astype(np.int64) - color).max(axis=-1)
    rgba[dist <= tolerance, 3] = 0
    return rgba


# ---------------------------------------------------------------------------
# Top-level pipeline — bytes in, PNG bytes + report out
# ---------------------------------------------------------------------------

def refine_pixel_art(data, grid="auto", cell_size=0, max_cells=AUTO_MAX_CELLS,
                     sampling="medoid", remove_bg=False, bg_color=None,
                     bg_tolerance=24, max_colors=0, palette=None,
                     dither="none", dither_strength=1.0,
                     outline="none", outline_color="#000000",
                     trim=True, scale=1):
    """Full PixelRefiner pipeline. Returns (png_bytes, report_dict).

    grid: 'auto' (detect), 'off' (keep resolution), or pass cell_size > 0.
    sampling: 'medoid' | 'mean' | 'hard'.
    palette: a RETRO_PALETTES key or a list of '#rrggbb' strings; max_colors
    (k-means) applies when no palette is given.
    """
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    report = {"input_size": [img.width, img.height]}

    if remove_bg:
        color = parse_hex_color(bg_color) if bg_color else None
        rgba = key_background(rgba, color=color, tolerance=bg_tolerance)

    if cell_size and int(cell_size) > 1:
        cs = int(cell_size)
        g = {"cell_w": cs, "cell_h": cs, "offset_x": 0, "offset_y": 0,
             "out_w": max(1, rgba.shape[1] // cs),
             "out_h": max(1, rgba.shape[0] // cs),
             "score": None, "detected": False}
        report["grid"] = {**g, "mode": "manual"}
    elif grid == "auto":
        g = detect_grid(rgba, max_cells_w=max_cells, max_cells_h=max_cells)
        report["grid"] = {**g, "mode": "auto"}
    else:
        g = {"cell_w": 1, "cell_h": 1, "offset_x": 0, "offset_y": 0,
             "out_w": rgba.shape[1], "out_h": rgba.shape[0],
             "score": None, "detected": False}
        report["grid"] = {**g, "mode": "off"}

    small = sample_cells(rgba, g, mode=sampling) \
        if (g["cell_w"] > 1 or g["cell_h"] > 1) else rgba.copy()

    colors_before, _, _ = _unique_weighted_colors(small)
    report["colors_before"] = int(colors_before.shape[0])

    if palette:
        if isinstance(palette, str):
            spec = RETRO_PALETTES.get(palette)
            if spec is None:
                raise ValueError(
                    f"Unknown palette '{palette}'. "
                    f"Available: {', '.join(sorted(RETRO_PALETTES))}")
            if spec.get("kmeans"):
                small = quantize(small, spec["kmeans"], dither=dither,
                                 strength=dither_strength)
                if spec.get("rgb555"):
                    small = rgb555_round(small)
                report["palette"] = palette
            else:
                pal = np.stack([parse_hex_color(c) for c in spec["colors"]])
                small = apply_palette(small, pal, dither=dither,
                                      strength=dither_strength)
                report["palette"] = palette
        else:
            pal = np.stack([parse_hex_color(c) for c in palette])
            small = apply_palette(small, pal, dither=dither,
                                  strength=dither_strength)
            report["palette"] = "custom"
    elif max_colors and int(max_colors) > 0:
        small = quantize(small, int(max_colors), dither=dither,
                         strength=dither_strength)
        report["palette"] = f"kmeans-{int(max_colors)}"

    if outline and outline != "none":
        small = add_outline(small, parse_hex_color(outline_color), style=outline)

    if trim:
        small = auto_trim(small)

    colors_after, _, _ = _unique_weighted_colors(small)
    report["colors_after"] = int(colors_after.shape[0])
    report["output_size"] = [int(small.shape[1]), int(small.shape[0])]

    scale = max(1, min(32, int(scale)))
    if scale > 1:
        small = scale_nearest(small, scale)
        report["export_scale"] = scale
        report["export_size"] = [int(small.shape[1]), int(small.shape[0])]

    buf = io.BytesIO()
    Image.fromarray(small, "RGBA").save(buf, format="PNG")
    return buf.getvalue(), report
