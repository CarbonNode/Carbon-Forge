"""Sheet -> cells: find the grid a model ACTUALLY rendered and cut along it.

Models asked for an NxM grid routinely take liberties: lines drift a few percent,
gutters vary in width, an outer border/margin appears, and sometimes the model
draws a DIFFERENT grid entirely (a 3x3 when asked for 2x3, duplicating subjects
to fill it). Blind equal division then slices straight through artwork.

Two layers of defence, best first:
1. Structure inference — find every full-height/full-width uniform band (gutters
   and outer margins) and derive the drawn grid from them: content bounds, line
   positions, real column/row count. Used whenever the bands form a regular grid,
   even when it disagrees with what was requested (that disagreement is reported).
2. Per-line snapping — when no clean band structure exists (borderless collage),
   snap each REQUESTED line to the flattest safe band near its ideal position,
   falling back to the exact uniform position per line. Even here the "flattest
   band" is by definition the least-content cut path.

Pure numpy + PIL — no heavy deps, unit-testable off-box.
"""
import io

import numpy as np
from PIL import Image

# Per-line snapping: how far from the ideal uniform position a line may drift
# (fraction of one cell).
SEARCH_WINDOW_FRAC = 0.16
# A band's smoothed variability must be at most this fraction of the profile's
# median to count as gutter-flat (drawn gutters score near zero).
GUTTER_STD_RATIO = 0.6
# Structure inference sanity: max grid lines per axis, min cell size (axis frac),
# max spread between the widest and narrowest cell of an axis.
MAX_CELLS_PER_AXIS = 8
MIN_CELL_FRAC = 0.06
# Outer cells are measured margin-to-line-centre vs. centre-to-centre for inner ones,
# so honest grids still show some spread; false structures typically exceed 2x.
MAX_CELL_SPREAD = 1.8


def _axis_profile(gray: np.ndarray) -> np.ndarray:
    """Per-column variability of a float (H, W) grayscale array: std down each column
    plus the mean |horizontal gradient|. A gutter/margin column — one flat color top
    to bottom — scores near zero; columns crossing artwork score high."""
    std = gray.std(axis=0)
    grad = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    return std + np.pad(grad, (1, 0), mode="edge")


def _smooth(profile: np.ndarray, cells_hint: int) -> np.ndarray:
    k = max(3, profile.size // (max(1, cells_hint) * 40))
    return np.convolve(profile, np.ones(k) / k, mode="same")


# A drawn grid line must deviate at least this much (gray levels) in mean brightness
# from its surrounding gutter background to count as a line at all — real lines score
# 20-70, blob-edge noise in an empty streak stays under ~4.
LINE_MIN_DEV = 6.0


def _line_in_run(bright: np.ndarray, a: int, b: int) -> tuple[int, float]:
    """Cut position inside the flat run [a, b] (absolute indices into `bright`): the
    centre of the drawn line — the band deviating most in brightness from the run's
    background — or the run's centre when no line is discernible. Returns
    (position, deviation_peak); a peak < LINE_MIN_DEV means no visible line."""
    sub = bright[a:b + 1]
    dev = np.abs(sub - float(np.median(sub)))
    peak = float(dev.max())
    if peak < LINE_MIN_DEV:  # no drawn line, just a flat band — cut its middle
        return (a + b) // 2, peak
    on_line = dev >= peak * 0.5
    j = int(dev.argmax())
    la = lb = j
    while la > 0 and on_line[la - 1]:
        la -= 1
    while lb < on_line.size - 1 and on_line[lb + 1]:
        lb += 1
    return a + (la + lb) // 2, peak


def _flat_bands(smooth: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous gutter-flat runs [(start, end)] along a smoothed profile. Runs
    separated by a whisker of variability are merged — a gutter zone routinely
    fragments around the drawn line's own highlight/bevel — then slivers drop."""
    thr = max(1e-6, float(np.median(smooth)) * GUTTER_STD_RATIO)
    idx = np.flatnonzero(smooth <= thr)
    if idx.size == 0:
        return []
    n = smooth.size
    gap = max(4, round(n * 0.02))
    bands = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap:
            bands.append((start, prev))
            start = i
        prev = i
    bands.append((start, prev))
    min_w = max(2, round(n * 0.004))
    return [(a, b) for a, b in bands if b - a + 1 >= min_w]


def _structure(smooth: np.ndarray, bright: np.ndarray) -> list[int] | None:
    """Derive one axis of the drawn grid from its flat bands: [start, line..., end]
    boundaries (content bounds from edge-touching margin bands, cut positions from
    interior bands). None when the bands don't form a plausibly regular grid."""
    n = smooth.size
    lead, trail, cands = 0, n, []
    for a, b in _flat_bands(smooth):
        if a <= 2:
            lead = max(lead, b + 1)
        elif b >= n - 3:
            trail = min(trail, a)
        else:
            pos, peak = _line_in_run(bright, a, b)
            # only a band containing a VISIBLE drawn line is a grid-line candidate —
            # a flat streak with no line is just empty space inside a cell
            if peak >= LINE_MIN_DEV:
                cands.append((peak, pos))
    if trail - lead < n * MIN_CELL_FRAC:
        return None
    # strongest lines first, keep the largest subset that forms a regular grid —
    # weeds out the odd spurious "line" a sparse sheet can cough up
    cands.sort(reverse=True)
    for k in range(min(len(cands), MAX_CELLS_PER_AXIS - 1), -1, -1):
        bounds = [lead, *sorted(p for _, p in cands[:k]), trail]
        widths = np.diff(bounds)
        if widths.min() >= n * MIN_CELL_FRAC and widths.max() / widths.min() <= MAX_CELL_SPREAD:
            return bounds
    return None


def _snap_lines(smooth: np.ndarray, bright: np.ndarray, cells: int) -> tuple[list[int], int]:
    """Fallback: positions of the cells-1 interior boundaries, each snapped to the
    flattest band near its ideal uniform spot when one exists. (positions, snapped)."""
    n = smooth.size
    if cells <= 1:
        return [], 0
    positions, snapped = [], 0
    for i in range(1, cells):
        ideal = n * i / cells
        half = max(4, int(n / cells * SEARCH_WINDOW_FRAC))
        lo, hi = max(1, int(ideal) - half), min(n - 1, int(ideal) + half)
        win = smooth[lo:hi]
        if win.size:
            j = int(win.argmin())
            med = float(np.median(win))
            if med > 1e-6 and float(win[j]) <= med * GUTTER_STD_RATIO:
                flat = win <= med * GUTTER_STD_RATIO
                a = b = j
                while a > 0 and flat[a - 1]:
                    a -= 1
                while b < win.size - 1 and flat[b + 1]:
                    b += 1
                positions.append(_line_in_run(bright, lo + a, lo + b)[0])
                snapped += 1
                continue
        positions.append(int(round(ideal)))
    return positions, snapped


def shave_uniform_frame(img, max_frac: float = 0.12):
    """Trim near-uniform border lines off a cut cell — the cell's own drawn frame or
    bevel and any gutter residue — stopping at real content. A border line is shaved
    only while it is flat (low std) AND clearly differs from the cell's interior, so
    a solid keying background (interior == border) and edge-touching artwork (high
    std) are both left alone."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]

    def frame_line(line, ref):
        # flat AND different from what lies at max shave depth on this side: a frame
        # ends there, a keying background continues (and so is never shaved)
        return (float(line.std(axis=0).mean()) < 12.0
                and float(np.abs(np.median(line, axis=0) - ref).mean()) > 20.0)

    top, bot, left, right = 0, h, 0, w
    lim_h, lim_w = max(1, int(h * max_frac)), max(1, int(w * max_frac))
    ref_t = np.median(a[min(lim_h, h - 1)], axis=0)
    ref_b = np.median(a[max(h - 1 - lim_h, 0)], axis=0)
    ref_l = np.median(a[:, min(lim_w, w - 1)], axis=0)
    ref_r = np.median(a[:, max(w - 1 - lim_w, 0)], axis=0)
    while top < lim_h and frame_line(a[top, left:right], ref_t):
        top += 1
    while h - bot < lim_h and frame_line(a[bot - 1, left:right], ref_b):
        bot -= 1
    while left < lim_w and frame_line(a[top:bot, left], ref_l):
        left += 1
    while w - right < lim_w and frame_line(a[top:bot, right - 1], ref_r):
        right -= 1
    if (top, bot, left, right) != (0, h, 0, w):
        return img.crop((left, top, right, bot))
    return img


def snap_cut(data: bytes, cols: int, rows: int, count: int,
             inset: int | None = None, snap: bool = True,
             max_cells: int = 24, shave: bool = True) -> tuple[list, dict]:
    """Cut a rendered sheet into its cells, in reading order.

    data: encoded sheet bytes. cols/rows: the grid the model was ASKED for.
    count: subjects requested. inset: px shaved off every cell edge (None = auto,
    ~2% of a cell, min 3). snap=False forces pure equal division of the requested
    grid (no detection).

    When the drawn structure is detected it wins — ALL drawn cells are returned
    (capped at max_cells) so a model that drew extra/duplicate cells still yields
    clean individual images; meta["mismatch"] flags a drawn grid that differs from
    the requested one. Otherwise the requested grid is cut with per-line snapping.
    Returns (cells: list[PIL.Image, RGB], meta)."""
    sheet = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = sheet.size
    xs = ys = None
    mode = "uniform"
    snapped = 0
    if snap:
        gray = np.asarray(sheet.convert("L"), dtype=np.float32)
        prof_v, prof_h = _axis_profile(gray), _axis_profile(gray.T)
        sm_v, sm_h = _smooth(prof_v, cols), _smooth(prof_h, rows)
        br_v, br_h = gray.mean(axis=0), gray.mean(axis=1)
        xs, ys = _structure(sm_v, br_v), _structure(sm_h, br_h)
        # Trust detection only when each axis either found real interior lines or was
        # asked for a single cell — a grid axis that "found" nothing is ambiguous
        # (undetectable gutters vs. genuinely one row/col), so fall back to snapping.
        ok_v = xs is not None and (len(xs) > 2 or cols == 1)
        ok_h = ys is not None and (len(ys) > 2 or rows == 1)
        if ok_v and ok_h and (len(xs) > 2 or len(ys) > 2 or (cols, rows) == (1, 1)):
            mode = "detected"
        else:
            col_lines, s_c = _snap_lines(sm_v, br_v, cols)
            row_lines, s_h = _snap_lines(sm_h, br_h, rows)
            xs, ys = [0, *col_lines, w], [0, *row_lines, h]
            mode, snapped = "snapped", s_c + s_h
    if mode == "uniform":
        xs = [round(w * i / cols) for i in range(cols + 1)]
        ys = [round(h * i / rows) for i in range(rows + 1)]
    d_cols, d_rows = len(xs) - 1, len(ys) - 1
    if inset is None:
        inset = max(3, round(min((xs[-1] - xs[0]) / d_cols, (ys[-1] - ys[0]) / d_rows) * 0.02))
    n_cells = d_cols * d_rows if mode == "detected" else min(count, d_cols * d_rows)
    cells = []
    for idx in range(min(n_cells, max_cells)):
        r_i, c_i = divmod(idx, d_cols)
        x1, x2 = xs[c_i] + inset, xs[c_i + 1] - inset
        y1, y2 = ys[r_i] + inset, ys[r_i + 1] - inset
        if x2 - x1 < 8 or y2 - y1 < 8:  # degenerate after inset — cut without it
            x1, x2, y1, y2 = xs[c_i], xs[c_i + 1], ys[r_i], ys[r_i + 1]
        cell = sheet.crop((x1, y1, x2, y2))
        cells.append(shave_uniform_frame(cell) if shave else cell)
    meta = {
        "mode": mode, "cols": d_cols, "rows": d_rows, "inset": inset,
        "requested": {"cols": cols, "rows": rows},
        "mismatch": (d_cols, d_rows) != (cols, rows),
        "snapped_lines": snapped if mode == "snapped" else (d_cols - 1) + (d_rows - 1),
        "expected_lines": (cols - 1) + (rows - 1),
        "col_lines": [int(v) for v in xs[1:-1]],
        "row_lines": [int(v) for v in ys[1:-1]],
    }
    return cells, meta
