"""clean_edges — remove background trapped inside a cut-out (letter counters,
keyholes, circles) while preserving outlines and artwork. Pure numpy/scipy, no
network or model downloads."""
import io

import numpy as np
from PIL import Image

from backend.processing import clean_edges


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _load(data):
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))


def test_clean_edges_punches_sealed_pocket_keeps_outline_and_art():
    a = np.zeros((40, 40, 4), dtype=np.uint8)          # fully transparent
    a[8:32, 8:32] = (200, 30, 30, 255)                 # red sprite body (a "letter")
    a[16:24, 16:24] = (255, 255, 255, 255)             # SEALED white pocket (the counter)
    a[8:12, 8:12] = (255, 255, 255, 255)               # white that touches the transparent edge (an outline)

    out = _load(clean_edges(_png(a)))

    # the sealed white pocket is cleared
    assert out[20, 20, 3] == 0
    # white on the silhouette edge (touches transparency) is preserved
    assert out[9, 9, 3] == 255 and tuple(out[9, 9, :3]) == (255, 255, 255)
    # the artwork itself is untouched
    assert out[14, 14, 3] == 255 and tuple(out[14, 14, :3]) == (200, 30, 30)
    # nothing leaked into the transparent border
    assert out[2, 2, 3] == 0


def test_clean_edges_only_targets_the_background_colour():
    a = np.zeros((30, 30, 4), dtype=np.uint8)
    a[6:24, 6:24] = (40, 40, 40, 255)                  # dark sprite body
    a[12:18, 12:18] = (0, 200, 0, 255)                 # sealed GREEN pocket

    # default target is white → the green pocket is not background-coloured → kept
    assert _load(clean_edges(_png(a)))[15, 15, 3] == 255
    # target the green → the sealed pocket is cleared
    assert _load(clean_edges(_png(a), color=(0, 200, 0)))[15, 15, 3] == 0


def test_clean_edges_size_guards():
    a = np.zeros((40, 40, 4), dtype=np.uint8)
    a[4:36, 4:36] = (30, 30, 200, 255)                 # blue body, opaque area = 32*32 = 1024
    a[10:30, 10:30] = (255, 255, 255, 255)             # big sealed white fill (400px ~ 39% of body)

    # max_pocket_frac is an UPPER guard: clear pockets up to the fraction, keep bigger ones.
    # 39% < default 0.5 → cleared; 39% > 0.2 → kept.
    assert _load(clean_edges(_png(a)))[20, 20, 3] == 0
    assert _load(clean_edges(_png(a), max_pocket_frac=0.2))[20, 20, 3] == 255
