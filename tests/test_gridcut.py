"""Grid cutter: structure inference, drifted grids, borderless collages, real sheets."""
import io
import pathlib
import random

from PIL import Image, ImageDraw

from forge_mcp import gridcut

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _sheet(size, col_lines, row_lines, gutter_px=6, bg=(30, 30, 34), line=(9, 9, 11), seed=7):
    """A synthetic rendered sheet: content blobs per cell, gutters at the given
    (imperfect) positions — like a model that drew its grid a few % off."""
    rng = random.Random(seed)
    w, h = size
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    xs, ys = [0, *col_lines, w], [0, *row_lines, h]
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            x1, x2, y1, y2 = xs[c], xs[c + 1], ys[r], ys[r + 1]
            for _ in range(6):  # bright blobs filling most of the cell
                bx1 = rng.randint(x1 + 14, max(x1 + 15, x2 - 40))
                by1 = rng.randint(y1 + 14, max(y1 + 15, y2 - 40))
                d.ellipse([bx1, by1, min(bx1 + rng.randint(24, 90), x2 - 12),
                           min(by1 + rng.randint(24, 90), y2 - 12)],
                          fill=(rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255)))
    for x in col_lines:
        d.rectangle([x - gutter_px // 2, 0, x + gutter_px // 2, h], fill=line)
    for y in row_lines:
        d.rectangle([0, y - gutter_px // 2, w, y + gutter_px // 2], fill=line)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_detects_drifted_gutters():
    # 3x3 on 1024px: ideal lines at 341/683 — drawn at 316/706 and 361/668 (~2-7% drift)
    data = _sheet((1024, 1024), [316, 706], [361, 668])
    cells, meta = gridcut.snap_cut(data, 3, 3, 9)
    assert len(cells) == 9
    assert meta["mode"] == "detected" and not meta["mismatch"]
    for found, true in zip(meta["col_lines"], [316, 706]):
        assert abs(found - true) <= 4, (found, true)
    for found, true in zip(meta["row_lines"], [361, 668]):
        assert abs(found - true) <= 4, (found, true)


def test_heavy_drift_2x2():
    # 2x2 on 900px: ideal 450 — drawn at 512 x / 396 y (~13% off, worst realistic case)
    data = _sheet((900, 900), [512], [396])
    cells, meta = gridcut.snap_cut(data, 2, 2, 4)
    assert abs(meta["col_lines"][0] - 512) <= 4
    assert abs(meta["row_lines"][0] - 396) <= 4
    assert not meta["mismatch"]


def test_real_imagen_sheet_3x3_when_2x3_asked():
    # Real Imagen output (downscaled): asked for a 2x3 grid of 6 spells, the model drew
    # a 3x3 with an outer border and duplicated subjects. The cutter must recover the
    # DRAWN grid — 9 clean cells — and flag the mismatch. Blind 2x3 division sliced
    # every subject in half (the bug this module exists to fix).
    data = (FIXTURES / "imagen-3x3-when-2x3-asked.png").read_bytes()
    cells, meta = gridcut.snap_cut(data, 2, 3, 6)
    assert meta["mode"] == "detected"
    assert (meta["cols"], meta["rows"]) == (3, 3)
    assert meta["mismatch"] is True
    assert len(cells) == 9
    # drawn magenta gutters sit near thirds (sheet is 512px here)
    for found, true in zip(meta["col_lines"], [512 * 1 / 3, 512 * 2 / 3]):
        assert abs(found - true) <= 512 * 0.04, (found, true)
    for found, true in zip(meta["row_lines"], [512 * 1 / 3, 512 * 2 / 3]):
        assert abs(found - true) <= 512 * 0.04, (found, true)


def test_borderless_collage_falls_back_to_snapping():
    # No gutters drawn at all — dense noise. Must still return count cells with lines
    # inside the search window of the ideals (either fallback or a legit flat band).
    rng = random.Random(3)
    im = Image.new("RGB", (600, 600))
    px = im.load()
    for y in range(600):
        for x in range(600):
            px[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    cells, meta = gridcut.snap_cut(buf.getvalue(), 2, 2, 4)
    assert meta["mode"] in ("snapped", "uniform")
    assert len(cells) == 4
    assert abs(meta["col_lines"][0] - 300) <= int(300 * gridcut.SEARCH_WINDOW_FRAC) + 1
    assert abs(meta["row_lines"][0] - 300) <= int(300 * gridcut.SEARCH_WINDOW_FRAC) + 1


def test_detected_returns_all_drawn_cells():
    # 5 subjects requested in a 2x3 grid; the drawn sheet genuinely has 6 content
    # cells — detection returns all 6 (the tool layer trims filler), inset applied.
    data = _sheet((800, 1200), [401], [396, 805])
    cells, meta = gridcut.snap_cut(data, 2, 3, 5)
    assert meta["mode"] == "detected" and len(cells) == 6
    assert meta["inset"] >= 3
    for cell in cells:  # inset + residue shave may trim a little, never gut a cell
        assert cell.width > 280 and cell.height > 280


def test_single_column_and_no_snap():
    data = _sheet((400, 800), [], [405])
    cells, meta = gridcut.snap_cut(data, 1, 2, 2, snap=False)
    assert len(cells) == 2 and meta["mode"] == "uniform"
    cells2, meta2 = gridcut.snap_cut(data, 1, 2, 2)
    assert abs(meta2["row_lines"][0] - 405) <= 4  # found even with a single column


def test_shave_uniform_frame():
    # A cut cell dragging its own drawn frame: magenta strips down the left and
    # bottom, dark interior with a blob. Shave removes the strips, keeps content.
    im = Image.new("RGB", (300, 300), (35, 28, 40))
    d = ImageDraw.Draw(im)
    d.ellipse([90, 80, 210, 200], fill=(240, 180, 60))
    d.rectangle([0, 0, 13, 299], fill=(200, 40, 190))    # left frame strip
    d.rectangle([0, 286, 299, 299], fill=(200, 40, 190))  # bottom frame strip
    out = gridcut.shave_uniform_frame(im)
    assert out.width <= 300 - 13 and out.height <= 300 - 13
    # a solid keying background must NOT be shaved (interior == border colour)
    solid = Image.new("RGB", (300, 300), (255, 0, 255))
    ImageDraw.Draw(solid).ellipse([90, 80, 210, 200], fill=(30, 200, 90))
    assert gridcut.shave_uniform_frame(solid).size == (300, 300)
