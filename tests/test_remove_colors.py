"""remove_colors / the pipeline's background key must not punch holes in the subject.
Pure numpy/scipy, no network or model downloads (skip_bg=True skips rembg)."""
import io

import numpy as np
from PIL import Image

from backend.processing import PipelineOptions, remove_colors, run_pipeline


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _load(data):
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))


def _paper_icon():
    """A white page with a dark outline on a white background: the page is the same
    colour as the background, sealed off from it by the outline."""
    a = np.full((60, 60, 4), 255, dtype=np.uint8)          # white background, opaque
    a[10:50, 15:45, :3] = 60                               # dark outline block
    a[13:47, 18:42, :3] = 255                              # white page inside it
    return a


def test_border_only_keeps_matching_colour_inside_the_subject():
    out = _load(remove_colors(_png(_paper_icon()), [(255, 255, 255)], 20, border_only=True))
    assert out[2, 2, 3] == 0          # background around the subject is cleared
    assert out[30, 30, 3] == 255      # the white page inside the outline stays
    assert out[11, 16, 3] == 255      # the outline stays


def test_global_key_still_clears_every_match_when_asked():
    out = _load(remove_colors(_png(_paper_icon()), [(255, 255, 255)], 20))
    assert out[2, 2, 3] == 0
    assert out[30, 30, 3] == 0        # named colours are keyed everywhere, as before


def test_pipeline_auto_detected_background_is_keyed_from_the_border_only():
    opts = PipelineOptions(skip_bg=True, color_remove=True, color_auto_detect=True)
    out = _load(run_pipeline(_png(_paper_icon()), opts))
    assert out[2, 2, 3] == 0
    assert out[30, 30, 3] == 255


def test_pipeline_named_colours_are_keyed_everywhere():
    a = _paper_icon()
    a[25:30, 25:30, :3] = (0, 200, 0)                      # a green mark on the page
    opts = PipelineOptions(skip_bg=True, color_remove=True, colors=[(0, 200, 0)])
    out = _load(run_pipeline(_png(a), opts))
    assert out[27, 27, 3] == 0        # the named green is removed even inside the subject
    assert out[2, 2, 3] == 255        # and the (un-named) white background is left alone


def test_border_only_with_no_border_match_clears_nothing():
    a = np.zeros((20, 20, 4), dtype=np.uint8)
    a[..., :3] = (10, 10, 10); a[..., 3] = 255
    a[8:12, 8:12, :3] = 255                                # sealed white pocket only
    out = _load(remove_colors(_png(a), [(255, 255, 255)], 20, border_only=True))
    assert out[10, 10, 3] == 255
