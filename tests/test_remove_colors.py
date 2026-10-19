"""remove_colors / the pipeline's background key must not punch holes in the subject.
Pure numpy/scipy, no network or model downloads (skip_bg=True skips rembg)."""
import io
from pathlib import Path

import numpy as np
from PIL import Image

from backend.processing import PipelineOptions, remove_colors, run_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


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


def test_border_only_reaches_fringe_through_already_transparent_background():
    """After rembg the background is transparent black, so a grey fringe left around the
    subject only touches the border through transparency. It must still be keyed."""
    a = np.zeros((40, 40, 4), dtype=np.uint8)              # transparent black (rembg output)
    a[10:30, 10:30] = (128, 128, 128, 200)                 # grey fringe ring around subject
    a[12:28, 12:28] = (30, 30, 30, 255)                    # dark subject outline block
    a[15:25, 15:25] = (128, 128, 128, 255)                 # grey INSIDE the subject (keep)
    out = _load(remove_colors(_png(a), [(128, 128, 128)], 20, border_only=True))
    assert out[11, 11, 3] == 0        # fringe cleared
    assert out[20, 20, 3] == 255      # sealed grey inside the subject kept
    assert out[13, 13, 3] == 255      # outline kept


def test_border_only_keeps_an_opaque_grey_face_that_touches_the_transparency():
    """A graphite subject cut from the grey icon backdrop: its opaque grey face meets the
    transparent background at the silhouette. Reaching it through transparency erased
    whole faces of real icons (2026-10-19); only the semi-transparent fringe may go."""
    a = np.zeros((40, 40, 4), dtype=np.uint8)              # transparent black (rembg output)
    a[8:32, 8:32] = (128, 128, 128, 200)                   # grey fringe ring around subject
    a[10:30, 10:30] = (120, 120, 120, 255)                 # opaque grey face, on the edge
    out = _load(remove_colors(_png(a), [(128, 128, 128)], 20, border_only=True))
    assert out[9, 9, 3] == 0          # fringe cleared
    assert out[10, 20, 3] == 255      # face edge kept
    assert out[20, 20, 3] == 255      # face kept


def test_real_graphite_icon_keeps_its_faces():
    """Real BiRefNet cut of an Imagen graphite icon rendered on the #808080 icon backdrop
    (backdrop detected as (122,122,122)). The key must take fringe only: when it flooded
    through transparency it erased 23.5% of this subject (2026-10-19). Synthetic tests
    alone let that through, so a real render pins it."""
    a = _load(open(FIXTURES / "icon-graphite-birefnet-cut.png", "rb").read())
    out = _load(remove_colors(_png(a), [(122, 122, 122)], 20, border_only=True))
    subject = a[..., 3] > 128
    lost = (subject & (out[..., 3] < 128)).sum() / subject.sum()
    assert lost < 0.01, f"background key erased {lost:.1%} of the subject"
