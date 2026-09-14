"""Tests for the 3D→pixel bake: backend/bake_spec.py (pure conventions) and
backend/sprite_bake.py (compose + Blender process plumbing) — no Blender needed."""
import io
import json
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend import bake_spec as bs
from backend import sprite_bake as sb


# ---------------------------------------------------------------------------
# bake_spec — the contract both sides of the Blender boundary rely on
# ---------------------------------------------------------------------------

def test_normalize_spec_defaults_and_validation():
    spec = bs.normalize_spec({"directions": "8", "frames": 6})
    assert spec["directions"] == 8 and spec["frames"] == 6
    assert spec["engine"] == "cycles" and spec["loop"] is True and spec["actions"] is None
    with pytest.raises(ValueError):
        bs.normalize_spec({"directions": 5})
    with pytest.raises(ValueError):
        bs.normalize_spec({"engine": "octane"})
    assert bs.normalize_spec({"actions": ["", " walk "]})["actions"] == [" walk "]
    assert bs.normalize_spec({"actions": [""]})["actions"] is None
    assert bs.normalize_spec({"elevation_deg": 200})["elevation_deg"] == 89.0


def test_direction_tags_and_angles():
    assert bs.direction_tags(4) == ["s", "e", "n", "w"]
    assert bs.direction_tags(8)[:3] == ["s", "se", "e"] and len(bs.direction_tags(16)) == 16
    assert bs.direction_angle_deg(2, 8) == 90.0
    assert bs.row_tag("walk", "s") == "walk_s" and bs.row_tag("walk", "s", single_action=True) == "s"


def test_sample_frame_numbers_loop_vs_oneshot():
    # a cycle never repeats its first pose: 4 samples over 0..8 -> 0,2,4,6
    assert bs.sample_frame_numbers(0, 8, 4, loop=True) == [0, 2, 4, 6]
    # a one-shot keeps its final pose: 0..9 in 4 -> 0,3,6,9
    assert bs.sample_frame_numbers(0, 9, 4, loop=False) == [0, 3, 6, 9]
    assert bs.sample_frame_numbers(5, 5, 3) == [5, 5, 5]
    assert bs.sample_frame_numbers(0, 10, 1) == [0]


def test_resolve_action_names_matches_meshy_style_names():
    avail = ["Armature|Casual_Walk", "Armature|Attack", "Dead"]
    assert bs.resolve_action_names(avail) == avail
    assert bs.resolve_action_names(avail, ["walk", "DEAD"]) == ["Armature|Casual_Walk", "Dead"]
    with pytest.raises(ValueError, match="not in the model"):
        bs.resolve_action_names(avail, ["fly"])


# ---------------------------------------------------------------------------
# sprite_bake — Blender plumbing (no Blender)
# ---------------------------------------------------------------------------

def test_bake_command_and_result_parsing():
    assert sb.bake_command(("blender", "/usr/bin/blender"), "/tmp/s.json")[:3] == ["/usr/bin/blender", "-b", "--python"]
    assert sb.bake_command(("python", "/opt/bpy/bin/python"), "/tmp/s.json") == \
        ["/opt/bpy/bin/python", sb.BAKE_SCRIPT, "/tmp/s.json"]
    out = "Fra:1 Mem:1M\nSaved: 'x.png'\nBAKE_RESULT {\"frames_rendered\": 3, \"rows\": 1}\n"
    assert sb.parse_bake_result(out) == {"frames_rendered": 3, "rows": 1}
    assert sb.parse_bake_result("nothing here") is None


def test_find_blender_prefers_explicit_paths(tmp_path, monkeypatch):
    fake = tmp_path / "blender"
    fake.write_text("")
    monkeypatch.delenv("FORGE_BLENDER_BIN", raising=False)
    monkeypatch.delenv("FORGE_BPY_PYTHON", raising=False)
    assert sb.find_blender(blender_bin=str(fake)) == ("blender", str(fake))
    assert sb.find_blender(bpy_python=str(fake), blender_bin="/nope") == ("python", str(fake))


def test_run_blender_reports_missing_runner(monkeypatch):
    monkeypatch.setattr(sb, "find_blender", lambda **kw: None)
    with pytest.raises(sb.BakeError, match="No Blender"):
        sb.run_blender_sync(b"glTF", {"directions": 4})


def test_run_blender_reads_manifest_and_frames(tmp_path, monkeypatch):
    """A fake 'blender' that writes the manifest + PNGs the real script would."""
    script = tmp_path / "fake_blender.py"
    script.write_text(
        "import json, sys, os\n"
        "spec = json.load(open(sys.argv[-1]))\n"
        "os.makedirs(spec['out_dir'], exist_ok=True)\n"
        "from PIL import Image\n"
        "rows = []\n"
        "for d in ['s', 'n']:\n"
        "    files = []\n"
        "    for i in range(2):\n"
        "        n = f'walk_{d}_{i:02d}.png'\n"
        "        Image.new('RGBA', (8, 8), (255, 0, 0, 255)).save(os.path.join(spec['out_dir'], n))\n"
        "        files.append(n)\n"
        "    rows.append({'action': 'walk', 'direction': d, 'tag': 'walk_' + d, 'files': files, 'frame_numbers': [0, 5]})\n"
        "json.dump({'rows': rows, 'directions': 2, 'frames_rendered': 4}, open(os.path.join(spec['out_dir'], 'manifest.json'), 'w'))\n"
        "print('BAKE_RESULT ' + json.dumps({'frames_rendered': 4}))\n")
    import sys
    monkeypatch.setattr(sb, "BAKE_SCRIPT", str(script))
    manifest, frames = sb.run_blender_sync(b"glTF....", {"directions": 2, "frames": 2},
                                           runner=("python", sys.executable))
    assert manifest["frames_rendered"] == 4 and len(frames) == 4
    assert set(frames) == {"walk_s_00.png", "walk_s_01.png", "walk_n_00.png", "walk_n_01.png"}
    rows = sb.rows_from_manifest(manifest, frames, action_prefix="stroll")
    assert [r["tag"] for r in rows] == ["stroll_s", "stroll_n"] and len(rows[0]["frames"]) == 2
    assert [r["tag"] for r in sb.rows_from_manifest(manifest, frames, action_prefix="")] == ["s", "n"]


# ---------------------------------------------------------------------------
# sprite_bake — compose (renders → pixel-art bundle)
# ---------------------------------------------------------------------------

def render_frame(size, offset_y, colors, supersample=4):
    """A fake supersampled render: a figure of stacked blobs on transparent, feet at a
    fixed baseline; offset_y raises the body (a jump) without moving the feet."""
    px = size * supersample
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    base = px - 4 * supersample
    d.rectangle([px // 2 - 3 * supersample, base - 2 * supersample, px // 2 + 3 * supersample, base],
                fill=colors[0])                                        # feet, never move
    top = base - (12 + offset_y) * supersample
    d.rectangle([px // 2 - 4 * supersample, top, px // 2 + 4 * supersample, base - 3 * supersample],
                fill=colors[1])                                        # body
    d.ellipse([px // 2 - 3 * supersample, top - 6 * supersample, px // 2 + 3 * supersample, top],
              fill=colors[2])                                          # head
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def load(png):
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_downsample_snaps_to_cells_and_hardens_alpha():
    small = sb.downsample(render_frame(32, 0, [(200, 40, 40, 255), (40, 200, 40, 255), (40, 40, 220, 255)]), 4)
    assert small.shape == (32, 32, 4)
    assert set(np.unique(small[..., 3]).tolist()) <= {0, 255}


def test_compose_locks_palette_crop_and_tags():
    cols = [(200, 40, 40, 255), (40, 200, 40, 255), (40, 40, 220, 255)]
    rows = [
        {"tag": "walk_s", "frames": [render_frame(32, 0, cols), render_frame(32, 1, cols)]},
        {"tag": "walk_n", "frames": [render_frame(32, 0, cols), render_frame(32, 2, cols)]},
        {"tag": "jump_s", "frames": [render_frame(32, 6, cols)]},
    ]
    res = sb.compose(rows, supersample=4, max_colors=8, outline="sharp", fps=10, name="unit")
    rep = res["report"]
    assert rep["frames_out"] == 5 and rep["rows"] == 3
    assert [t["name"] for t in rep["tags"]] == ["walk_s", "walk_n", "jump_s"]
    assert rep["tags"][1] == {"name": "walk_n", "from": 2, "to": 3}
    assert rep["palette_colors"] and len(rep["palette_colors"]) <= 8
    # one crop box for everything: the jump's raised head sets the height for ALL frames
    frames = [load(f) for f in res["frames"]]
    assert len({f.shape for f in frames}) == 1
    # the feet stay on the same bottom row in every frame (ground line never moves)
    def feet_row(f):
        ys = np.nonzero(f[..., 3])[0]
        return ys.max()
    assert len({feet_row(f) for f in frames}) == 1
    atlas = res["atlas"]
    assert atlas["meta"]["frameTags"][2]["name"] == "jump_s"
    assert atlas["meta"]["layout"]["columns"] == 2  # widest row
    assert set(res["row_gifs"]) == {"walk_s", "walk_n", "jump_s"}
    assert json.loads(res["atlas_json"])["frames"][0]["duration"] == 100
    assert res["sheet"][:8] == b"\x89PNG\r\n\x1a\n" and res["gif"][:3] == b"GIF"


def test_compose_custom_and_named_palettes():
    cols = [(200, 40, 40, 255), (40, 200, 40, 255), (40, 40, 220, 255)]
    rows = [{"tag": "s", "frames": [render_frame(24, 0, cols)]}]
    res = sb.compose(rows, supersample=4, palette_colors=["#ff0000", "#00ff00", "#0000ff"], outline="none")
    assert res["report"]["palette_colors"] == ["#ff0000", "#00ff00", "#0000ff"]
    assert res["report"]["unique_colors"] <= 3
    res = sb.compose(rows, supersample=4, palette="pico8")
    assert res["report"]["palette"] == "pico8"
    with pytest.raises(ValueError, match="Unknown palette"):
        sb.compose(rows, supersample=4, palette="not-a-palette")
    with pytest.raises(ValueError):
        sb.compose([], supersample=4)


def test_render_spec_and_bake_report():
    spec = sb.render_spec(cell=48, supersample=4, directions=8, frames=6, loop=False, actions=["walk"])
    assert spec["render_px"] == 192 and spec["loop"] is False and spec["actions"] == ["walk"]
    bs.normalize_spec(spec)  # must be a valid Blender-side spec
    m = {"directions": 8, "frames_rendered": 48, "blender": "4.5.13 LTS", "rows": []}
    assert sb.bake_report([m]) == {"directions": 8, "frames_rendered": 48, "blender": "4.5.13 LTS"}
    assert isinstance(sb.bake_report([m, m]), list) and len(sb.bake_report([m, m])) == 2


def test_blender_script_exists_and_is_stdlib_safe():
    assert os.path.isfile(sb.BAKE_SCRIPT)
    src = open(sb.BAKE_SCRIPT, encoding="utf-8").read()
    assert "BAKE_RESULT" in src and "from backend import bake_spec" in src
    # the Blender-side script must not pull the service's heavy deps into Blender's Python
    assert "import numpy" not in src and "from PIL" not in src


# ---------------------------------------------------------------------------
# Equipment sockets (attach) — the zero-drift variant route
# ---------------------------------------------------------------------------

def test_attach_normalizes_with_defaults():
    spec = bs.normalize_spec({"attach": [{"model": "/a/axe.glb", "bone": "handslot.r"}]})
    assert spec["attach"] == [{"model": "/a/axe.glb", "bone": "handslot.r",
                               "scale": 1.0, "offset": [0.0, 0.0, 0.0],
                               "rotation": [0.0, 0.0, 0.0]}]


def test_attach_requires_model_and_bone():
    for bad in ({"bone": "handslot.r"}, {"model": "a.glb"}, {"model": "", "bone": "b"}):
        with pytest.raises(ValueError):
            bs.normalize_spec({"attach": [bad]})
    with pytest.raises(ValueError):
        bs.normalize_spec({"attach": "not-a-list"})


def test_attach_defaults_to_empty():
    assert bs.normalize_spec({})["attach"] == []


def test_framing_excludes_attachments_by_default():
    """The camera must fit the BODY only. If equipment drove the bounds, a big
    weapon would re-frame the shot and the same character would land on
    different pixels in the armed and unarmed bakes — which is exactly the
    drift socket variants exist to avoid."""
    assert bs.normalize_spec({})["frame_attachments"] is False
    assert bs.normalize_spec({"frame_attachments": True})["frame_attachments"] is True


def test_render_spec_carries_attach_and_lighting():
    spec = sb.render_spec(cell=48, attach=[{"model": "a.glb", "bone": "handslot.r"}],
                          key_strength=0.6, ambient=1.6)
    assert spec["attach"][0]["bone"] == "handslot.r"
    norm = bs.normalize_spec(spec)
    assert norm["key_strength"] == 0.6 and norm["ambient"] == 1.6


def test_render_spec_leaves_lighting_at_defaults_when_unset():
    norm = bs.normalize_spec(sb.render_spec())
    assert norm["key_strength"] == bs.DEFAULT_SPEC["key_strength"]
    assert norm["ambient"] == bs.DEFAULT_SPEC["ambient"]


def test_attachment_bytes_are_written_beside_the_model(tmp_path):
    """Socket meshes arrive as bytes from a URL; they must be materialised as
    files BEFORE normalize_spec, which rebuilds each entry from known keys only
    and would otherwise drop the private payload."""
    work = str(tmp_path)
    spec = sb.render_spec(attach=[{"model": "axe.glb", "bone": "handslot.r",
                                   "_bytes": b"BLOB", "_ext": ".gltf"}])
    with pytest.raises(Exception):
        sb.run_blender_sync(b"glb", spec, runner=("python", "/bin/false"), keep_dir=work)
    written = json.loads((tmp_path / "spec.json").read_text())
    assert written["attach"][0]["model"].endswith("attach_0.gltf")
    assert (tmp_path / "attach_0.gltf").read_bytes() == b"BLOB"
