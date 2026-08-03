import asyncio
import dataclasses
import json
import os

from forge_mcp import storage, worldgen
from forge_mcp.config import load_config


def make_cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        token="t",
        workspace_root=str(tmp_path / "ws"),
        workspace_display_root="C:\\Workspace",
        results_root=str(tmp_path / "results"),
        public_url="https://forge.example.com",
    )


# ---- parse_detections ----

def test_parse_detections_clamps_and_dedupes():
    raw = {"objects": [
        {"label": "Old Oak Tree!", "category": "obstacle", "box_2d": [100, 100, 400, 300]},
        {"label": "Old Oak Tree!", "category": "obstacle", "box_2d": [-50, 900, 500, 1400]},
        {"label": "bad box", "category": "obstacle", "box_2d": [400, 300, 100, 100]},   # inverted
        {"label": "speck", "category": "decor", "box_2d": [10, 10, 11, 11]},            # degenerate
        {"label": "ghost", "category": "not_a_category", "box_2d": [0, 0, 100, 100]},
        {"label": "three", "category": "obstacle", "box_2d": [1, 2, 3]},                # wrong len
        "not a dict",
    ], "player_spawn": [1200, -5]}
    out = worldgen.parse_detections(raw)
    labels = [o["label"] for o in out["objects"]]
    assert labels == ["old_oak_tree", "old_oak_tree_2"]
    assert out["objects"][1]["box_2d"] == [0, 900, 500, 1000]  # clamped into 0-1000
    assert out["player_spawn"] == [1000, 0]


def test_parse_detections_max_objects_and_missing_spawn():
    raw = {"objects": [
        {"label": f"rock_{i}", "category": "obstacle", "box_2d": [0, 0, 50, 50]}
        for i in range(30)
    ]}
    out = worldgen.parse_detections(raw, max_objects=10)
    assert len(out["objects"]) == 10
    assert out["player_spawn"] is None


# ---- box_to_px ----

def test_box_to_px_pads_and_clamps():
    left, top, right, bottom = worldgen.box_to_px([0, 0, 1000, 1000], 1600, 900)
    assert (left, top, right, bottom) == (0, 0, 1600, 900)
    left, top, right, bottom = worldgen.box_to_px([500, 500, 600, 600], 1000, 1000, pad_frac=0.0)
    assert (left, top) == (498, 498) and (right, bottom) == (602, 602)  # min 2px pad


# ---- build_collision ----

def test_collision_obstacle_blocks_only_footprint_band():
    objects = [{"label": "tree", "category": "obstacle", "box_2d": [0, 0, 1000, 100]}]
    grid = worldgen.build_collision(objects, cols=10, rows=10, band_frac=0.30)
    blocked_rows = {i // 10 for i in grid["blocked"]}
    assert blocked_rows == {7, 8, 9}          # bottom 30% only
    assert {i % 10 for i in grid["blocked"]} == {0}


def test_collision_zone_blocks_fully_and_decor_never():
    objects = [
        {"label": "lake", "category": "zone_blocked", "box_2d": [0, 0, 1000, 500]},
        {"label": "path", "category": "decor", "box_2d": [0, 500, 1000, 1000]},
        {"label": "door", "category": "enterable", "box_2d": [0, 500, 1000, 1000]},
    ]
    grid = worldgen.build_collision(objects, cols=10, rows=10)
    assert len(grid["blocked"]) == 50          # left half only
    assert all(i % 10 < 5 for i in grid["blocked"])


def test_collision_indices_in_range():
    objects = [{"label": "wall", "category": "zone_blocked", "box_2d": [990, 990, 1000, 1000]}]
    grid = worldgen.build_collision(objects, cols=64, rows=36)
    assert grid["blocked"] and all(0 <= i < 64 * 36 for i in grid["blocked"])


# ---- manifest + preview ----

def _manifest():
    objects = [
        {"label": "well", "category": "obstacle", "box_2d": [100, 100, 300, 200]},
        {"label": "path", "category": "decor", "box_2d": [0, 0, 1000, 1000]},
    ]
    sprites = {"well": {"file": "sprite_well.png", "crop_px": [90, 95, 310, 205]}}
    grid = worldgen.build_collision(objects, cols=16, rows=9)
    return worldgen.build_manifest("test-world", "map.png", 1600, 900,
                                   objects, [500, 500], grid, sprites)


def test_manifest_wires_sprites_and_collision():
    m = _manifest()
    assert m["format"] == "carbon-forge-world/1"
    well = next(o for o in m["objects"] if o["label"] == "well")
    assert well["sprite"] == "sprite_well.png"
    assert well["crop_px"] == [90, 95, 310, 205]
    path = next(o for o in m["objects"] if o["label"] == "path")
    assert "sprite" not in path
    assert m["collision"]["cols"] == 16
    assert m["player_spawn"] == [500, 500]


def test_preview_embeds_manifest_and_relative_refs():
    html = worldgen.render_preview_html(_manifest())
    assert "sprite_well.png" in html
    assert '"map.png"' in html
    assert "__WORLD_JSON__" not in html
    # embedded JSON must round-trip
    start = html.index("const WORLD = ") + len("const WORLD = ")
    end = html.index(";\n", start)
    assert json.loads(html[start:end])["name"] == "test-world"


# ---- save_bundle ----

PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 64


def test_save_bundle_one_id_and_workspace(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    out = asyncio.run(storage.save_bundle(
        [("map.png", PNG), ("world.json", b"{}"), ("preview.html", b"<html>")],
        project="Proj", subpath="assets/forge/worlds/w1", cfg=cfg))
    # one cache id serves all three siblings
    ids = {u.rsplit("/", 2)[1] for u in out["files"].values()}
    assert len(ids) == 1
    file_id = ids.pop()
    for name in ("map.png", "world.json", "preview.html"):
        assert os.path.isfile(os.path.join(cfg.results_root, "files", file_id, name))
        # workspace copies keep exact names (relative refs must hold)
        assert os.path.isfile(os.path.join(cfg.workspace_root, "Proj",
                                           "assets", "forge", "worlds", "w1", name))
    assert out["base_url"].endswith(file_id)
    assert out["workspace_dir"].startswith("C:\\Workspace\\Proj")


def test_save_bundle_missing_project_keeps_cache(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(cfg.workspace_root)
    out = asyncio.run(storage.save_bundle([("a.png", PNG)], project="Nope",
                                          subpath=None, cfg=cfg))
    assert "workspace_write_error" in out
    assert list(out["files"]) == ["a.png"]


# ---- world_prompt ----

def test_world_prompt_style_leads():
    p = worldgen.world_prompt("a spooky forest village")
    assert p.startswith("A 16-bit pixel art")
    assert "a spooky forest village" in p
    assert "screenshot of a 2D game" in p
    p2 = worldgen.world_prompt("a beach", "watercolor storybook illustration")
    assert p2.startswith("A watercolor storybook illustration")
