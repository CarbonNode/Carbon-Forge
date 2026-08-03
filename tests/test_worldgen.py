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


def test_collision_enterable_carves_door_through_obstacle():
    objects = [
        # building whose footprint band is the bottom rows
        {"label": "tavern", "category": "obstacle", "box_2d": [0, 0, 1000, 1000]},
        # door in the middle of the facade
        {"label": "tavern_door", "category": "enterable", "box_2d": [800, 400, 1000, 600]},
    ]
    grid = worldgen.build_collision(objects, cols=10, rows=10, band_frac=0.30)
    # door columns (4-5) are carved open in the footprint rows; others stay blocked
    blocked_cols_bottom = {i % 10 for i in grid["blocked"] if i // 10 == 8}
    assert 4 not in blocked_cols_bottom and 5 not in blocked_cols_bottom
    assert 0 in blocked_cols_bottom and 9 in blocked_cols_bottom


def test_collision_indices_in_range():
    objects = [{"label": "wall", "category": "zone_blocked", "box_2d": [990, 990, 1000, 1000]}]
    grid = worldgen.build_collision(objects, cols=64, rows=36)
    assert grid["blocked"] and all(0 <= i < 64 * 36 for i in grid["blocked"])


# ---- manifest (v2) + preview ----

def _world_manifest():
    main_objects = [
        {"label": "well", "category": "obstacle", "box_2d": [100, 100, 300, 200]},
        {"label": "shed_door", "category": "enterable", "box_2d": [200, 400, 300, 500],
         "link": {"to": "shed", "spawn": [880, 500]}},
        {"label": "path", "category": "decor", "box_2d": [0, 0, 1000, 1000]},
    ]
    sprites = {"well": {"file": "sprite_main_well.png", "crop_px": [90, 95, 310, 205]}}
    grid = worldgen.build_collision(main_objects, cols=16, rows=9)
    main = worldgen.build_map_entry("map_main.png", 1600, 900, main_objects,
                                    [500, 500], grid, sprites)
    shed_objects = [
        {"label": "exit_door", "category": "enterable", "box_2d": [900, 400, 1000, 600],
         "link": {"to": "main", "spawn": [330, 450]}},
    ]
    shed = worldgen.build_map_entry("map_shed.png", 1408, 768, shed_objects, [880, 500], None)
    return worldgen.build_world_manifest("test-world", {"main": main, "shed": shed})


def test_manifest_v2_wires_sprites_links_collision():
    m = _world_manifest()
    assert m["format"] == "carbon-forge-world/2"
    assert m["start"] == "main"
    main = m["maps"]["main"]
    well = next(o for o in main["objects"] if o["label"] == "well")
    assert well["sprite"] == "sprite_main_well.png"
    assert well["crop_px"] == [90, 95, 310, 205]
    door = next(o for o in main["objects"] if o["label"] == "shed_door")
    assert door["link"] == {"to": "shed", "spawn": [880, 500]}
    path = next(o for o in main["objects"] if o["label"] == "path")
    assert "sprite" not in path and "link" not in path
    assert main["collision"]["cols"] == 16
    exit_door = m["maps"]["shed"]["objects"][0]
    assert exit_door["link"]["to"] == "main"


def test_manifest_start_falls_back_to_first_map():
    entry = worldgen.build_map_entry("map_x.png", 100, 100, [], None, None)
    m = worldgen.build_world_manifest("w", {"cave": entry}, start="main")
    assert m["start"] == "cave"


def test_preview_embeds_manifest_and_relative_refs():
    html = worldgen.render_preview_html(_world_manifest())
    assert "sprite_main_well.png" in html
    assert '"map_main.png"' in html and '"map_shed.png"' in html
    assert "__WORLD_JSON__" not in html
    # embedded JSON must round-trip
    start = html.index("const WORLD = ") + len("const WORLD = ")
    end = html.index(";\n", start)
    assert json.loads(html[start:end])["name"] == "test-world"


# ---- multi-map linking helpers ----

def test_pick_expandable_largest_doors_first():
    objects = [
        {"label": "small_door", "category": "enterable", "box_2d": [0, 0, 10, 10]},
        {"label": "big_door", "category": "enterable", "box_2d": [0, 0, 200, 200]},
        {"label": "tree", "category": "obstacle", "box_2d": [0, 0, 500, 500]},
    ]
    picked = worldgen.pick_expandable(objects, 1)
    assert [o["label"] for o in picked] == ["big_door"]
    assert worldgen.pick_expandable(objects, 0) == []


def test_map_key_for_strips_door_words():
    assert worldgen.map_key_for("brewing_shed_door") == "brewing_shed"
    assert worldgen.map_key_for("tavern_door_2") == "tavern_2"
    assert worldgen.map_key_for("cave_mouth") == "cave"
    assert worldgen.map_key_for("door") == "interior"


def test_door_return_spawn_below_box():
    assert worldgen.door_return_spawn([200, 400, 300, 500]) == [330, 450]
    assert worldgen.door_return_spawn([900, 0, 990, 100]) == [985, 50]  # clamped


def test_interior_entry_exit_prefers_bottom_center_door():
    objects = [
        {"label": "window", "category": "enterable", "box_2d": [100, 400, 200, 500]},
        {"label": "exit_door", "category": "enterable", "box_2d": [850, 450, 980, 560]},
    ]
    spawn, exit_label = worldgen.interior_entry_exit(objects)
    assert exit_label == "exit_door"
    assert spawn == [805, 505]  # just above the door


def test_interior_entry_exit_fallback_synthesizes():
    spawn, exit_label = worldgen.interior_entry_exit(
        [{"label": "rug", "category": "decor", "box_2d": [400, 400, 600, 600]}])
    assert exit_label is None
    assert spawn == [880, 500]
    exit_obj = worldgen.synthesize_exit()
    assert exit_obj["category"] == "enterable"
    assert exit_obj["box_2d"][0] >= 900  # bottom strip


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
