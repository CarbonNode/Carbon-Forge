"""forge_mcp/meshy_api.py — pure helpers (no network)."""
import pytest

from forge_mcp import meshy_api as M
from forge_mcp.generation import GenerationError


def test_resolve_action_names_and_ids():
    assert M.resolve_action("attack") == 4 and M.resolve_action(" Death ") == 8
    assert M.resolve_action("walk") == 30 and M.resolve_action("idle") == 0
    assert M.resolve_action(466) == 466 and M.resolve_action("466") == 466
    with pytest.raises(GenerationError, match="unknown action"):
        M.resolve_action("moonwalk")
    with pytest.raises(GenerationError):
        M.resolve_action(True)


def test_pick_glb_documented_and_nested_shapes():
    assert M.pick_glb("animate", {"result": {"animation_glb_url": "https://x/a.glb?sig=1"}}) == "https://x/a.glb?sig=1"
    assert M.pick_glb("animate", {"animation_glb_url": "https://x/top.glb"}) == "https://x/top.glb"
    assert M.pick_glb("rig", {"result": {"rigged_character_glb_url": "https://x/r.glb"}}) == "https://x/r.glb"
    assert M.pick_glb("image", {"model_urls": {"glb": "https://x/m.glb", "fbx": "https://x/m.fbx"}}) == "https://x/m.glb"
    assert M.pick_glb("text", {"result": {"model_urls": {"glb": "https://x/n.glb"}}}) == "https://x/n.glb"
    # unknown shape: any .glb url anywhere wins, .fbx never does
    assert M.pick_glb("rig", {"weird": {"deep": [{"u": "https://x/f.fbx"}, {"u": "https://x/z.glb"}]}}) == "https://x/z.glb"
    assert M.pick_glb("rig", {"status": "SUCCEEDED"}) is None


def test_rig_basic_animations():
    task = {"result": {"basic_animations": {"walking_glb_url": "https://x/w.glb", "running_glb_url": ""}}}
    assert M.rig_basic_animations(task) == {"walk": "https://x/w.glb"}
    assert M.rig_basic_animations({"result": {}}) == {}
    assert M.rig_basic_animations({"basic_animations": {"running_glb_url": "https://x/r.glb"}}) == {"run": "https://x/r.glb"}


def test_headers_require_key():
    with pytest.raises(GenerationError, match="MESHY_API_KEY"):
        M._headers("")
    assert M._headers("msy_x")["Authorization"] == "Bearer msy_x"


def test_task_id_extraction():
    assert M._task_id({"result": "abc"}) == "abc"
    assert M._task_id({"id": "def"}) == "def"
    with pytest.raises(GenerationError):
        M._task_id({})
