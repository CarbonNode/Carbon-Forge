"""Exercise the paid character pipeline with fake providers and real sheet composition."""
import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from forge_mcp.jobs import JobStore
from forge_mcp.tools import sprite3d


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


@pytest.fixture
def harness(tmp_path, monkeypatch):
    cfg = SimpleNamespace(meshy_api_key="test-key", bpy_python="", blender_bin="",
                          max_video_mb=1)
    jobs = JobStore(str(tmp_path / "jobs.json"))
    ctx = SimpleNamespace(cfg=cfg, jobs=jobs, http=object())
    registry = ToolRegistry()
    sprite3d.register(registry, ctx)
    monkeypatch.setattr(sprite3d.sb, "find_blender", lambda **kw: ("python", "fake-blender"))
    monkeypatch.setattr(sprite3d.storage, "validate_project", lambda *a, **kw: None)
    resolve = AsyncMock(return_value=SimpleNamespace(data=b"fixture", mime="image/png"))
    monkeypatch.setattr(sprite3d.storage, "resolve_input", resolve)
    return SimpleNamespace(ctx=ctx, tools=registry.tools, resolve=resolve)


@pytest.mark.parametrize("settings, expected", [
    ({}, {"key_strength": 3.0, "ambient": 0.55}),
    ({"shading": "flat", "attach": [{"model": "Game/lantern.glb", "bone": "LeftHand"}]},
     {"key_strength": 0.6, "ambient": 1.6}),
    ({"shading": "soft", "key_strength": 0, "ambient": 0,
      "light_azimuth": 0, "light_elevation": 15,
      "attach": [{"model": "Game/lantern.glb", "bone": "LeftHand", "scale": 0.5,
                  "offset": [0, 1, 2], "rotation": [90, 0, 0]}]},
     {"key_strength": 0.0, "ambient": 0.0,
      "light_azimuth_deg": 0.0, "light_elevation_deg": 15.0}),
])
def test_character_controls_reach_every_clip_and_facing(harness, monkeypatch, settings, expected):
    """The free walk and paid attack must use the same socket and lighting settings."""
    monkeypatch.setattr(sprite3d.M, "image_to_3d", AsyncMock(return_value="model-task"))
    monkeypatch.setattr(sprite3d.M, "rig", AsyncMock(return_value="rig-task"))
    animate = AsyncMock(return_value="attack-task")
    monkeypatch.setattr(sprite3d.M, "animate", animate)
    tasks = {
        "image": {"model_urls": {"glb": "https://fixture/model.glb"}},
        "rig": {"result": {"rigged_character_glb_url": "https://fixture/rig.glb",
                           "basic_animations": {"walking_glb_url": "https://fixture/walk.glb"}}},
        "animate": {"result": {"animation_glb_url": "https://fixture/attack.glb"}},
    }

    async def wait_task(client, key, kind, task_id, **kw):
        return tasks[kind]

    async def download(client, url, max_bytes):
        return url.encode()

    async def save_result(data, **kw):
        return {"url": f"https://fixture/{kw['filename']}.{kw['ext']}"}

    monkeypatch.setattr(sprite3d.M, "wait_task", wait_task)
    monkeypatch.setattr(sprite3d.M, "download", download)
    monkeypatch.setattr(sprite3d.storage, "save_result", save_result)
    delivered = AsyncMock(return_value={"kind": "sheet"})
    monkeypatch.setattr(sprite3d, "_deliver", delivered)
    renders = []
    png = io.BytesIO()
    Image.new("RGBA", (16, 16), (120, 60, 30, 255)).save(png, format="PNG")

    async def render_rows(glb, spec, *, action_prefix, runner):
        renders.append((glb, spec, action_prefix))
        tags = sprite3d.sb.bs.direction_tags(spec["directions"])
        manifest = {"actions_baked": [action_prefix], "direction_tags": tags}
        rows = [{"tag": f"{action_prefix}_{tag}", "frames": [png.getvalue()]} for tag in tags]
        return manifest, rows

    monkeypatch.setattr(sprite3d.sb, "render_rows", render_rows)
    spawned = []
    create_task = asyncio.create_task

    def capture_task(coro):
        task = create_task(coro)
        spawned.append(task)
        return task

    monkeypatch.setattr(sprite3d.asyncio, "create_task", capture_task)

    async def run():
        result = await harness.tools["character_to_sprites"](
            project="Game", image="Game/concept.png", actions=["walk", "attack"],
            directions=8, cell=16, supersample=1, frames=1, **settings)
        await asyncio.gather(*spawned)
        return harness.ctx.jobs.get(result["job_id"])

    job = asyncio.run(run())
    assert job["status"] == "done", job["error"]
    assert [r[2] for r in renders] == ["walk", "attack"]
    assert [r[1]["loop"] for r in renders] == [True, False]
    for glb, spec, action in renders:
        assert glb == f"https://fixture/{action}.glb".encode()
        assert spec["directions"] == 8
        for key, value in expected.items():
            assert spec[key] == value
        if settings.get("attach"):
            attachment = spec["attach"][0]
            assert attachment["bone"] == "LeftHand"
            assert attachment["model"] == "Game/lantern.glb"
            assert attachment["_bytes"] == b"fixture" and attachment["_ext"] == ".glb"
            normalized = sprite3d.sb.bs.normalize_spec(spec)["attach"]
            assert normalized == sprite3d.sb.bs.normalize_spec(settings)["attach"]
        else:
            assert spec["attach"] == []
    animate.assert_awaited_once()
    delivered.assert_awaited_once()
    report = delivered.await_args.args[0]["report"]
    assert report["rows"] == 16 and report["frames_out"] == 16
    assert [tag["name"] for tag in report["tags"]] == [
        f"{action}_{direction}" for action in ("walk", "attack")
        for direction in ("s", "se", "e", "ne", "n", "nw", "w", "sw")]
    if settings.get("attach"):
        # Download once, then reuse the same bytes in all clips. Do not mutate caller input.
        assert sum(c.args[0] == "Game/lantern.glb" for c in harness.resolve.await_args_list) == 1
        assert "_bytes" not in settings["attach"][0]


@pytest.mark.parametrize("settings, error", [
    ({"shading": "toon"}, "shading must be"),
    ({"attach": [{"model": "Game/lantern.glb", "bone": " "}]}, "bone is required"),
    ({"attach": [{"model": "Game/lantern.glb", "bone": "LeftHand"},
                 {"model": "Game/shield.glb", "bone": "RightHand", "offset": [0, 1]}]},
     "offset/rotation"),
])
def test_invalid_controls_fail_before_downloads_or_paid_job(harness, monkeypatch, settings, error):
    create = Mock(side_effect=AssertionError("must not start a paid job"))
    monkeypatch.setattr(harness.ctx.jobs, "create", create)
    with pytest.raises(ValueError, match=error):
        asyncio.run(harness.tools["character_to_sprites"](
            project="Game", image="Game/concept.png", **settings))
    create.assert_not_called()
    harness.resolve.assert_not_awaited()


def test_missing_attachment_fails_before_paid_job(harness, monkeypatch):
    harness.resolve.side_effect = ValueError("attachment file missing")
    create = Mock(side_effect=AssertionError("must not start a paid job"))
    monkeypatch.setattr(harness.ctx.jobs, "create", create)
    with pytest.raises(ValueError, match="attachment file missing"):
        asyncio.run(harness.tools["character_to_sprites"](
            project="Game", image="Game/concept.png",
            attach=[{"model": "Game/missing.glb", "bone": "LeftHand"}]))
    create.assert_not_called()


def test_mcp_schema_exposes_optional_character_controls(harness):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("sprite-controls-test")
    sprite3d.register(mcp, harness.ctx)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    schema = tools["character_to_sprites"].inputSchema
    for name in ("shading", "attach", "key_strength", "ambient", "light_azimuth", "light_elevation"):
        assert name in schema["properties"]
        assert name not in schema.get("required", [])
    assert schema["properties"]["shading"]["default"] == "lit"
