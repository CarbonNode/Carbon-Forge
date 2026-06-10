import asyncio
import dataclasses
import os
import time

import pytest

from forge_mcp import storage
from forge_mcp.config import load_config


def make_cfg(tmp_path):
    # Built on load_config() defaults so a new Config field can't break this fixture again
    # (the fixed kwarg list here went stale every time Config grew).
    return dataclasses.replace(
        load_config(),
        token="t",
        workspace_root=str(tmp_path / "ws"),
        workspace_display_root="C:\\Workspace",
        results_root=str(tmp_path / "results"),
        public_url="https://forge.example.com",
    )


PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 64
JPG = bytes.fromhex("ffd8ffe0") + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def test_sniff():
    assert storage.sniff_mime(PNG) == "image/png"
    assert storage.sniff_mime(JPG) == "image/jpeg"
    assert storage.sniff_mime(MP4) == "video/mp4"
    assert storage.sniff_mime(b"plain text here") is None


def test_workspace_input_traversal_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    with pytest.raises(storage.StorageError, match="traversal|invalid"):
        asyncio.run(storage.resolve_input("Proj/../../etc/passwd", cfg=cfg))


def test_workspace_input_reads_file(tmp_path):
    cfg = make_cfg(tmp_path)
    d = os.path.join(cfg.workspace_root, "Proj", "img")
    os.makedirs(d)
    with open(os.path.join(d, "a.png"), "wb") as f:
        f.write(PNG)
    r = asyncio.run(storage.resolve_input("Proj/img/a.png", cfg=cfg))
    assert r.mime == "image/png" and r.data == PNG


def test_validate_project_close_match(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "PerennaMono"))
    with pytest.raises(storage.StorageError, match="PerennaMono"):
        storage.validate_project("perenamono", cfg=cfg)


def test_save_result_writes_both_and_mints_url(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    res = asyncio.run(storage.save_result(
        PNG, project="Proj", subpath=None, filename="hero", ext="png", cfg=cfg))
    assert res["workspace_path"] == "C:\\Workspace\\Proj\\assets\\forge\\hero.png"
    assert os.path.isfile(os.path.join(cfg.workspace_root, "Proj", "assets", "forge", "hero.png"))
    assert res["url"].startswith("https://forge.example.com/files/")
    fid = res["url"].split("/files/")[1].split("/")[0]
    assert os.path.isfile(os.path.join(cfg.results_root, "files", fid, "hero.png"))
    assert "workspace_write_error" not in res


def test_save_result_collision_suffix(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(os.path.join(cfg.workspace_root, "Proj"))
    r1 = asyncio.run(storage.save_result(PNG, project="Proj", subpath=None, filename="x", ext="png", cfg=cfg))
    r2 = asyncio.run(storage.save_result(PNG, project="Proj", subpath=None, filename="x", ext="png", cfg=cfg))
    assert r1["workspace_path"].endswith("x.png")
    assert r2["workspace_path"].endswith("x-2.png")


def test_save_result_workspace_down_degrades(tmp_path):
    cfg = make_cfg(tmp_path)  # workspace_root never created -> write fails
    res = asyncio.run(storage.save_result(PNG, project="Proj", subpath=None, filename="y", ext="png", cfg=cfg))
    assert res["url"].startswith("https://")
    assert "workspace_write_error" in res


def test_janitor_prunes_old(tmp_path):
    cfg = make_cfg(tmp_path)
    old_dir = os.path.join(cfg.results_root, "files", "oldid123")
    os.makedirs(old_dir)
    with open(os.path.join(old_dir, "a.png"), "wb") as f:
        f.write(PNG)
    past = time.time() - 90 * 86400
    os.utime(old_dir, (past, past))
    new_res = asyncio.run(storage.save_result(PNG, project=None, subpath=None, filename="keep", ext="png", cfg=cfg))
    storage.prune_cache_once(cfg)
    assert not os.path.isdir(old_dir)
    kept_id = new_res["url"].split("/files/")[1].split("/")[0]
    assert os.path.isdir(os.path.join(cfg.results_root, "files", kept_id))
