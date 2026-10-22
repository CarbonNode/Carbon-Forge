import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from forge_mcp import voxcpm
from forge_mcp.generation import GenerationError
from voxcpm_service import server

WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"test audio"


def test_adapter_sends_design_and_transcript_without_retry():
    calls = []
    def handle(request):
        import json
        calls.append(request)
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer test-only"
        assert body["audio_prompt_b64"] == base64.b64encode(b"reference").decode()
        assert body["voice_description"] == "Warm, calm narrator"
        assert body["prompt_text"] == "The reference words."
        assert body["seed"] == 0
        return httpx.Response(200, content=WAV)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            cfg = SimpleNamespace(voxcpm_url="http://worker", voxcpm_token="test-only")
            assert await voxcpm.synthesize(client, cfg, "Hello", reference=b"reference",
                voice_description="Warm, calm narrator", prompt_text="The reference words.", seed=0) == WAV
    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("status,content", [(503, b"GPU busy"), (200, b"<html>error</html>")])
def test_adapter_refuses_error_or_non_audio_and_does_not_retry(status, content):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, content=content)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(GenerationError):
                await voxcpm.synthesize(client, SimpleNamespace(voxcpm_url="http://worker", voxcpm_token="test"), "Hello")
    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("args", [dict(model="some-model"), dict(prompt_text="words"),
    dict(cfg_value=float("nan")), dict(inference_timesteps=0), dict(seed=-1),
    dict(voice_description="bad)spoken text(")])
def test_invalid_generation_settings(args):
    with pytest.raises(GenerationError):
        voxcpm.validate("Hello", **args)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGE_TOKEN", "test-only")
    monkeypatch.setattr(server, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(server, "gpu_reason", lambda: None)
    (tmp_path / ".ready").write_text("test")
    monkeypatch.setattr(server, "_busy", False)
    with TestClient(server.app) as client:
        yield client


AUTH = {"authorization": "Bearer test-only"}


def test_worker_auth_busy_and_validation_do_not_run_inference(worker, monkeypatch):
    async def never(_):
        pytest.fail("inference must not start")
    monkeypatch.setattr(server, "run_inference", never)
    assert worker.post("/tts", json={"text": "Hi"}).status_code == 401
    assert worker.post("/tts", json={"text": "Hi", "prompt_text": "reference"}, headers=AUTH).status_code == 422
    assert worker.post("/tts", json={"text": "Hi", "audio_prompt_b64": "!!!"}, headers=AUTH).status_code == 400
    monkeypatch.setattr(server, "gpu_reason", lambda: "GPU busy")
    assert worker.post("/tts", json={"text": "Hi"}, headers=AUTH).status_code == 503
    assert server._busy is False
    monkeypatch.setattr(server, "_busy", True)
    assert worker.post("/tts", json={"text": "Hi"}, headers=AUTH).status_code == 409


def test_worker_passes_reference_and_cleans_up(worker, monkeypatch):
    directories = []
    async def infer(directory):
        import json
        d = Path(directory)
        directories.append(d)
        assert (d / "reference.input").read_bytes() == b"my voice"
        request = json.loads((d / "request.json").read_text())
        assert request["voice_description"] == "Calm"
        assert "audio_prompt_b64" not in request
        (d / "speech.wav").write_bytes(WAV)
    monkeypatch.setattr(server, "run_inference", infer)
    result = worker.post("/tts", json={"text": "Hello", "voice_description": "Calm",
                         "audio_prompt_b64": base64.b64encode(b"my voice").decode()}, headers=AUTH)
    assert result.status_code == 200
    assert result.headers["content-type"] == "audio/wav"
    assert result.content == WAV
    assert server._busy is False
    assert all(not d.exists() for d in directories)


def test_worker_releases_slot_on_failure(worker, monkeypatch):
    from fastapi import HTTPException
    async def fail(_):
        raise HTTPException(504, "timed out")
    monkeypatch.setattr(server, "run_inference", fail)
    assert worker.post("/tts", json={"text": "Hello"}, headers=AUTH).status_code == 504
    assert server._busy is False
