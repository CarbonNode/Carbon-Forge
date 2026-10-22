"""Authenticated local TTS. A child owns the CUDA context and exits after every line.

This prevents a cold, unused speech model from retaining VRAM on a shared gaming PC.
No synthesis retry, unbounded queue, model load on health checks, or cloud fallback.
"""
import asyncio
import base64
import binascii
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

MODEL = "openbmb/VoxCPM2"
MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "/models/voxcpm2")
MAX_REF = 10 * 1024 * 1024
app = FastAPI(title="forge-voxcpm")
_busy = False


class TTSReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice_description: str | None = Field(default=None, max_length=500)
    prompt_text: str | None = Field(default=None, max_length=2000)
    cfg_value: float = Field(default=2.0, ge=0.5, le=5, allow_inf_nan=False)
    inference_timesteps: int = Field(default=10, ge=1, le=50, strict=True)
    seed: int | None = Field(default=None, ge=0, le=2147483647, strict=True)
    audio_prompt_b64: str | None = Field(default=None, max_length=4 * ((MAX_REF + 2) // 3))

    @model_validator(mode="after")
    def valid_pair(self):
        if not self.text.strip():
            raise ValueError("text is required")
        if self.prompt_text and (not self.prompt_text.strip() or not self.audio_prompt_b64):
            raise ValueError("Reference transcript requires a reference clip")
        if self.voice_description and any(c in self.voice_description for c in "()\n\r"):
            raise ValueError("Voice description cannot contain parentheses or newlines")
        return self


def gpu_reason():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.free",
                            "--format=csv,noheader,nounits", "--id=0"],
                           capture_output=True, text=True, timeout=3, check=True)
        util, free = [float(v.strip()) for v in r.stdout.strip().split(",")]
        if util >= 50:
            return "GPU is busy; try when the current workload finishes"
        if free < 9000:
            return "VoxCPM2 needs at least 9 GB of free GPU memory"
        return None
    except (OSError, subprocess.SubprocessError, ValueError):
        return "GPU availability could not be verified"


@app.get("/health")
def health():
    reason = "VoxCPM is generating another clip" if _busy else gpu_reason()
    if not Path(MODEL_DIR, ".ready").is_file():
        reason = "VoxCPM2 model download is not complete"
    if not os.environ.get("FORGE_TOKEN"):
        reason = "Worker authentication is not configured"
    return {"ok": True, "ready": reason is None, "reason": reason, "model": MODEL,
            "sample_rate": 48000, "busy": _busy, "model_loaded": _busy}


async def run_inference(directory):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "voxcpm_service.infer", str(directory),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    try:
        await asyncio.wait_for(proc.wait(), timeout=90)
        if proc.returncode:
            error = Path(directory, "error.txt")
            detail = error.read_text()[:300] if error.exists() else "Worker failed to generate audio"
            raise HTTPException(500, detail)
    except asyncio.TimeoutError:
        raise HTTPException(504, "VoxCPM generation timed out; worker stopped, no retry made")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@app.post("/tts")
async def tts(req: TTSReq, authorization: str | None = Header(default=None)):
    global _busy
    token = os.environ.get("FORGE_TOKEN", "")
    if not token or not hmac.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(401, "Unauthorized")
    # No await between checking and claiming this slot. Run exactly one Uvicorn worker.
    if _busy:
        raise HTTPException(409, "VoxCPM is generating another clip; no request was queued")
    _busy = True
    try:
        reason = await asyncio.to_thread(gpu_reason)
        if reason:
            raise HTTPException(503, reason)
        if not Path(MODEL_DIR, ".ready").is_file():
            raise HTTPException(503, "VoxCPM2 model download is not complete")
        with tempfile.TemporaryDirectory(prefix="voxcpm-") as directory:
            payload = req.model_dump(exclude={"audio_prompt_b64"})
            if req.audio_prompt_b64:
                try:
                    reference = base64.b64decode(req.audio_prompt_b64, validate=True)
                except (ValueError, binascii.Error):
                    raise HTTPException(400, "Reference audio is not valid base64")
                if not reference or len(reference) > MAX_REF:
                    raise HTTPException(400, "Reference audio must contain 1 byte to 10 MB")
                await asyncio.to_thread(Path(directory, "reference.input").write_bytes, reference)
            await asyncio.to_thread(Path(directory, "request.json").write_text, json.dumps(payload))
            await run_inference(directory)
            audio = await asyncio.to_thread(Path(directory, "speech.wav").read_bytes)
            return Response(audio, media_type="audio/wav")
    finally:
        _busy = False
