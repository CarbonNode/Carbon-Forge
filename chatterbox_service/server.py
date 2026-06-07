"""Chatterbox TTS micro-service (isolated GPU container — NOT a ComfyUI node, so its
pinned transformers==4.46.3 can't conflict with ComfyUI's image/video stack).

Forge calls POST /tts; presence/overflow routing across boxes lives in Forge. Open on
the LAN like the ComfyUI backends (unguessable port, internal only)."""
import base64
import io
import os
import tempfile
import threading

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS

app = FastAPI(title="forge-chatterbox")
_model = None
_lock = threading.Lock()


def get_model():
    """Lazy-load once (first request pays the model download/load; then warm)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                _model = ChatterboxTTS.from_pretrained(device=dev)
    return _model


class TTSReq(BaseModel):
    text: str
    exaggeration: float = 0.5          # Chatterbox emotion/intensity knob
    cfg_weight: float = 0.5            # pacing/adherence
    audio_prompt_b64: str | None = None  # reference wav (base64) for zero-shot voice cloning


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(),
            "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "model_loaded": _model is not None}


@app.post("/tts")
def tts(req: TTSReq):
    if not (req.text or "").strip():
        raise HTTPException(400, "text is required")
    ref_path = None
    try:
        if req.audio_prompt_b64:
            data = base64.b64decode(req.audio_prompt_b64)
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.write(data)
            tf.close()
            ref_path = tf.name
        m = get_model()
        with _lock:  # single GPU job at a time (Forge overflow-routes concurrency across boxes)
            wav = m.generate(req.text, audio_prompt_path=ref_path,
                             exaggeration=float(req.exaggeration), cfg_weight=float(req.cfg_weight))
        buf = io.BytesIO()
        ta.save(buf, wav.cpu(), m.sr, format="wav")
        return Response(content=buf.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — surface a readable error to Forge
        raise HTTPException(500, f"chatterbox generation failed: {e}")
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)
