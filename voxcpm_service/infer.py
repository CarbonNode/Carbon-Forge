"""One request per process. Exiting releases all model tensors and the CUDA context."""
import json
import os
from pathlib import Path
import subprocess
import sys
import wave


def generate(directory):
    import soundfile as sf
    import numpy as np
    import random
    import torch
    from voxcpm import VoxCPM

    torch.set_num_threads(2)
    d = Path(directory)
    request = json.loads((d / "request.json").read_text())
    reference = d / "reference.input"
    kwargs = {}
    if reference.exists():
        wav = d / "reference.wav"
        # Never let a reference container act as a playlist fetching network resources.
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
                        "-i", str(reference), "-t", "31", "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(wav)], check=True, capture_output=True, timeout=20)
        with wave.open(str(wav)) as source:
            duration = source.getnframes() / source.getframerate()
        if not 1 <= duration <= 30:
            raise ValueError("Use 1–30 seconds of clean reference speech")
        kwargs["reference_wav_path"] = str(wav)
        if request.get("prompt_text"):
            kwargs.update(prompt_wav_path=str(wav), prompt_text=request["prompt_text"])
    text = request["text"].strip()
    if request.get("voice_description"):
        text = f"({request['voice_description'].strip()}){text}"
    if request.get("seed") is not None:
        random.seed(request["seed"])
        np.random.seed(request["seed"])
        torch.manual_seed(request["seed"])
    model = VoxCPM.from_pretrained(os.environ.get("VOXCPM_MODEL_DIR", "/models/voxcpm2"),
                                   load_denoiser=False, optimize=False, device="cuda")
    audio = model.generate(text=text, cfg_value=request["cfg_value"],
                           inference_timesteps=request["inference_timesteps"],
                           max_len=1200, retry_badcase=False, **kwargs)
    if len(audio) == 0:
        raise ValueError("VoxCPM returned empty audio")
    sf.write(str(d / "speech.wav"), audio, model.tts_model.sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    try:
        generate(sys.argv[1])
    except Exception as exc:
        # Keep model diagnostics off the public API. Reference transcripts are not logged.
        message = str(exc) if isinstance(exc, ValueError) else f"VoxCPM inference failed ({type(exc).__name__})"
        Path(sys.argv[1], "error.txt").write_text(message)
        raise SystemExit(1)
