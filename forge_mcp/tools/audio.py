"""Audio / TTS generation. Two providers: ElevenLabs (cloud, expressive, large voice
library) and Chatterbox (local, on the 4090 GPU container(s) — free, on-prem, zero-shot
voice cloning; primary→overflow routing, yields only while the GPU is actually busy —
someone merely being at the box does not block a synth)."""

from forge_mcp import generation as g
from forge_mcp import storage
from forge_mcp import voxcpm


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def generate_speech(text: str, project: str, provider: str = "elevenlabs",
                              voice: str | None = None, model: str | None = None,
                              stability: float = 0.5, similarity_boost: float = 0.75,
                              style: float = 0.0, exaggeration: float = 0.5,
                              cfg_weight: float = 0.5, subpath: str | None = None,
                              filename: str | None = None,
                              voice_description: str | None = None, prompt_text: str | None = None,
                              cfg_value: float = 2.0, inference_timesteps: int = 10,
                              seed: int | None = None) -> dict:
        """Generate speech audio (TTS) from text, saved into the project's workspace folder
        (and a shareable cache URL), same I/O contract as the image/video tools.

        provider:
          - 'voxcpm' — local VoxCPM2: 30 languages, 48kHz WAV, voice design and cloning.
            voice is an optional reference clip (workspace path or URL). voice_description
            describes a voice or its delivery. prompt_text is the clip's exact transcript
            for continuation cloning; omit it for reference-only cloning. model is
            openbmb/VoxCPM2. cfg_value (0.5–5), inference_timesteps (1–50), seed are optional.
            Maximum 2000 text characters, 10MB reference, 1–30 seconds of clean reference speech.
            A busy GPU returns an error; no paid provider is chosen automatically.
          - 'elevenlabs' (default) — cloud, expressive, large prebuilt voice library + cloning.
            Needs ELEVENLABS_API_KEY on the service. Use list_voices() to discover voice ids.
          - 'chatterbox' — local, on the 4090 GPU container(s); free, on-prem, zero-shot
            voice cloning + emotion control. Routes primary→overflow; yields only while
            the GPU is genuinely busy (util >= 50% or VRAM full), never on mere presence.
        voice: for elevenlabs = voice_id (omit → the account's own first voice, since free-tier
          keys can't use Voice-Library voices; see list_voices); for chatterbox =
          a workspace path / URL to a reference clip to clone (omit for the default voice).
        model: ElevenLabs model_id (default eleven_multilingual_v2; 'eleven_turbo_v2_5' = low latency).
        stability / similarity_boost / style: ElevenLabs voice_settings (0..1).
        exaggeration / cfg_weight: Chatterbox emotion intensity / pacing (0..1).
        Returns the saved audio result (workspace path + url)."""
        if provider == "elevenlabs":
            audio = await g.call_elevenlabs(
                ctx.http, cfg.elevenlabs_api_key, text, voice_id=voice, model_id=model,
                stability=stability, similarity_boost=similarity_boost, style=style)
            ext = "mp3"
        elif provider == "chatterbox":
            # Local, on the 4090 GPU container(s) — zero-shot voice cloning + emotion control.
            # For chatterbox, `voice` is a workspace path / URL to a reference clip to clone;
            # exaggeration (emotion intensity) + cfg_weight (pacing) tune the delivery.
            ref_bytes = None
            if voice:
                ref = await storage.resolve_input(voice, cfg=cfg, kind="audio")
                ref_bytes = ref.data
            url, label = await g.select_chatterbox(ctx.http, cfg)
            if not url:
                raise g.GenerationError(f"No local Chatterbox backend available: {label}")
            audio = await g.call_chatterbox(ctx.http, url, text, exaggeration=exaggeration,
                                            cfg_weight=cfg_weight, audio_prompt_bytes=ref_bytes)
            ext = "wav"
        elif provider == "voxcpm":
            voxcpm.validate(text, model, voice_description, prompt_text, voice,
                            cfg_value, inference_timesteps, seed)
            ref = await storage.resolve_input(voice, cfg=cfg, kind="audio") if voice else None
            audio = await voxcpm.synthesize(
                ctx.http, cfg, text, reference=ref.data if ref else None,
                voice_description=voice_description, prompt_text=prompt_text,
                cfg_value=cfg_value, inference_timesteps=inference_timesteps, seed=seed)
            ext = "wav"
        else:
            raise g.GenerationError("provider must be 'elevenlabs', 'chatterbox' or 'voxcpm'")
        base = storage.safe_filename(filename or text[:48] or "speech")
        result = await storage.save_result(audio, project=project, subpath=subpath,
                                           filename=base, ext=ext, cfg=cfg)
        return {**result, "provider": provider, "model": voxcpm.MODEL if provider == "voxcpm" else model,
                "mime": "audio/wav" if ext == "wav" else "audio/mpeg"}

    @mcp.tool()
    async def speech_providers() -> dict:
        """Speech engines and current VoxCPM readiness. No model load or GPU generation.
        Local engines accept a reference clip as voice; ElevenLabs accepts a voice id."""
        return {"providers": [
            {"id": "elevenlabs", "name": "ElevenLabs", "reference_audio": False},
            {"id": "chatterbox", "name": "Chatterbox", "reference_audio": True},
            {"id": "voxcpm", "name": "VoxCPM2", "reference_audio": True,
             "voice_design": True, "model": voxcpm.MODEL, **await voxcpm.status(ctx.http, cfg)},
        ]}

    @mcp.tool()
    async def list_voices() -> dict:
        """List available ElevenLabs voices (voice_id + name + category) for
        generate_speech(provider='elevenlabs', voice=<voice_id>)."""
        voices = await g.list_elevenlabs_voices(ctx.http, cfg.elevenlabs_api_key)
        return {"count": len(voices), "voices": voices}
