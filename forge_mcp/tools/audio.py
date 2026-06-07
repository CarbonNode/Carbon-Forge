"""Audio / TTS generation. ElevenLabs (cloud) now; local Chatterbox on the 4090
ComfyUI boxes is wired in a follow-up (provider='chatterbox')."""

from forge_mcp import generation as g
from forge_mcp import storage


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def generate_speech(text: str, project: str, provider: str = "elevenlabs",
                              voice: str | None = None, model: str | None = None,
                              stability: float = 0.5, similarity_boost: float = 0.75,
                              style: float = 0.0, subpath: str | None = None,
                              filename: str | None = None) -> dict:
        """Generate speech audio (TTS) from text, saved into the project's workspace folder
        (and a shareable cache URL), same I/O contract as the image/video tools.

        provider:
          - 'elevenlabs' (default) — cloud, expressive, large prebuilt voice library + cloning.
            Needs ELEVENLABS_API_KEY on the service. Use list_voices() to discover voice ids.
          - 'chatterbox' — local, on the 4090 ComfyUI boxes (no cloud, free). COMING SOON.
        voice: ElevenLabs voice_id (default 'Rachel'); model: ElevenLabs model_id
          (default eleven_multilingual_v2; 'eleven_turbo_v2_5' for low latency).
        stability / similarity_boost / style: ElevenLabs voice_settings, each 0..1.
        Returns the saved audio result (workspace path + url)."""
        if provider == "elevenlabs":
            audio = await g.call_elevenlabs(
                ctx.http, cfg.elevenlabs_api_key, text, voice_id=voice, model_id=model,
                stability=stability, similarity_boost=similarity_boost, style=style)
            ext = "mp3"
        elif provider == "chatterbox":
            raise g.GenerationError(
                "Local Chatterbox TTS isn't wired up yet — use provider='elevenlabs' for now.")
        else:
            raise g.GenerationError("provider must be 'elevenlabs' or 'chatterbox'")
        base = storage.safe_filename(filename or text[:48] or "speech")
        return await storage.save_result(audio, project=project, subpath=subpath,
                                          filename=base, ext=ext, cfg=cfg)

    @mcp.tool()
    async def list_voices() -> dict:
        """List available ElevenLabs voices (voice_id + name + category) for
        generate_speech(provider='elevenlabs', voice=<voice_id>)."""
        voices = await g.list_elevenlabs_voices(ctx.http, cfg.elevenlabs_api_key)
        return {"count": len(voices), "voices": voices}
