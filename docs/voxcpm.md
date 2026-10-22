# VoxCPM2 speech

`forge__generate_speech(provider="voxcpm", text="Hello", project="General")`
generates 48 kHz WAV with the pinned `openbmb/VoxCPM2` model. The existing default
remains ElevenLabs. There is no automatic fallback to a paid provider.

- `voice`: optional workspace path or HTTPS URL to a reference clip (1–30 seconds,
  at most 10 MB). The same input resolver used by Chatterbox supplies the bytes.
- `voice_description`: optional plain-language voice/delivery description, without
  parentheses. Omit `voice` to design a voice; supply both to steer a cloned voice.
- `prompt_text`: optional exact transcript of `voice` for continuation cloning.
- `cfg_value`: guidance, default 2; `inference_timesteps`: default 10; `seed`: optional.
- `model`: omit or use `openbmb/VoxCPM2`. Other checkpoints are refused.
- `speech_providers`: metadata and current VoxCPM readiness; never loads the model.

`voxcpm_service` is an independent FastAPI worker, not part of Chatterbox or ComfyUI.
Every synthesis owns a short-lived process: one job at a time, no queued retries,
90-second deadline, and all GPU memory released when that child exits. Cold
generation refuses at GPU utilization >=50%, less than 9 GB free VRAM, or if the
GPU cannot be checked. Worker health does not imply a generation has been tested.
Reference conversion permits only local file/pipe protocols and deletes all inputs
when the request ends. Health is LAN-readable; synthesis requires `FORGE_TOKEN`.

## Deployment

Commit and push, then use Conduit `deploy` for **both** environments at the same SHA:

- `carbon-forge / prod-voxcpm`: isolated checkout `C:\Programming\ForgeVoxCPM`
  on laybackrig, `docker-compose.voxcpm.yml`, port 5127. Reads the existing token
  from `C:\Programming\CarbonForge\.env`; leaves that old checkout untouched.
  Downloads the pinned public model into the `voxcpm-models` volume without loading
  it. Other Forge/CHIM containers are not restarted.
- `carbon-forge / prod`: orchestrator on super_server. `FORGE_VOXCPM_URL` defaults
  in its compose file to `http://192.168.0.177:5127`. `FORGE_VOXCPM_TOKEN` can override
  the default shared `FORGE_TOKEN` if the worker uses a different token.

Verify `speech_providers`, then a short original speech sample and a reference-only
clone of that sample. Confirm RIFF/WAVE, sample rate, nonzero duration and audible
speech; wait for idle GPU headroom if the worker refuses. The pinned package is
`voxcpm==2.0.3`; model revision is recorded in `voxcpm_service/download.py`.

Upstream: https://github.com/OpenBMB/VoxCPM (Apache-2.0 code and model).
Tests: `python -m pytest tests/test_voxcpm.py tests/test_config.py -q` (no model download).
