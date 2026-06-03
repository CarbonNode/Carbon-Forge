"""Async bridge to backend.processing. CPU-heavy work runs in threads behind a
small semaphore so concurrent MCP calls can't thrash the box."""
import asyncio
import os
import threading

from backend import processing
from backend.processing import PipelineOptions, AVAILABLE_MODELS, DEFAULT_MODEL  # noqa: F401 (re-export)

_cpu_gate = asyncio.Semaphore(2)
_load_lock = threading.Lock()


def _ensure_session(model_id: str):
    with _load_lock:  # rembg new_session is not safe to race
        processing.get_session(model_id)


async def preload_default_model():
    await asyncio.to_thread(_ensure_session, DEFAULT_MODEL)
    print("MODEL_READY", flush=True)


async def run_pipeline(data: bytes, opts: PipelineOptions) -> bytes:
    async with _cpu_gate:
        await asyncio.to_thread(_ensure_session, opts.model)
        return await asyncio.to_thread(processing.run_pipeline, data, opts)


async def run_split_pipeline(data: bytes, opts: PipelineOptions, min_area: int) -> list:
    async with _cpu_gate:
        await asyncio.to_thread(_ensure_session, opts.model)
        return await asyncio.to_thread(processing.run_split_pipeline, data, opts, min_area)


async def split_only(data: bytes, min_area: int) -> list:
    async with _cpu_gate:
        return await asyncio.to_thread(processing.split_sprites, data, min_area)


def status() -> dict:
    return {
        "loaded_models": sorted(processing.sessions.keys()),
        "lama_model_present": os.path.exists(processing.LAMA_MODEL_PATH),
        "lama_loaded": processing.lama_session is not None,
    }
