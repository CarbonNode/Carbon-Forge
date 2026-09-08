"""Retro Diffusion (Astropulse) — the official api.retrodiffusion.ai v2 client.

Used for "advanced animations": animate YOUR OWN start frame (walking / idle /
jump / crouch / attack / destroy / custom_action / subtle_motion) at its native
size (32-256 px), which the Replicate-hosted rd-animation model does not
expose. Thin httpx client: submit (v2 always queues) → poll the task → decode.
One submit per call, never auto-retried (a retry double-charges the balance).

Docs: https://github.com/Retro-Diffusion/api-examples (README + llms.txt).
"""
import asyncio
import base64

import httpx

from forge_mcp.generation import GenerationError

API = "https://api.retrodiffusion.ai/v2"

# Advanced-animation verbs → prompt_style; price per run (USD, docs 2026-09).
ADVANCED_ACTIONS = {
    "walking": 0.14, "idle": 0.14, "jump": 0.14, "crouch": 0.14, "attack": 0.14,
    "destroy": 0.14, "custom_action": 0.25, "subtle_motion": 0.25,
}
FRAME_COUNTS = (4, 6, 8, 10, 12, 16)
MIN_SIZE, MAX_SIZE = 32, 256


def snap_frames(n) -> int:
    """frames_duration only accepts 4|6|8|10|12|16 — pick the nearest."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 8
    return min(FRAME_COUNTS, key=lambda f: (abs(f - n), f))


def _headers(key) -> dict:
    if not key:
        raise GenerationError("RETRO_DIFFUSION_API_KEY is not configured on the forge service")
    return {"X-RD-Token": key, "Content-Type": "application/json"}


def advanced_payload(prompt: str, action: str, png: bytes, width: int, height: int,
                     frames: int = 8, seed=None, spritesheet: bool = True,
                     bypass_expansion: bool = False) -> dict:
    if action not in ADVANCED_ACTIONS:
        raise GenerationError(f"action must be one of {', '.join(ADVANCED_ACTIONS)}")
    if not (MIN_SIZE <= width <= MAX_SIZE and MIN_SIZE <= height <= MAX_SIZE):
        raise GenerationError(f"advanced animation frames must be {MIN_SIZE}-{MAX_SIZE} px "
                              f"(got {width}x{height})")
    body = {
        "prompt": prompt, "prompt_style": f"rd_advanced_animation__{action}",
        "width": int(width), "height": int(height), "num_images": 1,
        "frames_duration": snap_frames(frames),
        "input_image": base64.b64encode(png).decode("ascii"),
        "return_spritesheet": bool(spritesheet),
    }
    if seed is not None:
        body["seed"] = int(seed)
    if bypass_expansion:
        body["bypass_prompt_expansion"] = True
    return body


async def check_cost(client, key, payload: dict) -> dict:
    """Free dry run — the price the same payload would charge."""
    try:
        resp = await client.post(f"{API}/inferences", json={**payload, "check_cost": True},
                                 headers=_headers(key), timeout=60)
    except httpx.HTTPError as e:
        raise GenerationError(f"Retro Diffusion request failed: {e}") from e
    if resp.status_code >= 400:
        raise GenerationError(f"Retro Diffusion: HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


async def submit(client, key, payload: dict) -> dict:
    """POST /v2/inferences — returns {status:'accepted', task_id} (v2 always queues);
    some deployments answer synchronously with the result — both are handled."""
    try:
        resp = await client.post(f"{API}/inferences", json=payload, headers=_headers(key), timeout=120)
    except httpx.HTTPError as e:
        raise GenerationError(f"Retro Diffusion request failed: {e}") from e
    if resp.status_code >= 400:
        raise GenerationError(f"Retro Diffusion: HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


async def get_task(client, key, task_id: str) -> dict:
    try:
        resp = await client.get(f"{API}/inferences/tasks/{task_id}", headers=_headers(key), timeout=60)
    except httpx.HTTPError as e:
        raise GenerationError(f"Retro Diffusion poll failed: {e}") from e
    if resp.status_code >= 400:
        raise GenerationError(f"Retro Diffusion: HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


async def wait_task(client, key, submitted: dict, budget_s: float = 600, poll_s: float = 3.0,
                    on_status=None) -> dict:
    """Poll until succeeded/failed. Returns the RESULT dict (base64_images /
    output_urls / balance_cost / remaining_balance)."""
    if submitted.get("base64_images") or submitted.get("output_urls"):
        return submitted  # synchronous answer
    task_id = submitted.get("task_id")
    if not task_id:
        raise GenerationError(f"Retro Diffusion returned no task_id: {str(submitted)[:300]}")
    elapsed = 0.0
    while elapsed < budget_s:
        await asyncio.sleep(poll_s)
        elapsed += poll_s
        task = await get_task(client, key, task_id)
        status = task.get("status")
        if on_status:
            on_status(status)
        if status == "succeeded":
            return task.get("result") or task
        if status == "failed":
            raise GenerationError(f"Retro Diffusion task failed: {task.get('error') or 'no detail'}")
    raise GenerationError(f"Retro Diffusion task {task_id} still running after {int(budget_s)}s")


async def first_image(client, key, result: dict) -> bytes:
    """The generated PNG/GIF bytes: inline base64 first, else the hosted URL."""
    b64 = (result.get("base64_images") or [None])[0]
    if b64:
        return base64.b64decode(b64)
    url = (result.get("output_urls") or [None])[0]
    if not url:
        raise GenerationError("Retro Diffusion returned no image")
    try:
        resp = await client.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GenerationError(f"Could not download Retro Diffusion output: {e}") from e
    return resp.content
