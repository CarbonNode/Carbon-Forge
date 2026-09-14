"""Meshy.ai — thin async client for the 3D half of the sprite pipeline.

image → 3D (textured mesh) → rig (humanoid armature; free walk/run clips) →
animate (one of Meshy's ~600 library clips baked onto the rig) → GLB URLs.
Same endpoints the Carbon-Cortex blender connector uses (verified live there
2026-07-31); result-shape quirks are handled in pick_glb.

Every task costs credits (preview 20, refine 10, image→3D ~20, rig 5,
animate 3), so tasks are created ONCE and never auto-retried.
Docs: https://docs.meshy.ai/en/api
"""
import asyncio

import httpx

from forge_mcp.generation import GenerationError

API = "https://api.meshy.ai/openapi"
PATHS = {
    "text": "/v2/text-to-3d",
    "image": "/v1/image-to-3d",
    "retexture": "/v1/retexture",
    "rig": "/v1/rigging",
    "animate": "/v1/animations",
}
KINDS = tuple(PATHS)

# Library clips that matter for a game unit, by the name a caller would say.
# (ids from Meshy's Animation Library reference, 2026-09.) 'walk' and 'run'
# also come FREE with every rig task (result.basic_animations) — the pipeline
# uses those first and only spends an animate task when a clip is missing.
ACTIONS = {
    "idle": 0, "idle2": 11, "idle3": 12,
    "walk": 30, "walk_woman": 1, "walk_fight": 21, "walk_back": 20,
    "run": 14, "run_fast": 16, "jog": 15,
    "attack": 4, "combo": 92, "slash": 219, "charged_slash": 242, "axe_chop": 237, "axe_spin": 238,
    "death": 8, "die": 8, "dead": 8,
    "hurt": 178, "hit": 178, "hit2": 179, "knockback": 7,
    "jump": 466, "jump_run": 13,
    "combat_idle": 89, "cast": 125, "dance": 22,
}
RIG_BASIC = {"walk": "walking_glb_url", "run": "running_glb_url"}
CREDITS = {"image": 20, "text_preview": 20, "text_refine": 10, "rig": 5, "animate": 3}


def resolve_action(name):
    """'attack' → 4, '8' → 8, 8 → 8. Raises on an unknown name."""
    if isinstance(name, bool):
        raise GenerationError(f"bad action {name!r}")
    if isinstance(name, int):
        return int(name)
    s = str(name).strip().lower()
    if s.lstrip("-").isdigit():
        return int(s)
    if s in ACTIONS:
        return ACTIONS[s]
    raise GenerationError(f"unknown action '{name}' — use one of {', '.join(sorted(ACTIONS))} "
                          f"or a numeric Meshy action_id")


def _headers(key):
    if not key:
        raise GenerationError("MESHY_API_KEY is not configured on the forge service")
    return {"Authorization": f"Bearer {key}"}


async def _call(client: httpx.AsyncClient, key, method, path, body=None):
    try:
        r = await client.request(method, API + path, headers=_headers(key), json=body, timeout=60)
    except httpx.HTTPError as e:
        raise GenerationError(f"Meshy {method} {path}: {e}") from e
    try:
        data = r.json()
    except ValueError:
        data = None
    if r.status_code >= 400:
        detail = (data or {}).get("message") if isinstance(data, dict) else None
        raise GenerationError(f"Meshy {method} {path} → HTTP {r.status_code}: {detail or r.text[:300]}")
    return data if data is not None else {}


def _task_id(json_resp) -> str:
    tid = json_resp.get("result") or json_resp.get("id") or ""
    if not tid:
        raise GenerationError(f"Meshy returned no task id: {json_resp}")
    return str(tid)


async def image_to_3d(client, key, image_url, *, should_texture=True, enable_pbr=False,
                      should_remesh=True, target_polycount=None, ai_model=None) -> str:
    body = {"image_url": image_url, "should_texture": bool(should_texture),
            "enable_pbr": bool(enable_pbr), "should_remesh": bool(should_remesh)}
    if target_polycount:
        body["target_polycount"] = int(target_polycount)
    if ai_model:
        body["ai_model"] = ai_model
    return _task_id(await _call(client, key, "POST", PATHS["image"], body))


async def text_to_3d(client, key, prompt, *, mode="preview", preview_task_id=None, art_style=None,
                     enable_pbr=False, should_remesh=True, target_polycount=None, ai_model=None) -> str:
    body = {"mode": mode}
    if mode == "refine":
        body["preview_task_id"] = preview_task_id
        body["enable_pbr"] = bool(enable_pbr)
    else:
        body["prompt"] = prompt
        if art_style:
            body["art_style"] = art_style
        body["should_remesh"] = bool(should_remesh)
        if target_polycount:
            body["target_polycount"] = int(target_polycount)
    if ai_model:
        body["ai_model"] = ai_model
    return _task_id(await _call(client, key, "POST", PATHS["text"], body))


async def rig(client, key, *, input_task_id=None, model_url=None, height_meters=1.7,
              texture_image_url=None) -> str:
    if not input_task_id and not model_url:
        raise GenerationError("rig needs input_task_id or model_url")
    body = {"height_meters": float(height_meters)}
    if input_task_id:
        body["input_task_id"] = input_task_id
    else:
        body["model_url"] = model_url
    if texture_image_url:
        body["texture_image_url"] = texture_image_url
    return _task_id(await _call(client, key, "POST", PATHS["rig"], body))


async def animate(client, key, rig_task_id, action_id, *, fps=None) -> str:
    body = {"rig_task_id": rig_task_id, "action_id": int(action_id)}
    if fps:
        body["fps"] = int(fps)
    return _task_id(await _call(client, key, "POST", PATHS["animate"], body))


async def get_task(client, key, kind, task_id) -> dict:
    if kind not in PATHS:
        raise GenerationError(f"kind must be one of {KINDS}")
    return await _call(client, key, "GET", f"{PATHS[kind]}/{task_id}")


async def balance(client, key):
    try:
        data = await _call(client, key, "GET", "/v1/balance")
        return int(data.get("balance"))
    except Exception:  # noqa: BLE001 — informational only
        return None


async def wait_task(client, key, kind, task_id, *, budget_s=1800, poll_s=6.0, on_status=None) -> dict:
    """Poll until SUCCEEDED. FAILED/CANCELED/timeout → GenerationError (never re-submits)."""
    deadline = asyncio.get_event_loop().time() + budget_s
    last = None
    while True:
        task = await get_task(client, key, kind, task_id)
        status = str(task.get("status") or "").upper()
        msg = f"{kind} {status.lower() or 'pending'} {task.get('progress') or 0}%"
        if on_status and msg != last:
            on_status(msg)
            last = msg
        if status == "SUCCEEDED":
            return task
        if status in ("FAILED", "CANCELED", "EXPIRED"):
            err = task.get("task_error") or {}
            raise GenerationError(f"Meshy {kind} task {task_id} {status.lower()}: "
                                  f"{err.get('message') if isinstance(err, dict) else err or 'no detail'}")
        if asyncio.get_event_loop().time() > deadline:
            raise GenerationError(f"Meshy {kind} task {task_id} still {status or 'pending'} after {budget_s}s")
        await asyncio.sleep(poll_s)


def _walk_for_glb(v, depth=0):
    if depth > 6:
        return None
    if isinstance(v, str):
        low = v.split("?")[0].lower()
        return v if low.startswith("http") and low.endswith(".glb") else None
    if isinstance(v, list):
        for x in v:
            f = _walk_for_glb(x, depth + 1)
            if f:
                return f
    if isinstance(v, dict):
        for x in v.values():
            f = _walk_for_glb(x, depth + 1)
            if f:
                return f
    return None


def pick_glb(kind, task) -> str | None:
    """The downloadable GLB of a finished task — documented field first, then the
    verified-live nesting, then any .glb URL anywhere in the payload."""
    r = task.get("result") if isinstance(task.get("result"), dict) else {}
    candidates = []
    if kind == "animate":
        candidates += [r.get("animation_glb_url"), task.get("animation_glb_url")]
    if kind == "rig":
        candidates += [r.get("rigged_character_glb_url"), task.get("rigged_character_glb_url")]
    mu = task.get("model_urls") if isinstance(task.get("model_urls"), dict) else {}
    rmu = r.get("model_urls") if isinstance(r.get("model_urls"), dict) else {}
    candidates += [mu.get("glb"), rmu.get("glb"), task.get("glb_url")]
    for c in candidates:
        if isinstance(c, str) and c:
            return c
    return _walk_for_glb(task)


def rig_basic_animations(task) -> dict:
    """{'walk': url, 'run': url} — the free clips a rig task ships with (either may be missing)."""
    r = task.get("result") if isinstance(task.get("result"), dict) else {}
    basic = r.get("basic_animations") or task.get("basic_animations") or {}
    out = {}
    for name, field in RIG_BASIC.items():
        url = basic.get(field) if isinstance(basic, dict) else None
        if isinstance(url, str) and url:
            out[name] = url
    return out


async def download(client, url, max_bytes) -> bytes:
    try:
        async with client.stream("GET", url, timeout=300, follow_redirects=True) as r:
            r.raise_for_status()
            chunks, total = [], 0
            async for chunk in r.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise GenerationError(f"Meshy download exceeds {max_bytes // (1024 * 1024)} MB")
                chunks.append(chunk)
    except httpx.HTTPError as e:
        raise GenerationError(f"Meshy download failed: {e}") from e
    data = b"".join(chunks)
    if len(data) < 20 or data[:4] != b"glTF":
        raise GenerationError("Meshy download is not a binary glTF (.glb)")
    return data
