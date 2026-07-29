"""Replicate — the whole api.replicate.com catalog through forge.

Thin async client (httpx, no SDK): catalog search (the documented HTTP QUERY
method), model + input-schema inspection, prediction create/poll, and the files
API for feeding local inputs to models. Predictions are created single-shot
(never auto-retried — a blind retry can double-charge a running prediction);
reads go through generation.fetch_json and keep its 429/5xx backoff.
"""
import asyncio

import httpx

from forge_mcp.generation import GenerationError, fetch_json

API = "https://api.replicate.com/v1"

MIME_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif",
    "image/bmp": "bmp", "video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm",
    "audio/mpeg": "mp3", "audio/wav": "wav", "audio/ogg": "ogg", "audio/flac": "flac",
    "audio/mp4": "m4a", "model/gltf-binary": "glb",
}


def _headers(token) -> dict:
    if not token:
        raise GenerationError("REPLICATE_API_TOKEN is not configured on the forge service")
    return {"Authorization": f"Bearer {token}"}


def model_brief(m: dict) -> dict:
    """Compact catalog row for search results."""
    desc = (m.get("description") or "").strip()
    out = {
        "model": f"{m.get('owner')}/{m.get('name')}",
        "description": desc[:200] + ("…" if len(desc) > 200 else ""),
        "run_count": m.get("run_count"),
    }
    if m.get("is_official"):
        out["official"] = True
    return out


def summarize_inputs(version: dict) -> list:
    """The model's input schema as a compact ordered list — enough to call it right."""
    schema = ((version or {}).get("openapi_schema") or {}).get("components", {}).get("schemas", {})
    inp = schema.get("Input") or {}
    required = set(inp.get("required") or [])
    props = inp.get("properties") or {}

    def resolve(prop):
        # enums arrive as allOf -> $ref to a components schema carrying the enum list
        for sub in prop.get("allOf") or []:
            ref = sub.get("$ref", "")
            name = ref.rsplit("/", 1)[-1]
            if name in schema:
                return {**schema[name], **{k: v for k, v in prop.items() if k != "allOf"}}
        return prop

    rows = []
    for name, raw in props.items():
        p = resolve(raw)
        row = {"name": name, "type": p.get("type") or ("array" if "items" in p else "string")}
        if name in required:
            row["required"] = True
        for k_src, k_out in (("description", "description"), ("default", "default"),
                             ("enum", "enum"), ("minimum", "min"), ("maximum", "max"),
                             ("format", "format")):
            if p.get(k_src) is not None:
                row[k_out] = p[k_src]
        if isinstance(row.get("description"), str) and len(row["description"]) > 220:
            row["description"] = row["description"][:220] + "…"
        rows.append((p.get("x-order", 999), row))
    rows.sort(key=lambda t: t[0])
    return [r for _, r in rows]


def collect_file_urls(output) -> list:
    """Every http(s) URL in a prediction's output, in traversal order — models return a
    bare URL, a list of URLs, or nested dicts; this walks them all."""
    urls = []

    def walk(node):
        if isinstance(node, str):
            if node.startswith("http://") or node.startswith("https://"):
                urls.append(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for item in node.values():
                walk(item)

    walk(output)
    return urls


def output_text(output) -> str | None:
    """Joined text for non-file outputs (LLMs stream token lists; some models return a
    plain string). None when the output is file URLs / structured data."""
    if isinstance(output, str) and not output.startswith(("http://", "https://")):
        return output
    if isinstance(output, list) and output and all(
            isinstance(s, str) and not s.startswith(("http://", "https://")) for s in output):
        return "".join(output)
    return None


def ext_for(url: str, data: bytes, sniff) -> str:
    """Output-file extension: the URL's own suffix when sane, else magic-byte sniff."""
    path = url.split("?", 1)[0].rsplit("/", 1)[-1]
    if "." in path:
        suffix = path.rsplit(".", 1)[-1].lower()
        if 1 <= len(suffix) <= 5 and suffix.isalnum():
            return "jpg" if suffix == "jpeg" else suffix
    return MIME_EXT.get(sniff(data) or "", "bin")


async def search_models(client, token, query: str, limit: int = 8) -> list:
    resp = await client.request("QUERY", f"{API}/models", content=query.encode(),
                                headers={**_headers(token), "Content-Type": "text/plain"},
                                timeout=30)
    if resp.status_code >= 400:
        raise GenerationError(f"Replicate search failed: HTTP {resp.status_code}: {resp.text[:300]}")
    return [model_brief(m) for m in (resp.json().get("results") or [])[:max(1, limit)]]


async def get_model(client, token, model: str) -> dict:
    if model.count("/") != 1:
        raise GenerationError("model must be 'owner/name' (e.g. 'black-forest-labs/flux-schnell')")
    return await fetch_json(client, f"{API}/models/{model}", method="GET", headers=_headers(token))


async def upload_file(client, token, data: bytes, filename: str) -> str:
    """Push bytes to Replicate's files API; the returned URL is usable as a model input."""
    resp = await client.post(f"{API}/files", headers=_headers(token),
                             files={"content": (filename, data)}, timeout=120)
    if resp.status_code >= 400:
        raise GenerationError(f"Replicate file upload failed: HTTP {resp.status_code}: {resp.text[:300]}")
    url = ((resp.json().get("urls") or {}).get("get"))
    if not url:
        raise GenerationError("Replicate file upload returned no serving URL")
    return url


async def create_prediction(client, token, model: str, input: dict, wait: int = 60) -> dict:
    """Start a prediction — 'owner/name' runs the latest version, 'owner/name:version'
    pins one. Single-shot on purpose (no retry). Prefer:wait holds the connection up to
    60s so fast models come back completed in one call."""
    headers = _headers(token)
    if wait > 0:
        headers["Prefer"] = f"wait={min(60, wait)}"
    if ":" in model:
        name, _, version = model.partition(":")
        if name.count("/") != 1 or not version:
            raise GenerationError("model must be 'owner/name' or 'owner/name:version'")
        url, body = f"{API}/predictions", {"version": version, "input": input}
    else:
        if model.count("/") != 1:
            raise GenerationError("model must be 'owner/name' or 'owner/name:version'")
        url, body = f"{API}/models/{model}/predictions", {"input": input}
    try:
        resp = await client.post(url, json=body, headers=headers, timeout=90)
    except httpx.HTTPError as e:
        raise GenerationError(f"Replicate request failed: {e}") from e
    if resp.status_code >= 400:
        raise GenerationError(f"Replicate: HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


async def get_prediction(client, token, prediction: dict) -> dict:
    url = (prediction.get("urls") or {}).get("get") or f"{API}/predictions/{prediction['id']}"
    return await fetch_json(client, url, method="GET", headers=_headers(token))


async def wait_prediction(client, token, prediction: dict, budget_s: float,
                          poll_s: float = 3.0) -> dict:
    """Poll until the prediction settles or the time budget runs out."""
    elapsed = 0.0
    while prediction.get("status") in ("starting", "processing") and elapsed < budget_s:
        await asyncio.sleep(poll_s)
        elapsed += poll_s
        prediction = await get_prediction(client, token, prediction)
    return prediction


async def download_output(client, token, url: str, max_bytes: int) -> bytes:
    """Fetch one output file. replicate.delivery URLs are public-signed; files served
    from the API host need the bearer."""
    headers = _headers(token) if url.startswith(API) else {}
    try:
        resp = await client.get(url, headers=headers, follow_redirects=True, timeout=300)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GenerationError(f"Could not download Replicate output {url}: {e}") from e
    if len(resp.content) > max_bytes:
        raise GenerationError(
            f"Replicate output too large: {len(resp.content) // (1024 * 1024)} MB "
            f"(limit {max_bytes // (1024 * 1024)} MB)")
    return resp.content
