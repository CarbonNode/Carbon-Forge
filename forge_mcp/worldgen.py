"""Game-world pipeline: one generated map image -> detected objects -> cutout
sprites + collision grid + playable preview (the "capybara.build" workflow).

The map is generated as ONE painting (perfect internal style consistency, baked
lighting), then Gemini spatial understanding returns labeled box_2d detections
([ymin, xmin, ymax, xmax], normalized 0-1000 — Gemini's native convention, kept
end-to-end), obstacle boxes are cut out as transparent sprites via the shared
rembg engine, and walkability is derived from obstacle "footprint bands" (a
character walks BEHIND a tree's canopy but collides with its trunk base).

Pure helpers (parse/clamp/grid/manifest/preview) stay side-effect free for tests;
only detect_scene talks to the network.
"""
import base64
import json
from io import BytesIO

from forge_mcp import generation as g

DETECT_MODEL = "gemini-2.5-flash"
DETECT_MAX_SIDE = 1536  # detection copy is downscaled; boxes are normalized so scale is free

CATEGORIES = ("obstacle", "decor", "enterable", "zone_blocked")

DEFAULT_WORLD_STYLE = (
    "16-bit pixel art, hand-painted in the style of Stardew Valley and Eastward, "
    "three-quarter top-down RPG perspective"
)


def world_prompt(scene: str, style: str = DEFAULT_WORLD_STYLE) -> str:
    """Style-locked map prompt. The style MUST lead and the frame must read as a
    game screenshot — trailing style descriptors after 'top-down map' get ignored
    and Imagen drifts to a photorealistic aerial photo (observed live)."""
    return (
        f"A {style} video game world map, screenshot of a 2D game. "
        f"The map depicts {scene.strip()}. "
        "Soft baked lighting with hard-edged cast shadows, cohesive limited palette, "
        "richly detailed. No characters, no people, no animals, no text, no watermark, "
        "no UI. The scene fills the entire frame edge to edge."
    )

_DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["label", "category", "box_2d"],
            },
        },
        "player_spawn": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["objects"],
}

_DETECT_PROMPT = """You are labeling a top-down 2D game map so it can become playable.
Detect every distinct object and return it with a tight bounding box.

Categories:
- obstacle: anything a walking character collides with and can stand BEHIND — trees, buildings, rocks, fences, furniture, wells, statues, barrels, walls.
- zone_blocked: impassable REGIONS with no sprite silhouette — water, cliffs, pits, the map's border walls.
- enterable: doors, gates, stairs, cave mouths — places a player could transition through.
- decor: flat walkable ground detail — paths, rugs, flowers, grass tufts, floor patterns, painted shadows.

Rules:
- box_2d is [ymin, xmin, ymax, xmax], integers normalized to 0-1000 of the image.
- Boxes must be TIGHT around each object, including its cast shadow for obstacles.
- Give each object a short snake_case label, numbered when repeated (barrel_1, barrel_2).
- Prefer many individual objects over one merged box; never return a box covering most of the map except for zone_blocked regions that truly are that large.
- Also return player_spawn as [y, x] (0-1000): a point on open, walkable ground away from obstacles.
"""


async def detect_scene(http, api_keys, image_bytes: bytes, mime: str,
                       hints: str | None = None, max_objects: int = 48,
                       model: str = DETECT_MODEL) -> dict:
    """Gemini spatial detection -> {"objects": [...], "player_spawn": [y, x] | None}."""
    prompt = _DETECT_PROMPT
    if hints:
        prompt += f"\nExtra guidance from the user: {hints.strip()}\n"
    prompt += f"\nReturn at most {max(4, min(int(max_objects), 100))} objects, largest/most important first."
    small = _detection_copy(image_bytes)
    url = f"{g.GEMINI_API}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/png" if small is not image_bytes else (mime or "image/png"),
                            "data": base64.b64encode(small).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _DETECT_SCHEMA,
        },
    }
    resp = await g._gemini_fetch(http, url, api_keys, body)  # same key rotation as image gen
    candidates = resp.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text.strip():
        raise g.GenerationError("Gemini returned no detection output for the map image")
    try:
        raw = json.loads(text)
    except ValueError as e:
        raise g.GenerationError(f"Gemini detection was not valid JSON: {e}") from e
    return parse_detections(raw, max_objects=max_objects)


def _detection_copy(image_bytes: bytes) -> bytes:
    """Downscale the detection copy (boxes are normalized, so scale doesn't matter)."""
    from PIL import Image
    with Image.open(BytesIO(image_bytes)) as im:
        if max(im.size) <= DETECT_MAX_SIDE:
            return image_bytes
        im = im.convert("RGB")
        scale = DETECT_MAX_SIDE / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()


def parse_detections(raw: dict, max_objects: int = 48) -> dict:
    """Validate/clamp a Gemini detection payload. Drops malformed or degenerate boxes,
    dedupes labels, clamps coordinates into 0-1000, and normalizes spawn to [y, x]."""
    objects, seen = [], {}
    for obj in (raw.get("objects") or [])[: max(4, min(int(max_objects), 100))]:
        if not isinstance(obj, dict):
            continue
        box = obj.get("box_2d")
        cat = obj.get("category")
        if cat not in CATEGORIES or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            y1, x1, y2, x2 = (max(0, min(1000, int(v))) for v in box)
        except (TypeError, ValueError):
            continue
        if y2 <= y1 or x2 <= x1:
            continue
        if (y2 - y1) * (x2 - x1) < 4:  # degenerate speck
            continue
        label = _safe_label(str(obj.get("label") or cat))
        n = seen.get(label, 0) + 1
        seen[label] = n
        if n > 1:
            label = f"{label}_{n}"
        objects.append({"label": label, "category": cat, "box_2d": [y1, x1, y2, x2]})

    spawn = raw.get("player_spawn")
    if isinstance(spawn, (list, tuple)) and len(spawn) == 2:
        try:
            spawn = [max(0, min(1000, int(spawn[0]))), max(0, min(1000, int(spawn[1])))]
        except (TypeError, ValueError):
            spawn = None
    else:
        spawn = None
    return {"objects": objects, "player_spawn": spawn}


def _safe_label(label: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in label.strip().lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:60] or "object"


def box_to_px(box, width: int, height: int, pad_frac: float = 0.015) -> tuple:
    """Normalized [ymin,xmin,ymax,xmax] -> padded pixel crop box (left, top, right, bottom)."""
    y1, x1, y2, x2 = box
    pad = max(2, int(min(width, height) * pad_frac))
    left = max(0, int(x1 / 1000 * width) - pad)
    top = max(0, int(y1 / 1000 * height) - pad)
    right = min(width, int(x2 / 1000 * width) + pad)
    bottom = min(height, int(y2 / 1000 * height) + pad)
    return left, top, right, bottom


def build_collision(objects, cols: int = 64, rows: int = 36, band_frac: float = 0.30) -> dict:
    """Walkability grid (cols x rows over the 0-1000 space) from detections.

    obstacle: only its footprint band blocks — the bottom `band_frac` of the box
    (min 10 units tall), so characters walk behind canopies/roofs but collide at
    the base. zone_blocked: the whole box blocks. decor/enterable: walkable.
    Returns {cols, rows, blocked: sorted [cell_index...]} (index = row * cols + col).
    """
    cols, rows = max(8, int(cols)), max(8, int(rows))
    blocked = set()
    for obj in objects:
        y1, x1, y2, x2 = obj["box_2d"]
        if obj["category"] == "obstacle":
            band_top = max(y1, y2 - max(10, int((y2 - y1) * band_frac)))
            region = (band_top, x1, y2, x2)
        elif obj["category"] == "zone_blocked":
            region = (y1, x1, y2, x2)
        else:
            continue
        ry1, rx1, ry2, rx2 = region
        c1 = max(0, min(cols - 1, int(rx1 * cols / 1000)))
        c2 = max(0, min(cols - 1, int((rx2 - 1) * cols / 1000)))
        r1 = max(0, min(rows - 1, int(ry1 * rows / 1000)))
        r2 = max(0, min(rows - 1, int((ry2 - 1) * rows / 1000)))
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                blocked.add(r * cols + c)
    return {"cols": cols, "rows": rows, "blocked": sorted(blocked)}


def build_manifest(name: str, map_file: str, width: int, height: int,
                   objects: list, spawn, collision: dict | None,
                   sprite_files: dict | None = None) -> dict:
    """The self-describing world.json. sprite_files maps label ->
    {"file": bundle filename, "crop_px": [top, left, bottom, right]} — crop_px is the
    exact padded pixel box the sprite was cut from, so renderers can redraw it in place."""
    out_objects = []
    for obj in objects:
        entry = {"label": obj["label"], "category": obj["category"], "box_2d": obj["box_2d"]}
        cut = (sprite_files or {}).get(obj["label"])
        if cut:
            entry["sprite"] = cut["file"]
            entry["crop_px"] = cut["crop_px"]
        out_objects.append(entry)
    manifest = {
        "format": "carbon-forge-world/1",
        "name": name,
        "map": map_file,
        "width": width,
        "height": height,
        "coordinate_space": {
            "units": "normalized 0-1000",
            "box_2d": "[ymin, xmin, ymax, xmax] (y before x — Gemini spatial convention)",
        },
        "player_spawn": spawn or [500, 500],
        "objects": out_objects,
    }
    if collision:
        manifest["collision"] = collision
    return manifest


def render_preview_html(manifest: dict) -> str:
    """Self-contained playable preview: map + depth-sorted obstacle sprites + collision
    walker. References bundle siblings by relative filename, so it works from the results
    cache URL and from the workspace folder alike."""
    return _PREVIEW_TEMPLATE.replace("__WORLD_JSON__", json.dumps(manifest))


_PREVIEW_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World preview</title>
<style>
  html,body{margin:0;height:100%;background:#0e0c14;overflow:hidden}
  canvas{display:block;width:100vw;height:100vh;object-fit:contain;image-rendering:auto}
  #hud{position:fixed;left:10px;bottom:8px;color:#cbc6de;font:12px/1.4 monospace;opacity:.8;
       user-select:none;pointer-events:none}
</style></head><body>
<canvas id="c"></canvas><div id="hud">WASD / arrows / drag to walk &middot; G toggles collision grid</div>
<script>
const WORLD = __WORLD_JSON__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const N = 1000, col = WORLD.collision || null;
let showGrid = false, imgs = {}, mapImg = new Image();
mapImg.src = WORLD.map;
const obstacles = WORLD.objects.filter(o => o.sprite);
for (const o of obstacles) { const im = new Image(); im.src = o.sprite; imgs[o.label] = im; }
const player = { y: WORLD.player_spawn[0], x: WORLD.player_spawn[1], r: 9, speed: 220 };
const keys = {}, drag = { on:false, x:0, y:0, dx:0, dy:0 };
addEventListener('keydown', e => { keys[e.key.toLowerCase()] = true; if (e.key.toLowerCase() === 'g') showGrid = !showGrid; });
addEventListener('keyup', e => keys[e.key.toLowerCase()] = false);
cv.addEventListener('pointerdown', e => { drag.on = true; drag.x = e.clientX; drag.y = e.clientY; });
addEventListener('pointermove', e => { if (drag.on) { drag.dx = e.clientX - drag.x; drag.dy = e.clientY - drag.y; } });
addEventListener('pointerup', () => { drag.on = false; drag.dx = drag.dy = 0; });
function blocked(x, y) {
  if (x < 8 || x > N - 8 || y < 8 || y > N - 8) return true;
  if (!col) return false;
  const c = Math.min(col.cols - 1, Math.max(0, Math.floor(x * col.cols / N)));
  const r = Math.min(col.rows - 1, Math.max(0, Math.floor(y * col.rows / N)));
  return blockedSet.has(r * col.cols + c);
}
const blockedSet = new Set(col ? col.blocked : []);
let last = performance.now();
function tick(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  let mx = (keys['d'] || keys['arrowright'] ? 1 : 0) - (keys['a'] || keys['arrowleft'] ? 1 : 0);
  let my = (keys['s'] || keys['arrowdown'] ? 1 : 0) - (keys['w'] || keys['arrowup'] ? 1 : 0);
  if (drag.on && (Math.abs(drag.dx) > 8 || Math.abs(drag.dy) > 8)) {
    const m = Math.hypot(drag.dx, drag.dy); mx = drag.dx / m; my = drag.dy / m;
  }
  const len = Math.hypot(mx, my) || 1;
  const nx = player.x + (mx / len) * player.speed * dt;
  const ny = player.y + (my / len) * player.speed * dt;
  if (!blocked(nx, player.y)) player.x = nx;
  if (!blocked(player.x, ny)) player.y = ny;
  draw(); requestAnimationFrame(tick);
}
function draw() {
  const W = mapImg.naturalWidth || 1600, H = mapImg.naturalHeight || 900;
  if (cv.width !== W) { cv.width = W; cv.height = H; }
  ctx.clearRect(0, 0, W, H);
  if (mapImg.complete) ctx.drawImage(mapImg, 0, 0, W, H);
  const px = player.x / N * W, py = player.y / N * H;
  const layers = obstacles.map(o => ({ o, foot: o.box_2d[2] })).concat([{ player: true, foot: player.y }]);
  layers.sort((a, b) => a.foot - b.foot);
  for (const l of layers) {
    if (l.player) { drawPlayer(px, py, H); continue; }
    const im = imgs[l.o.label]; if (!im || !im.complete || !im.naturalWidth) continue;
    const [y1, x1, y2, x2] = l.o.crop_px || [];
    if (l.o.crop_px) ctx.drawImage(im, x1, y1, x2 - x1, y2 - y1);
  }
  if (showGrid && col) {
    ctx.fillStyle = 'rgba(255,60,60,0.28)';
    const cw = W / col.cols, ch = H / col.rows;
    for (const idx of blockedSet) ctx.fillRect((idx % col.cols) * cw, Math.floor(idx / col.cols) * ch, cw, ch);
  }
}
function drawPlayer(px, py, H) {
  const s = H / 22;
  ctx.fillStyle = 'rgba(0,0,0,0.30)';
  ctx.beginPath(); ctx.ellipse(px, py, s * 0.55, s * 0.2, 0, 0, 7); ctx.fill();
  ctx.fillStyle = '#3b7dd8'; ctx.strokeStyle = '#16233c'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(px - s * 0.4, py - s * 1.5, s * 0.8, s * 1.5, s * 0.35);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#f2d4b0';
  ctx.beginPath(); ctx.arc(px, py - s * 1.55, s * 0.34, 0, 7); ctx.fill(); ctx.stroke();
}
requestAnimationFrame(tick);
</script></body></html>
"""
