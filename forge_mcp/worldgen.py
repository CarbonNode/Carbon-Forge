"""Game-world pipeline: one generated map image -> detected objects -> cutout
sprites + collision grid + playable preview (the "capybara.build" workflow).

Each map is painted as ONE image (perfect internal style consistency, baked
lighting), then Gemini spatial understanding returns labeled box_2d detections
([ymin, xmin, ymax, xmax], normalized 0-1000 — Gemini's native convention, kept
end-to-end), obstacle boxes are cut out as transparent sprites via the shared
rembg engine, and walkability is derived from obstacle "footprint bands" (a
character walks BEHIND a tree's canopy but collides with its trunk base).

Explorable worlds are MULTI-MAP (the classic RPG model): detected `enterable`
doors can each expand into a style-matched interior map (Gemini image edit with
the exterior as reference), linked both ways; the preview walks between them
with a follow camera.

Pure helpers (parse/clamp/grid/manifest/links/preview) stay side-effect free for
tests; only detect_scene talks to the network.
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


def interior_prompt(label: str, scene: str, style: str = DEFAULT_WORLD_STYLE) -> str:
    """Interior-map prompt for a Gemini image EDIT call that carries the exterior map
    as a reference image — that reference is what locks palette and rendering style."""
    pretty = label.replace("_", " ")
    return (
        f"Using the reference image's exact art style, palette, lighting and pixel "
        f"density, paint a DIFFERENT image: the single-room INTERIOR of the {pretty} "
        f"from that scene (the scene is {scene.strip()}). "
        f"A {style} video game interior map, screenshot of a 2D game: one room seen "
        "from above, floor filling most of the frame, furniture and props along the "
        "walls, and the entrance DOOR on the BOTTOM edge of the room. "
        "No characters, no people, no animals, no text, no watermark, no UI. "
        "The room fills the entire frame edge to edge."
    )


INTERIOR_DETECT_HINTS = (
    "This is a single INDOOR room. The walls and anything outside them are zone_blocked. "
    "The entrance door on the bottom edge is enterable (label it exit_door). Furniture "
    "and props inside the room are obstacles."
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
- For enterable doors, name the label after what they lead into (brewing_shed_door, tavern_door).
- Prefer many individual objects over one merged box; never return a box covering most of the map except for zone_blocked regions that truly are that large.
- Also return player_spawn as [y, x] (0-1000): a point on open, walkable ground away from obstacles.
"""


async def detect_scene(http, api_keys, image_bytes: bytes, mime: str,
                       hints: str | None = None, max_objects: int = 48,
                       model: str = DETECT_MODEL) -> dict:
    """Gemini spatial detection -> {"objects": [...], "player_spawn": [y, x] | None}."""
    prompt = _DETECT_PROMPT
    if hints:
        prompt += f"\nExtra guidance: {hints.strip()}\n"
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
    the base. zone_blocked: the whole box blocks. decor: walkable. enterable:
    CARVES an opening — its cells (plus a small approach apron below) are
    unblocked LAST, so a door in a building's facade stays reachable and the
    player can actually step into its transition trigger (a door buried in its
    building's footprint band soft-locked the world, observed live).
    Returns {cols, rows, blocked: sorted [cell_index...]} (index = row * cols + col).
    """
    cols, rows = max(8, int(cols)), max(8, int(rows))

    def cells(ry1, rx1, ry2, rx2):
        c1 = max(0, min(cols - 1, int(rx1 * cols / 1000)))
        c2 = max(0, min(cols - 1, int((rx2 - 1) * cols / 1000)))
        r1 = max(0, min(rows - 1, int(ry1 * rows / 1000)))
        r2 = max(0, min(rows - 1, int((ry2 - 1) * rows / 1000)))
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                yield r * cols + c

    blocked = set()
    for obj in objects:
        y1, x1, y2, x2 = obj["box_2d"]
        if obj["category"] == "obstacle":
            band_top = max(y1, y2 - max(10, int((y2 - y1) * band_frac)))
            blocked.update(cells(band_top, x1, y2, x2))
        elif obj["category"] == "zone_blocked":
            blocked.update(cells(y1, x1, y2, x2))
    for obj in objects:
        if obj["category"] != "enterable":
            continue
        y1, x1, y2, x2 = obj["box_2d"]
        # Carve all the way through the HOST obstacle's base (a door sits above its
        # building's wall base, so a fixed apron leaves a blocked strip below it —
        # observed live: the player stopped 50 units short of the tavern door).
        carve_bottom = y2 + 30
        for host in objects:
            if host["category"] not in ("obstacle", "zone_blocked"):
                continue
            hy1, hx1, hy2, hx2 = host["box_2d"]
            if hx1 < x2 and hx2 > x1 and hy1 < y2 and hy2 > y1:  # overlaps the door
                carve_bottom = max(carve_bottom, hy2 + 15)
        blocked.difference_update(cells(y1, x1, min(1000, carve_bottom), x2))
    return {"cols": cols, "rows": rows, "blocked": sorted(blocked)}


# ---- sprite triage + masks + QA ----

MIN_SPRITE_HEIGHT = 70  # normalized units; shorter obstacles are background-only + collision


def needs_occlusion_sprite(obj, min_height: int = MIN_SPRITE_HEIGHT) -> bool:
    """Only TALL obstacles need cutout sprites — a player can visibly stand behind a
    tree or a building, never behind a basket. Small props stay painted into the
    background (which renders them perfectly) and keep only their collision, which
    eliminates their clipping risk entirely."""
    if obj["category"] != "obstacle":
        return False
    y1, _x1, y2, _x2 = obj["box_2d"]
    return (y2 - y1) >= min_height


def accumulate_mask_union(union, mask, box_px, contain_thresh: float = 0.75):
    """Fold one SAM mask into a sprite's alpha union if the mask lives mostly inside
    the sprite's crop box. `mask` is a bool ndarray (H, W) at full map resolution;
    `union` is a bool ndarray shaped to the crop (or None). Returns the new union.
    Streaming (one mask at a time) keeps peak memory at one full-map mask."""
    left, top, right, bottom = box_px
    total = int(mask.sum())
    if total < 24:
        return union
    inside = int(mask[top:bottom, left:right].sum())
    if inside / total < contain_thresh:
        return union
    piece = mask[top:bottom, left:right]
    if union is None:
        return piece.copy()
    union |= piece
    return union


def union_coverage(union, box_px) -> float:
    """How much of the crop box the accumulated union explains (0-1)."""
    if union is None:
        return 0.0
    left, top, right, bottom = box_px
    area = max(1, (right - left) * (bottom - top))
    return float(union.sum()) / area


def apply_mask_alpha(crop_png: bytes, union) -> bytes:
    """Apply a bool mask as the crop's alpha channel (1px dilation + soft edge so the
    matte doesn't shave the outline)."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation, gaussian_filter
    im = Image.open(BytesIO(crop_png)).convert("RGBA")
    mask = binary_dilation(union, iterations=1)
    alpha = gaussian_filter(mask.astype("float32"), sigma=0.7)
    alpha = np.clip(alpha * 1.4, 0.0, 1.0)
    arr = np.array(im)
    arr[:, :, 3] = (alpha * 255).astype("uint8")
    return _png_bytes(Image.fromarray(arr))


def composite_on_magenta(sprite_png: bytes, max_side: int = 384) -> bytes:
    """Flatten a transparent sprite onto magenta for the QA judge (alpha holes and
    contamination both read clearly against it), downscaled to keep the call cheap."""
    from PIL import Image
    im = Image.open(BytesIO(sprite_png)).convert("RGBA")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.LANCZOS)
    bg = Image.new("RGBA", im.size, (255, 0, 255, 255))
    bg.alpha_composite(im)
    return _png_bytes(bg.convert("RGB"))


def _png_bytes(im) -> bytes:
    buf = BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


QA_VERDICTS = ("clean", "clipped", "contaminated", "empty")

_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(QA_VERDICTS)},
                },
                "required": ["index", "verdict"],
            },
        },
    },
    "required": ["verdicts"],
}

_QA_PROMPT = """Each image is a cutout video-game sprite composited on a MAGENTA background,
numbered in the order given, starting at index 0. The expected object for each index:
{labels}

Judge every image:
- clean: one complete object (its attached cast shadow is fine).
- clipped: the object is visibly cut off at the crop edge (missing top/side).
- contaminated: large parts of OTHER objects or of the ground/scenery are included.
- empty: no recognizable object (mostly magenta).

Return a verdict for every index."""


async def judge_sprites(http, api_keys, sprites: list, labels: list,
                        model: str = DETECT_MODEL) -> list:
    """Batch-QA sprite cutouts with Gemini. sprites: list of magenta-composited PNG
    bytes. Returns one verdict per sprite ('clean' when the judge skipped one)."""
    parts = [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b).decode()}}
             for b in sprites]
    label_lines = "\n".join(f"{i}: {lab}" for i, lab in enumerate(labels))
    parts.append({"text": _QA_PROMPT.format(labels=label_lines)})
    url = f"{g.GEMINI_API}/models/{model}:generateContent"
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": _QA_SCHEMA}}
    resp = await g._gemini_fetch(http, url, api_keys, body)
    candidates = resp.get("candidates") or []
    rparts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(p.get("text", "") for p in rparts if isinstance(p, dict))
    try:
        raw = json.loads(text)
    except ValueError:
        raw = {}
    return parse_verdicts(raw, len(sprites))


def parse_verdicts(raw: dict, count: int) -> list:
    """Defensive parse: every sprite gets a verdict; unknown/missing -> 'clean'."""
    out = ["clean"] * count
    for v in (raw.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        idx, verdict = v.get("index"), v.get("verdict")
        if isinstance(idx, int) and 0 <= idx < count and verdict in QA_VERDICTS:
            out[idx] = verdict
    return out


# ---- multi-map linking ----

def pick_expandable(objects, limit: int) -> list:
    """The enterables worth turning into interiors: largest boxes first."""
    doors = [o for o in objects if o["category"] == "enterable"]
    doors.sort(key=lambda o: (o["box_2d"][2] - o["box_2d"][0]) * (o["box_2d"][3] - o["box_2d"][1]),
               reverse=True)
    return doors[: max(0, int(limit))]


def map_key_for(label: str) -> str:
    """brewing_shed_door -> brewing_shed; tavern_door_2 -> tavern_2."""
    parts = [p for p in label.split("_") if p]
    suffix = ""
    if parts and parts[-1].isdigit():
        suffix = "_" + parts.pop()
    if parts and parts[-1] in ("door", "gate", "entrance", "doorway", "stairs", "mouth"):
        parts.pop()
    return ("_".join(parts) or "interior") + suffix


def door_return_spawn(door_box) -> list:
    """Where the player reappears on the PARENT map after exiting an interior:
    just below the door's box, centered."""
    y1, x1, y2, x2 = door_box
    return [min(985, y2 + 30), (x1 + x2) // 2]


def interior_entry_exit(objects) -> tuple:
    """(entry_spawn [y,x], exit_label) for an interior map. Prefers a detected
    enterable near the bottom-center (the entrance door the prompt asked for);
    otherwise callers should synthesize an exit strip with synthesize_exit()."""
    best = None
    for obj in objects:
        if obj["category"] != "enterable":
            continue
        y1, x1, y2, x2 = obj["box_2d"]
        cx = (x1 + x2) / 2
        if y2 >= 650 and 200 <= cx <= 800:
            if best is None or y2 > best["box_2d"][2]:
                best = obj
    if best:
        y1, x1, y2, x2 = best["box_2d"]
        return [max(40, y1 - 45), (x1 + x2) // 2], best["label"]
    return [880, 500], None


def synthesize_exit() -> dict:
    """Fallback exit strip at the bottom-center of an interior with no detected door."""
    return {"label": "exit", "category": "enterable", "box_2d": [950, 360, 1000, 640]}


# ---- manifest (format v2: multi-map) ----

def build_map_entry(map_file: str, width: int, height: int, objects: list, spawn,
                    collision: dict | None, sprite_files: dict | None = None) -> dict:
    """One map's entry for world.json. sprite_files maps label ->
    {"file": bundle filename, "crop_px": [top, left, bottom, right]} — crop_px is the
    exact padded pixel box the sprite was cut from, so renderers redraw it in place.
    Enterable objects may carry "link": {"to": <map key>, "spawn": [y, x]}."""
    out_objects = []
    for obj in objects:
        entry = {"label": obj["label"], "category": obj["category"], "box_2d": obj["box_2d"]}
        cut = (sprite_files or {}).get(obj["label"])
        if cut:
            entry["sprite"] = cut["file"]
            entry["crop_px"] = cut["crop_px"]
        if "link" in obj:
            entry["link"] = obj["link"]
        out_objects.append(entry)
    entry = {
        "map": map_file,
        "width": width,
        "height": height,
        "player_spawn": spawn or [500, 500],
        "objects": out_objects,
    }
    if collision:
        entry["collision"] = collision
    return entry


def build_world_manifest(name: str, maps: dict, start: str = "main") -> dict:
    """world.json (format 2): {maps: {key: map_entry}, start}. Single-map worlds are
    just a v2 manifest with one entry."""
    return {
        "format": "carbon-forge-world/2",
        "name": name,
        "coordinate_space": {
            "units": "normalized 0-1000 per map",
            "box_2d": "[ymin, xmin, ymax, xmax] (y before x — Gemini spatial convention)",
        },
        "start": start if start in maps else next(iter(maps)),
        "maps": maps,
    }


def render_preview_html(manifest: dict) -> str:
    """Self-contained playable preview: follow camera, depth-sorted obstacle sprites,
    collision walker, and map transitions through linked enterables. References bundle
    siblings by relative filename, so it works from the results cache URL and from the
    workspace folder alike."""
    return _PREVIEW_TEMPLATE.replace("__WORLD_JSON__", json.dumps(manifest))


_PREVIEW_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World preview</title>
<style>
  html,body{margin:0;height:100%;background:#0e0c14;overflow:hidden}
  canvas{display:block;width:100vw;height:100vh;touch-action:none}
  #hud{position:fixed;left:10px;bottom:8px;color:#cbc6de;font:12px/1.4 monospace;opacity:.8;
       user-select:none;pointer-events:none}
  #mapname{position:fixed;top:12px;left:50%;transform:translateX(-50%);color:#efe9ff;
       font:bold 14px/1 monospace;letter-spacing:2px;text-transform:uppercase;opacity:0;
       transition:opacity .4s;text-shadow:0 2px 6px #000;pointer-events:none}
  #fade{position:fixed;inset:0;background:#000;opacity:0;transition:opacity .28s;pointer-events:none}
</style></head><body>
<canvas id="c"></canvas><div id="mapname"></div><div id="fade"></div>
<div id="hud">WASD / arrows / drag to walk &middot; walk into doors &middot; G collision grid &middot; Z zoom</div>
<script>
const WORLD = __WORLD_JSON__;
const N = 1000, cv = document.getElementById('c'), ctx = cv.getContext('2d');
const fadeEl = document.getElementById('fade'), nameEl = document.getElementById('mapname');
const ZOOMS = [2.2, 1.5, 1.0];  // viewport shows 1/zoom of the map height
let zoomIdx = 0, showGrid = false, transitioning = false, linkCooldown = 0;
const state = { key: WORLD.start, map: WORLD.maps[WORLD.start] };
const player = { y: state.map.player_spawn[0], x: state.map.player_spawn[1], speed: 230 };
const assets = {};  // per map key: {bg: Image, sprites: {label: Image}, blocked: Set}
function loadMap(key) {
  if (assets[key]) return assets[key];
  const m = WORLD.maps[key], bg = new Image(); bg.src = m.map;
  const sprites = {};
  for (const o of m.objects) if (o.sprite) { const im = new Image(); im.src = o.sprite; sprites[o.label] = im; }
  return assets[key] = { bg, sprites, blocked: new Set(m.collision ? m.collision.blocked : []) };
}
loadMap(state.key);
for (const key of Object.keys(WORLD.maps)) loadMap(key);  // prefetch the rest
function announce(key) {
  nameEl.textContent = key.replace(/_/g, ' ');
  nameEl.style.opacity = 1; clearTimeout(announce._t);
  announce._t = setTimeout(() => nameEl.style.opacity = 0, 1600);
}
announce(state.key);
const keys = {}, drag = { on:false, x:0, y:0, dx:0, dy:0 };
addEventListener('keydown', e => { const k = e.key.toLowerCase(); keys[k] = true;
  if (k === 'g') showGrid = !showGrid; if (k === 'z') zoomIdx = (zoomIdx + 1) % ZOOMS.length; });
addEventListener('keyup', e => keys[e.key.toLowerCase()] = false);
cv.addEventListener('pointerdown', e => { drag.on = true; drag.x = e.clientX; drag.y = e.clientY; });
addEventListener('pointermove', e => { if (drag.on) { drag.dx = e.clientX - drag.x; drag.dy = e.clientY - drag.y; } });
addEventListener('pointerup', () => { drag.on = false; drag.dx = drag.dy = 0; });
function blocked(x, y) {
  if (x < 8 || x > N - 8 || y < 4 || y > N - 4) return true;
  const col = state.map.collision; if (!col) return false;
  const c = Math.min(col.cols - 1, Math.max(0, Math.floor(x * col.cols / N)));
  const r = Math.min(col.rows - 1, Math.max(0, Math.floor(y * col.rows / N)));
  return assets[state.key].blocked.has(r * col.cols + c);
}
function linkAt(x, y) {
  for (const o of state.map.objects) {
    if (!o.link) continue;
    const [y1, x1, y2, x2] = o.box_2d, m = 12;  // slightly forgiving trigger
    if (x >= x1 - m && x <= x2 + m && y >= y1 - m && y <= y2 + m) return o.link;
  }
  return null;
}
function travel(link) {
  transitioning = true; fadeEl.style.opacity = 1;
  setTimeout(() => {
    state.key = link.to; state.map = WORLD.maps[link.to];
    player.y = link.spawn[0]; player.x = link.spawn[1];
    linkCooldown = 1.0; announce(state.key);
    fadeEl.style.opacity = 0; transitioning = false;
  }, 300);
}
let last = performance.now();
function tick(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (!transitioning) {
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
    linkCooldown = Math.max(0, linkCooldown - dt);
    if (linkCooldown === 0 && (mx || my)) {
      const link = linkAt(player.x, player.y);
      if (link) travel(link);
    }
  }
  draw(); requestAnimationFrame(tick);
}
let camX = null, camY = null;
function draw() {
  const a = assets[state.key], m = state.map;
  const W = innerWidth * devicePixelRatio, H = innerHeight * devicePixelRatio;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const mw = m.width, mh = m.height;
  const zoom = ZOOMS[zoomIdx];
  // scale so the viewport shows (map height / zoom), never upscaling past fit-width
  let scale = H / (mh / zoom);
  const viewW = W / scale, viewH = H / scale;
  const px = player.x / N * mw, py = player.y / N * mh;
  let tx = px - viewW / 2, ty = py - viewH / 2;
  tx = Math.max(0, Math.min(Math.max(0, mw - viewW), tx));
  ty = Math.max(0, Math.min(Math.max(0, mh - viewH), ty));
  if (camX === null) { camX = tx; camY = ty; }
  camX += (tx - camX) * 0.12; camY += (ty - camY) * 0.12;
  ctx.setTransform(scale, 0, 0, scale, -camX * scale, -camY * scale);
  ctx.clearRect(camX, camY, viewW, viewH);
  ctx.imageSmoothingEnabled = true;
  if (a.bg.complete) ctx.drawImage(a.bg, 0, 0, mw, mh);
  else { ctx.fillStyle = '#1a1626'; ctx.fillRect(0, 0, mw, mh); }
  const layers = m.objects.filter(o => o.sprite).map(o => ({ o, foot: o.box_2d[2] }))
    .concat([{ player: true, foot: player.y }]);
  layers.sort((x, y) => x.foot - y.foot);
  for (const l of layers) {
    if (l.player) { drawPlayer(px, py, mh); continue; }
    const im = a.sprites[l.o.label];
    if (!im || !im.complete || !im.naturalWidth || !l.o.crop_px) continue;
    const [t, lft, b, r] = l.o.crop_px;
    ctx.drawImage(im, lft, t, r - lft, b - t);
  }
  if (showGrid && m.collision) {
    const col = m.collision, cw = mw / col.cols, ch = mh / col.rows;
    ctx.fillStyle = 'rgba(255,60,60,0.28)';
    for (const idx of a.blocked) ctx.fillRect((idx % col.cols) * cw, Math.floor(idx / col.cols) * ch, cw, ch);
    ctx.fillStyle = 'rgba(80,160,255,0.35)';
    for (const o of m.objects) if (o.link) {
      const [y1, x1, y2, x2] = o.box_2d;
      ctx.fillRect(x1 / N * mw, y1 / N * mh, (x2 - x1) / N * mw, (y2 - y1) / N * mh);
    }
  }
}
function drawPlayer(px, py, mh) {
  const s = mh / 26;
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
