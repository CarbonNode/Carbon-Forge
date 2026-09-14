"""Pure, stdlib-only helpers shared by the 3D→pixel sprite bake.

This module is imported on BOTH sides of the Blender process boundary:
  - backend/blender_bake.py runs INSIDE Blender (its own Python, no PIL, no
    numpy guarantees) and needs the frame sampling + direction conventions,
  - backend/sprite_bake.py / the MCP tools run in the forge service and need
    the same conventions to tag rows and validate a spec.
Keep it dependency-free so the two never disagree.

Conventions (the contract every consumer relies on):
  - Direction 0 is the character FACING THE CAMERA ("s", screen-down). The
    model is rotated COUNTER-clockwise (seen from above) by k * 360/N, so its
    face sweeps s → se → e → ne → n → nw → w → sw. A glTF character faces +Z,
    which Blender's importer turns into -Y; the bake camera sits on -Y.
  - A row of the sheet = one (action, direction) pair, tagged "<action>_<dir>"
    (or just "<dir>" when only one action is baked).
"""

DIRECTION_TAGS = {
    1: ["s"],
    2: ["s", "n"],
    4: ["s", "e", "n", "w"],
    8: ["s", "se", "e", "ne", "n", "nw", "w", "sw"],
    16: ["s", "sse", "se", "ese", "e", "ene", "ne", "nne",
         "n", "nnw", "nw", "wnw", "w", "wsw", "sw", "ssw"],
}
SUPPORTED_DIRECTIONS = tuple(sorted(DIRECTION_TAGS))

# What a sprite bake asks Blender for. Everything is validated + defaulted
# here so the bpy script can trust its input.
DEFAULT_SPEC = {
    "directions": 8,
    "frames": 8,            # frames sampled per action
    "render_px": 256,       # render resolution (square) — cell * supersample
    "elevation_deg": 35.0,  # camera pitch above the ground plane (0 = side view, 90 = top-down)
    "fov_pad": 1.08,        # extra ortho room around the union bounds
    "engine": "cycles",     # cycles (headless-safe) | workbench (needs a GPU/GL context)
    "samples": 16,          # cycles samples (frames are supersampled + medoid-downsampled anyway)
    "loop": True,           # cycles (walk/idle) drop the last frame (== first); one-shots keep it
    "actions": None,        # None = every action in the file; else list of action names
    "light_azimuth_deg": -35.0,  # key light around the camera axis (negative = from screen-left)
    "light_elevation_deg": 55.0,
    "ambient": 0.55,        # world light strength (flat fill so shadows never go black)
    "key_strength": 3.0,    # sun strength
    "clip_actions": {},     # per-action {"start": f, "end": f} frame overrides
}


def normalize_spec(spec):
    """Fill defaults + validate. Raises ValueError with a readable message."""
    out = dict(DEFAULT_SPEC)
    out.update({k: v for k, v in (spec or {}).items() if v is not None})
    try:
        out["directions"] = int(out["directions"])
    except (TypeError, ValueError):
        raise ValueError("directions must be an integer")
    if out["directions"] not in DIRECTION_TAGS:
        raise ValueError(f"directions must be one of {SUPPORTED_DIRECTIONS}")
    out["frames"] = max(1, min(64, int(out["frames"])))
    out["render_px"] = max(16, min(2048, int(out["render_px"])))
    out["elevation_deg"] = float(max(0.0, min(89.0, float(out["elevation_deg"]))))
    out["fov_pad"] = float(max(1.0, min(2.0, float(out["fov_pad"]))))
    out["samples"] = max(1, min(256, int(out["samples"])))
    eng = str(out["engine"]).lower()
    if eng not in ("cycles", "workbench", "eevee"):
        raise ValueError("engine must be cycles, workbench or eevee")
    out["engine"] = eng
    out["loop"] = bool(out["loop"])
    if out["actions"] is not None:
        acts = [str(a) for a in out["actions"] if str(a).strip()]
        out["actions"] = acts or None
    out["clip_actions"] = dict(out.get("clip_actions") or {})
    return out


def direction_tags(n):
    return list(DIRECTION_TAGS[int(n)])


def direction_angle_deg(index, n):
    """Rotation (about +Z, counter-clockwise from above) that produces direction `index`."""
    return (360.0 / int(n)) * int(index)


def row_tag(action, direction, single_action=False):
    return direction if single_action else f"{action}_{direction}"


def sample_frame_numbers(start, end, count, loop=True):
    """Evenly spaced frame numbers over an action's [start, end].

    loop=True treats the range as a cycle: the sample one step short of `end`
    is the last one, so a walk cycle does not play its first pose twice.
    Frames are rounded to integers (sub-frame sampling makes Cycles re-evaluate
    the rig between keys, which looks the same but costs time)."""
    start, end = float(start), float(end)
    count = max(1, int(count))
    if end <= start or count == 1:
        return [int(round(start))] * count
    span = end - start
    if loop:
        step = span / count
        vals = [start + step * i for i in range(count)]
    else:
        step = span / (count - 1)
        vals = [start + step * i for i in range(count)]
    return [int(round(v)) for v in vals]


def resolve_action_names(available, wanted=None):
    """Pick which actions of a GLB to bake. `wanted` names match case-insensitively
    and as substrings (Meshy names its clips like 'Armature|Casual_Walk'); None =
    every action, in file order."""
    available = list(available)
    if not wanted:
        return available
    out = []
    for w in wanted:
        wl = str(w).lower()
        hit = next((a for a in available if a.lower() == wl), None) or \
            next((a for a in available if wl in a.lower()), None)
        if hit is None:
            raise ValueError(f"action '{w}' not in the model — available: {', '.join(available) or 'none'}")
        if hit not in out:
            out.append(hit)
    return out
