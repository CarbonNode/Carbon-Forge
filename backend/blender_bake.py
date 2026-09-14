"""Runs INSIDE Blender: an animated GLB → orthographic frames from N directions.

Invoked by backend/sprite_bake.py as a subprocess, either way:
    blender -b --python backend/blender_bake.py -- <spec.json>
    <python with the bpy wheel> backend/blender_bake.py <spec.json>

spec.json (see backend.bake_spec.DEFAULT_SPEC for every key):
    {"glb": "/path/model.glb", "out_dir": "/path/frames", "directions": 8,
     "frames": 8, "render_px": 256, "elevation_deg": 35, "actions": null, ...}

What it does — the Warcraft II / Diablo recipe, automated:
  1. import the GLB (armature + actions), parent everything to one root empty
  2. measure the union bounds of EVERY sampled frame of EVERY action so the
     orthographic camera frames the whole animation — a sword swing or a death
     collapse must not clip, and the ground line must not move between frames
  3. for each action × direction: rotate the root about +Z, sample the action's
     frames, render each with a transparent film and Standard view transform
     (no AgX/Filmic tone-mapping — pixel art wants the texture's real colors)
  4. write <out_dir>/<action>_<dir>_<ii>.png + manifest.json (rows, tags, files)

Writes ONLY into out_dir. Prints one line 'BAKE_RESULT <json>' at the end so
the caller can parse it without scraping Blender's chatter.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from backend import bake_spec as bs  # noqa: E402  (stdlib-only module)


def _spec_path():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: blender -b --python blender_bake.py -- spec.json")
    return argv[0]


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # The factory file (and the bpy wheel's) can still carry orphan datablocks; a stale
    # mesh in bpy.data would pollute the bounds pass, so start from truly nothing.
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials,
                 bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)
    scene = bpy.context.scene
    scene.frame_set(0)
    return scene


def _import_glb(path):
    before = {o.name for o in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o.name not in before]
    if not new:
        raise RuntimeError("the GLB imported no objects")
    return new


def _root_for(objects):
    """One empty above every imported top-level object so a single rotation turns the whole character."""
    root = bpy.data.objects.new("SpriteRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for o in objects:
        if o.parent is None or o.parent not in objects:
            o.parent = root
            o.matrix_parent_inverse = root.matrix_world.inverted()
    return root


def _assign_action(armatures, action):
    for arm in armatures:
        ad = arm.animation_data or arm.animation_data_create()
        ad.action = action
        # Blender 4.4+ slotted actions: an action assigned without a slot does nothing.
        slots = getattr(action, "slots", None)
        if slots and hasattr(ad, "action_slot"):
            try:
                ad.action_slot = slots[0]
            except Exception:  # noqa: BLE001 — older builds raise on the attr
                pass


def _bounds(meshes, depsgraph, center_xy=None):
    """World-space (min, max) of the evaluated (deformed) meshes at the current frame,
    plus the largest horizontal distance of any vertex from center_xy (0 if None).
    Uses real vertices: a skinned mesh's bound_box is the REST pose box, which both
    under-measures a stretched-out attack and pads a standing figure by a metre."""
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    radius = 0.0
    for o in meshes:
        ev = o.evaluated_get(depsgraph)
        me = ev.to_mesh()
        try:
            mw = ev.matrix_world
            for v in me.vertices:
                p = mw @ v.co
                if p.x < lo.x: lo.x = p.x
                if p.y < lo.y: lo.y = p.y
                if p.z < lo.z: lo.z = p.z
                if p.x > hi.x: hi.x = p.x
                if p.y > hi.y: hi.y = p.y
                if p.z > hi.z: hi.z = p.z
                if center_xy is not None:
                    d = math.hypot(p.x - center_xy[0], p.y - center_xy[1])
                    if d > radius:
                        radius = d
        finally:
            ev.to_mesh_clear()
    if lo.x == math.inf:
        raise RuntimeError("no mesh bounds — the GLB has no renderable geometry")
    return lo, hi, radius


def _setup_render(scene, spec):
    r = scene.render
    r.resolution_x = r.resolution_y = int(spec["render_px"])
    r.resolution_percentage = 100
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    r.use_persistent_data = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    eng = spec["engine"]
    if eng == "cycles":
        scene.render.engine = "CYCLES"
        cy = scene.cycles
        cy.device = "CPU"
        cy.samples = int(spec["samples"])
        cy.use_adaptive_sampling = True
        cy.use_denoising = False
        cy.max_bounces = 2
        cy.diffuse_bounces = 1
        cy.glossy_bounces = 1
        cy.transmission_bounces = 1
        cy.transparent_max_bounces = 4
        cy.caustics_reflective = False
        cy.caustics_refractive = False
        cy.film_transparent_glass = True
    elif eng == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        sh = scene.display.shading
        sh.light = "STUDIO"
        sh.color_type = "TEXTURE"
        sh.show_shadows = False
        sh.show_cavity = False
        sh.use_dof = False
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.types, "SceneEEVEE") else "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = max(4, int(spec["samples"]))


def _setup_world(scene, spec):
    world = bpy.data.worlds.new("SpriteWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = float(spec["ambient"])


def _setup_light(scene, spec):
    data = bpy.data.lights.new("SpriteKey", "SUN")
    data.energy = float(spec["key_strength"])
    data.angle = math.radians(8.0)  # soft-ish terminator, still reads as one key light
    sun = bpy.data.objects.new("SpriteKey", data)
    scene.collection.objects.link(sun)
    az = math.radians(float(spec["light_azimuth_deg"]))
    el = math.radians(float(spec["light_elevation_deg"]))
    # a sun's -Z axis is its direction; point it from (az, el) around the camera's -Y side
    sun.rotation_euler = (math.radians(90.0) - el, 0.0, az)
    return sun


def _setup_camera(scene, spec, center, radius, height):
    cam_data = bpy.data.cameras.new("SpriteCam")
    cam_data.type = "ORTHO"
    cam_data.clip_start = 0.01
    cam = bpy.data.objects.new("SpriteCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    elev = math.radians(float(spec["elevation_deg"]))
    extent_w = 2.0 * radius
    extent_h = height * math.cos(elev) + 2.0 * radius * math.sin(elev)
    cam_data.ortho_scale = max(extent_w, extent_h, 1e-3) * float(spec["fov_pad"])
    dist = 10.0 + 4.0 * max(radius, height)
    cam_data.clip_end = dist * 4.0
    direction = Vector((0.0, -math.cos(elev), math.sin(elev)))
    cam.location = center + direction * dist
    cam.rotation_euler = (math.radians(90.0) - elev, 0.0, 0.0)
    return cam


def bake(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    spec = bs.normalize_spec(raw)
    glb, out_dir = raw["glb"], raw["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    scene = _reset_scene()
    objects = _import_glb(glb)
    # Only what actually renders: the glTF importer also creates an unlinked "Icosphere"
    # it uses as a custom bone shape, and a file may carry hide_render helpers.
    in_scene = {o.name for o in scene.objects}
    objects = [o for o in objects if o.name in in_scene]

    def _renders(o):
        if o.hide_render or any(c.hide_render for c in o.users_collection):
            return False  # the importer parks its bone-shape sphere in 'glTF_not_exported'
        try:
            return o.visible_get()
        except Exception:  # noqa: BLE001 — no view layer in some background setups
            return True

    meshes = [o for o in objects if o.type == "MESH" and _renders(o)]
    armatures = [o for o in objects if o.type == "ARMATURE"]
    root = _root_for(objects)

    available = [a.name for a in bpy.data.actions]
    wanted = bs.resolve_action_names(available, spec["actions"]) if available else []
    clips = []
    if wanted:
        for name in wanted:
            act = bpy.data.actions[name]
            f0, f1 = act.frame_range
            ov = spec["clip_actions"].get(name) or {}
            f0, f1 = float(ov.get("start", f0)), float(ov.get("end", f1))
            clips.append({"name": name, "action": act, "start": f0, "end": f1,
                          "frames": bs.sample_frame_numbers(f0, f1, spec["frames"], spec["loop"])})
    else:
        clips.append({"name": "still", "action": None, "start": 0.0, "end": 0.0, "frames": [0]})

    depsgraph = bpy.context.evaluated_depsgraph_get()
    # Union bounds over every clip + frame at direction 0. Rotation about the vertical axis
    # keeps the vertical extent; the horizontal radius about the centre bounds any yaw
    # (a vertex's distance from the spin axis is invariant under the spin).
    samples = []
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for clip in clips:
        if clip["action"] is not None and armatures:
            _assign_action(armatures, clip["action"])
        for fr in clip["frames"]:
            scene.frame_set(fr)
            depsgraph.update()
            a, b, _ = _bounds(meshes, depsgraph)
            lo = Vector((min(lo.x, a.x), min(lo.y, a.y), min(lo.z, a.z)))
            hi = Vector((max(hi.x, b.x), max(hi.y, b.y), max(hi.z, b.z)))
            samples.append((clip, fr))
    cx, cy = (lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0
    height = hi.z - lo.z
    radius = 0.0
    for clip, fr in samples:
        if clip["action"] is not None and armatures:
            _assign_action(armatures, clip["action"])
        scene.frame_set(fr)
        depsgraph.update()
        _, _, r = _bounds(meshes, depsgraph, center_xy=(cx, cy))
        radius = max(radius, r)
    center = Vector((cx, cy, (lo.z + hi.z) / 2.0))

    # Spin about the character's own horizontal centre, not the file origin.
    root.location = (cx, cy, 0.0)
    for o in objects:
        if o.parent is root:
            o.matrix_parent_inverse = root.matrix_world.inverted()

    _setup_render(scene, spec)
    _setup_world(scene, spec)
    sun = _setup_light(scene, spec)
    _setup_camera(scene, spec, center, radius, height)

    n_dir = spec["directions"]
    tags = bs.direction_tags(n_dir)
    single = len(clips) == 1
    rows, rendered = [], 0
    for clip in clips:
        if clip["action"] is not None and armatures:
            _assign_action(armatures, clip["action"])
        safe_action = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip["name"]).strip("_") or "action"
        for di in range(n_dir):
            root.rotation_euler = (0.0, 0.0, math.radians(bs.direction_angle_deg(di, n_dir)))
            # the key light turns WITH the character so every facing is lit the same way —
            # a sprite lit from screen-left in all 8 directions reads as one consistent set
            files = []
            for i, fr in enumerate(clip["frames"]):
                scene.frame_set(fr)
                fname = f"{safe_action}_{tags[di]}_{i:02d}.png"
                scene.render.filepath = os.path.join(out_dir, fname)
                bpy.ops.render.render(write_still=True)
                files.append(fname)
                rendered += 1
            rows.append({"action": clip["name"], "direction": tags[di],
                         "tag": bs.row_tag(safe_action, tags[di], single_action=single),
                         "files": files, "frame_numbers": clip["frames"],
                         "start": clip["start"], "end": clip["end"]})
    del sun
    manifest = {
        "glb": glb, "render_px": spec["render_px"], "directions": n_dir, "direction_tags": tags,
        "actions_available": available, "actions_baked": [c["name"] for c in clips],
        "frames_per_action": spec["frames"], "loop": spec["loop"], "engine": spec["engine"],
        "elevation_deg": spec["elevation_deg"],
        "bounds": {"min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z],
                   "height": height, "radius": radius},
        "rows": rows, "frames_rendered": rendered,
        "blender": bpy.app.version_string,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("BAKE_RESULT " + json.dumps({"frames_rendered": rendered, "rows": len(rows),
                                       "actions": manifest["actions_baked"],
                                       "manifest": os.path.join(out_dir, "manifest.json")}))
    return manifest


if __name__ == "__main__":
    bake(_spec_path())
