"""Pure image-processing engine shared by the desktop backend (server.py) and the
hosted MCP service (forge_mcp). No Flask, no HTTP — bytes in, bytes out.

Functions here were extracted verbatim from server.py; do not fork the logic."""
import io
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label
from rembg import remove, new_session

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

sessions = {}
DEFAULT_MODEL = "u2net"
AVAILABLE_MODELS = [
    {"id": "u2net", "name": "U2-Net (General)", "desc": "Best all-around model"},
    {"id": "u2netp", "name": "U2-Net Lite (Fast)", "desc": "Faster, slightly less accurate"},
    {"id": "u2net_human_seg", "name": "U2-Net Human", "desc": "Optimized for people"},
    {"id": "isnet-general-use", "name": "IS-Net (General)", "desc": "Great edge detection"},
    {"id": "silueta", "name": "Silueta", "desc": "Lightweight general purpose"},
]

# LaMa watermark removal model — env override (hosted service) > frozen exe dir
# (packaged desktop builds) > module dir (dev)
_env_lama = os.environ.get("FORGE_LAMA_DIR")
if _env_lama:
    LAMA_MODEL_DIR = _env_lama
elif getattr(sys, 'frozen', False):
    LAMA_MODEL_DIR = os.path.join(os.path.dirname(sys.executable), "models")
else:
    LAMA_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LAMA_MODEL_PATH = os.path.join(LAMA_MODEL_DIR, "lama_fp32.onnx")
LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
LAMA_INPUT_SIZE = 512
lama_session = None


def get_session(model_id):
    if model_id not in sessions:
        print(f"Loading model: {model_id}...", flush=True)
        sessions[model_id] = new_session(model_id)
        print(f"Model {model_id} ready", flush=True)
    return sessions[model_id]


# --- Watermark Removal (LaMa inpainting) ---

def download_lama_model():
    """Download LaMa ONNX model if not present."""
    if os.path.exists(LAMA_MODEL_PATH):
        return True
    os.makedirs(LAMA_MODEL_DIR, exist_ok=True)
    print(f"Downloading LaMa model to {LAMA_MODEL_PATH}...", flush=True)
    try:
        import urllib.request
        def progress(block_num, block_size, total_size):
            if total_size > 0 and block_num % 200 == 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"LaMa download: {pct}%", flush=True)
        urllib.request.urlretrieve(LAMA_MODEL_URL, LAMA_MODEL_PATH, reporthook=progress)
        print("LaMa model downloaded successfully", flush=True)
        return True
    except Exception as e:
        print(f"Failed to download LaMa model: {e}", flush=True)
        if os.path.exists(LAMA_MODEL_PATH):
            os.remove(LAMA_MODEL_PATH)
        return False


def get_lama_session():
    """Load or return cached LaMa ONNX session."""
    global lama_session
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime not installed — run: pip install onnxruntime")
    if lama_session is None:
        if not os.path.exists(LAMA_MODEL_PATH):
            if not download_lama_model():
                raise RuntimeError(
                    f"LaMa model not found at {LAMA_MODEL_PATH}. "
                    "Download it manually or check the URL."
                )
        print("Loading LaMa watermark model...", flush=True)
        lama_session = ort.InferenceSession(LAMA_MODEL_PATH)
        print("LaMa model ready", flush=True)
    return lama_session


def remove_watermark(img_bytes, position="bottom-right", height_ratio=0.15, width_ratio=0.15):
    """Remove watermark using LaMa inpainting model.
    Ported from Gemini-Watermark-Remover Chrome extension.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    model_size = LAMA_INPUT_SIZE

    # Resize to model input size (512x512)
    resized = img.resize((model_size, model_size), Image.LANCZOS)
    img_array = np.array(resized, dtype=np.float32) / 255.0

    # Image tensor: (1, 3, H, W) — HWC to CHW
    img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # Mask tensor: (1, 1, H, W) — 1.0 = inpaint, 0.0 = keep
    mask = np.zeros((1, 1, model_size, model_size), dtype=np.float32)
    region_h = int(model_size * height_ratio)
    region_w = int(model_size * width_ratio)

    if position == "bottom-right":
        mask[0, 0, model_size - region_h:, model_size - region_w:] = 1.0
    elif position == "bottom-left":
        mask[0, 0, model_size - region_h:, :region_w] = 1.0
    elif position == "top-right":
        mask[0, 0, :region_h, model_size - region_w:] = 1.0
    elif position == "top-left":
        mask[0, 0, :region_h, :region_w] = 1.0
    elif position == "bottom-center":
        mask[0, 0, model_size - region_h:, :] = 1.0
    elif position == "top-center":
        mask[0, 0, :region_h, :] = 1.0

    # Run LaMa inference
    session = get_lama_session()
    input_names = [inp.name for inp in session.get_inputs()]
    feeds = {}
    for name in input_names:
        if "mask" in name.lower():
            feeds[name] = mask
        else:
            feeds[name] = img_tensor

    outputs = session.run(None, feeds)
    output = outputs[0]  # (1, 3, H, W)

    # Auto-detect value range
    sample = output[0, :, :min(100, model_size), :min(100, model_size)]
    is_normalized = np.max(np.abs(sample)) <= 2.0

    # CHW to HWC
    output_hwc = output[0].transpose(1, 2, 0)
    if is_normalized:
        output_hwc = output_hwc * 255.0
    output_hwc = np.clip(output_hwc, 0, 255).astype(np.uint8)

    processed = Image.fromarray(output_hwc)

    # Compose: only replace watermark region at original resolution
    extended = 1.067  # Slightly larger for seamless blending
    ext_h = height_ratio * extended
    ext_w = width_ratio * extended

    orig_rh = int(orig_h * ext_h)
    orig_rw = int(orig_w * ext_w)
    proc_rh = int(model_size * ext_h)
    proc_rw = int(model_size * ext_w)

    final = img.copy()

    if position == "bottom-right":
        crop = processed.crop((model_size - proc_rw, model_size - proc_rh, model_size, model_size))
        crop = crop.resize((orig_rw, orig_rh), Image.LANCZOS)
        final.paste(crop, (orig_w - orig_rw, orig_h - orig_rh))
    elif position == "bottom-left":
        crop = processed.crop((0, model_size - proc_rh, proc_rw, model_size))
        crop = crop.resize((orig_rw, orig_rh), Image.LANCZOS)
        final.paste(crop, (0, orig_h - orig_rh))
    elif position == "top-right":
        crop = processed.crop((model_size - proc_rw, 0, model_size, proc_rh))
        crop = crop.resize((orig_rw, orig_rh), Image.LANCZOS)
        final.paste(crop, (orig_w - orig_rw, 0))
    elif position == "top-left":
        crop = processed.crop((0, 0, proc_rw, proc_rh))
        crop = crop.resize((orig_rw, orig_rh), Image.LANCZOS)
        final.paste(crop, (0, 0))
    elif position == "bottom-center":
        crop = processed.crop((0, model_size - proc_rh, model_size, model_size))
        crop = crop.resize((orig_w, orig_rh), Image.LANCZOS)
        final.paste(crop, (0, orig_h - orig_rh))
    elif position == "top-center":
        crop = processed.crop((0, 0, model_size, proc_rh))
        crop = crop.resize((orig_w, orig_rh), Image.LANCZOS)
        final.paste(crop, (0, 0))

    print(f"Watermark removed: position={position}, region={height_ratio*100:.0f}%x{width_ratio*100:.0f}%", flush=True)

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    return buf.getvalue()


# --- Post-processing ---

def remove_colors(img_bytes, colors, tolerance):
    """Post-process: make pixels close to any target color transparent."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    max_dist = 5 + (tolerance / 100) * 250

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    saturation = np.where(cmax > 0, (delta / cmax) * 255, 0)

    combined_mask = np.zeros(r.shape, dtype=bool)
    for tr, tg, tb in colors:
        distance = np.sqrt((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2)
        combined_mask |= (distance < max_dist)

    # Saturation filter: low threshold for white/grey bg removal, disabled for chromatic targets
    # If any target color is saturated (> 50), skip the saturation filter entirely
    target_is_chromatic = any(
        (max(tr, tg, tb) - min(tr, tg, tb)) > 50 for tr, tg, tb in colors
    )
    if target_is_chromatic:
        mask = combined_mask
    else:
        mask = combined_mask & (saturation < 40)
    data[:, :, 3] = np.where(mask, 0, a)

    print(f"Color removal: {len(colors)} colors, tolerance={tolerance}, max_dist={max_dist:.1f}, "
          f"pixels removed={mask.sum()}", flush=True)

    result = Image.fromarray(data.astype(np.uint8))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def smooth_edges(img_bytes, strength, trim_px):
    """Smooth jagged alpha edges, defringe discolored edge pixels, and trim."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.float32)
    alpha = data[:, :, 3]

    if trim_px > 0:
        opaque_mask = alpha > 128
        eroded = binary_erosion(opaque_mask, iterations=trim_px)
        alpha = np.where(eroded, alpha, 0)
        data[:, :, 3] = alpha

    opaque = alpha > 200
    transparent = alpha < 10
    defringe_band = 2 + int(strength / 100 * 3)
    edge_band = binary_dilation(transparent, iterations=defringe_band) & opaque
    interior = binary_erosion(opaque, iterations=defringe_band)

    if interior.any() and edge_band.any():
        sigma = 1.0 + (strength / 100) * 3.0
        for ch in range(3):
            channel = data[:, :, ch]
            blurred = gaussian_filter(channel * interior.astype(np.float32), sigma=sigma)
            weight = gaussian_filter(interior.astype(np.float32), sigma=sigma)
            weight = np.maximum(weight, 1e-6)
            clean = blurred / weight
            data[:, :, ch] = np.where(edge_band, clean, channel)

    sigma_smooth = 0.3 + (strength / 100) * 1.5
    data[:, :, 3] = gaussian_filter(data[:, :, 3], sigma=sigma_smooth)

    print(f"Edge cleanup: strength={strength}, trim={trim_px}px, "
          f"edge_pixels={edge_band.sum() if edge_band.any() else 0}", flush=True)

    result = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def clean_edges(img_bytes, color=None, tolerance=32, alpha_threshold=128,
                min_pocket_px=4, max_pocket_frac=0.5, feather=0):
    """Remove leftover background TRAPPED INSIDE a cut-out.

    After background removal, regions the matte couldn't reach because they
    aren't connected to the outer edge — the counters of letters (the holes in
    D / A / e / o), keyholes, the dot of a '?', rings and circles — stay filled
    with solid opaque background. This makes those pockets transparent.

    A pixel is cleared only if BOTH hold: (1) it's the background colour
    (``color``, default white) within ``tolerance``, and (2) its connected blob
    is fully *sealed* — it never touches a transparent pixel or the image
    border. The silhouette edge and every outline touch the transparent
    background, so they're always preserved. (Contrast smooth_edges, which
    feathers a soft semi-transparent HALO; this clears SOLID fills on a hard
    binary matte.)

    color: (r, g, b) target background colour; None → white (255, 255, 255).
    tolerance: colour-match radius in RGB distance (0–255, default 32).
    min_pocket_px: ignore sealed blobs smaller than this (speck guard).
    max_pocket_frac: never clear a blob larger than this fraction of the opaque
        area (guards a legitimately large enclosed fill).
    feather: if > 0, anti-alias the freshly-cut borders by N px (default 0 = crisp).
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img)
    h, w = data.shape[:2]
    alpha = data[:, :, 3]
    opaque = alpha >= alpha_threshold
    if not opaque.any():
        return img_bytes

    if color is None:
        color = (255, 255, 255)
    tr, tg, tb = (int(color[0]), int(color[1]), int(color[2]))
    rgb = data[:, :, :3].astype(np.float32)
    dist = np.sqrt((rgb[:, :, 0] - tr) ** 2 + (rgb[:, :, 1] - tg) ** 2 + (rgb[:, :, 2] - tb) ** 2)
    bgcol = opaque & (dist <= tolerance)
    if not bgcol.any():
        return img_bytes

    # 8-connected components of background-coloured opaque pixels.
    labeled, n = label(bgcol, structure=np.ones((3, 3), dtype=int))

    # "Open" = the transparent background OR the image border. A blob is trapped
    # iff it touches NO open pixel. A bg pixel adjacent (4-conn) to transparency
    # is flagged by dilating the transparent mask; border pixels count as open so
    # a blob that runs off-frame is treated as background, not a trapped pocket.
    transparent = ~opaque
    near_open = binary_dilation(transparent)
    near_open[0, :] = near_open[-1, :] = near_open[:, 0] = near_open[:, -1] = True
    open_labels = set(np.unique(labeled[near_open & bgcol]).tolist())
    open_labels.discard(0)

    sizes = np.bincount(labeled.ravel())
    sprite_area = int(opaque.sum())
    max_px = max_pocket_frac * sprite_area
    remove = np.zeros((h, w), dtype=bool)
    removed = 0
    for lab in range(1, n + 1):
        if lab in open_labels:           # touches the transparent edge → outline/silhouette, keep
            continue
        sz = int(sizes[lab])
        if sz < min_pocket_px or sz > max_px:
            continue
        remove |= labeled == lab
        removed += 1

    new_alpha = alpha.astype(np.float32)
    new_alpha[remove] = 0.0

    if feather > 0 and remove.any():
        # Soften only the newly-opened borders: blur, then take the min so we
        # never *add* opacity back — only ease the hard cut.
        soft = gaussian_filter(new_alpha, sigma=float(feather))
        band = binary_dilation(remove, iterations=max(1, int(feather))) & opaque & ~remove
        new_alpha[band] = np.minimum(new_alpha[band], soft[band])

    data[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
    print(f"Clean edges: removed {removed} trapped pockets ({int(remove.sum())}px), "
          f"color=({tr},{tg},{tb}), tolerance={tolerance}", flush=True)

    result = Image.fromarray(data)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def trim_transparent(img_bytes):
    """Remove transparent padding around the image."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]

    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        return img_bytes

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - 1)
    cmin = max(0, cmin - 1)
    rmax = min(img.height - 1, rmax + 1)
    cmax = min(img.width - 1, cmax + 1)

    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    print(f"Trim: {img.width}x{img.height} -> {cropped.width}x{cropped.height}", flush=True)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def detect_bg_color(img_bytes):
    """Detect the dominant background color by sampling edges of the image."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    data = np.array(img)
    h, w = data.shape[:2]

    # Sample pixels from all 4 edges (8px deep)
    edge = min(8, h // 4, w // 4)
    samples = np.concatenate([
        data[:edge, :, :].reshape(-1, 3),       # top
        data[-edge:, :, :].reshape(-1, 3),       # bottom
        data[:, :edge, :].reshape(-1, 3),        # left
        data[:, -edge:, :].reshape(-1, 3),       # right
    ])

    # Use median for robustness against outliers
    bg = np.median(samples, axis=0).astype(int)
    print(f"Detected background color: ({bg[0]}, {bg[1]}, {bg[2]})", flush=True)
    return (int(bg[0]), int(bg[1]), int(bg[2]))


def split_sprites(img_bytes, min_area=400):
    """Split a transparent image into individual sprites using connected components.
    Returns list of trimmed PNG byte buffers.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]

    # Binary mask of visible pixels
    binary = alpha > 10

    # 8-connectivity so diagonal touches count as same sprite
    struct = np.ones((3, 3), dtype=int)
    labeled, num_features = label(binary, structure=struct)

    print(f"Split: found {num_features} connected components", flush=True)

    sprites = []
    for i in range(1, num_features + 1):
        component = labeled == i

        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        w = cmax - cmin + 1
        h = rmax - rmin + 1

        # Skip tiny components (noise)
        if w * h < min_area or w < 10 or h < 10:
            continue

        # Add 1px padding
        rmin = max(0, rmin - 1)
        cmin = max(0, cmin - 1)
        rmax = min(img.height - 1, rmax + 1)
        cmax = min(img.width - 1, cmax + 1)

        # Extract region
        sprite_data = data[rmin:rmax + 1, cmin:cmax + 1].copy()

        # Zero out pixels that belong to other components
        component_crop = component[rmin:rmax + 1, cmin:cmax + 1]
        sprite_data[:, :, 3] = np.where(component_crop, sprite_data[:, :, 3], 0)

        sprite_img = Image.fromarray(sprite_data)
        buf = io.BytesIO()
        sprite_img.save(buf, format="PNG")
        sprites.append(buf.getvalue())

    # Sort by position (left-to-right, top-to-bottom)
    print(f"Split: {len(sprites)} sprites after filtering", flush=True)
    return sprites


# --- Shared pipelines (the desktop /remove-bg and /split-sprites flows) ---

from dataclasses import dataclass, field


@dataclass
class PipelineOptions:
    model: str = DEFAULT_MODEL
    skip_bg: bool = False
    alpha_matting: bool = False
    fg_threshold: int = 240
    bg_threshold: int = 10
    erode_size: int = 10
    color_remove: bool = False
    colors: list = field(default_factory=list)   # [(r,g,b), ...]
    color_auto_detect: bool = False
    color_tolerance: int = 20
    edge_smooth: bool = False
    edge_strength: int = 50
    edge_trim: int = 0
    auto_trim: bool = False
    watermark_remove: bool = False
    watermark_position: str = "bottom-right"
    watermark_size_pct: int = 15
    clean_iso: bool = False            # punch out sealed trapped-background pockets
    clean_color: tuple = None          # (r,g,b) target bg colour; None → white
    clean_tolerance: int = 32
    clean_min_px: int = 4
    clean_max_frac: float = 0.5
    clean_feather: int = 0


def parse_colors(raw_colors):
    """Hex strings or RGB triples -> [(r,g,b)]. Invalid entries are skipped."""
    colors = []
    for c in raw_colors or []:
        try:
            if isinstance(c, str) and c.startswith("#"):
                h = c.lstrip("#")
                colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
            elif isinstance(c, (list, tuple)) and len(c) >= 3:
                colors.append((int(c[0]), int(c[1]), int(c[2])))
        except (TypeError, ValueError):
            continue
    return colors


def run_pipeline(data: bytes, opts: PipelineOptions) -> bytes:
    """The desktop /remove-bg pipeline: watermark -> bg removal -> colors -> edges -> trim."""
    colors = list(opts.colors)
    if opts.color_auto_detect or (opts.color_remove and not colors):
        colors = [detect_bg_color(data)]

    if opts.watermark_remove:
        ratio = opts.watermark_size_pct / 100
        data = remove_watermark(data, opts.watermark_position, ratio, ratio)

    if not opts.skip_bg:
        session = get_session(opts.model)
        kwargs = dict(session=session)
        if opts.alpha_matting:
            kwargs["alpha_matting"] = True
            kwargs["alpha_matting_foreground_threshold"] = opts.fg_threshold
            kwargs["alpha_matting_background_threshold"] = opts.bg_threshold
            kwargs["alpha_matting_erode_size"] = opts.erode_size
        result = remove(data, **kwargs)

        print(f"Settings: model={opts.model}, alpha_matting={opts.alpha_matting}, "
              f"color_remove={opts.color_remove}, tolerance={opts.color_tolerance}", flush=True)
    else:
        result = data

    if opts.color_remove and colors:
        result = remove_colors(result, colors, opts.color_tolerance)

    if opts.clean_iso:
        result = clean_edges(result, color=opts.clean_color, tolerance=opts.clean_tolerance,
                             min_pocket_px=opts.clean_min_px, max_pocket_frac=opts.clean_max_frac,
                             feather=opts.clean_feather)

    if opts.edge_smooth:
        result = smooth_edges(result, opts.edge_strength, opts.edge_trim)

    if opts.auto_trim:
        result = trim_transparent(result)

    return result


def run_split_pipeline(data: bytes, opts: PipelineOptions, min_area: int = 400) -> list:
    """The desktop /split-sprites pipeline. Returns list of PNG byte buffers."""
    bg_color = detect_bg_color(data)

    if opts.watermark_remove:
        ratio = opts.watermark_size_pct / 100
        data = remove_watermark(data, opts.watermark_position, ratio, ratio)

    session = get_session(opts.model)
    kwargs = dict(session=session)
    if opts.alpha_matting:
        kwargs["alpha_matting"] = True
        kwargs["alpha_matting_foreground_threshold"] = opts.fg_threshold
        kwargs["alpha_matting_background_threshold"] = opts.bg_threshold
        kwargs["alpha_matting_erode_size"] = opts.erode_size
    result = remove(data, **kwargs)

    all_colors = [bg_color] + list(opts.colors)
    result = remove_colors(result, all_colors, max(opts.color_tolerance, 25))

    smooth_str = opts.edge_strength if opts.edge_smooth else 60
    smooth_trim = opts.edge_trim if opts.edge_smooth else 2
    result = smooth_edges(result, smooth_str, smooth_trim)

    return split_sprites(result, min_area)
