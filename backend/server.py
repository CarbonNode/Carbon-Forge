import sys
import io
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from flask import Flask, request, Response, jsonify
from rembg import remove, new_session

app = Flask(__name__)

sessions = {}
DEFAULT_MODEL = "u2net"
AVAILABLE_MODELS = [
    {"id": "u2net", "name": "U2-Net (General)", "desc": "Best all-around model"},
    {"id": "u2netp", "name": "U2-Net Lite (Fast)", "desc": "Faster, slightly less accurate"},
    {"id": "u2net_human_seg", "name": "U2-Net Human", "desc": "Optimized for people"},
    {"id": "isnet-general-use", "name": "IS-Net (General)", "desc": "Great edge detection"},
    {"id": "silueta", "name": "Silueta", "desc": "Lightweight general purpose"},
]


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


def get_session(model_id):
    if model_id not in sessions:
        print(f"Loading model: {model_id}...", flush=True)
        sessions[model_id] = new_session(model_id)
        print(f"Model {model_id} ready", flush=True)
    return sessions[model_id]


def remove_colors(img_bytes, colors, tolerance):
    """Post-process: make pixels close to any target color transparent.
    colors: list of (r, g, b) tuples
    tolerance: 1-100, maps to a color distance threshold
    Protects saturated pixels (vivid colors) from being removed.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # tolerance 1-100 maps to color distance 5-255
    max_dist = 5 + (tolerance / 100) * 250

    # Compute saturation (0-255) to protect vivid colors
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    saturation = np.where(cmax > 0, (delta / cmax) * 255, 0)

    # Build combined mask: pixel matches ANY of the target colors
    combined_mask = np.zeros(r.shape, dtype=bool)
    for tr, tg, tb in colors:
        distance = np.sqrt((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2)
        combined_mask |= (distance < max_dist)

    # Only remove low-saturation pixels
    mask = combined_mask & (saturation < 40)
    data[:, :, 3] = np.where(mask, 0, a)

    print(f"Color removal: {len(colors)} colors, tolerance={tolerance}, max_dist={max_dist:.1f}, "
          f"pixels removed={mask.sum()}", flush=True)

    result = Image.fromarray(data.astype(np.uint8))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def smooth_edges(img_bytes, strength, trim_px):
    """Smooth jagged alpha edges, defringe discolored edge pixels, and trim.
    strength: 1-100, controls smoothing amount.
    trim_px: 0-10, number of pixels to erode from edges (removes fringe).
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.float32)
    alpha = data[:, :, 3]

    # Step 1: Trim edges — erode the alpha mask to cut off fringe pixels
    if trim_px > 0:
        opaque_mask = alpha > 128
        eroded = binary_erosion(opaque_mask, iterations=trim_px)
        # Zero out alpha for trimmed pixels
        alpha = np.where(eroded, alpha, 0)
        data[:, :, 3] = alpha

    # Step 2: Defringe — replace edge pixel colors with interior colors
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

    # Step 3: Smooth alpha for anti-aliasing
    sigma_smooth = 0.3 + (strength / 100) * 1.5
    data[:, :, 3] = gaussian_filter(data[:, :, 3], sigma=sigma_smooth)

    print(f"Edge cleanup: strength={strength}, trim={trim_px}px, "
          f"edge_pixels={edge_band.sum() if edge_band.any() else 0}", flush=True)

    result = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


print("Loading AI model...", flush=True)
sessions[DEFAULT_MODEL] = new_session(DEFAULT_MODEL)
print("MODEL_READY", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return jsonify(AVAILABLE_MODELS)


@app.post("/remove-bg")
def remove_bg():
    data = request.get_data()
    if not data:
        return {"error": "No image data"}, 400

    model_id = request.headers.get("X-Model", DEFAULT_MODEL)
    alpha_matting = request.headers.get("X-Alpha-Matting", "false") == "true"
    fg_threshold = int(request.headers.get("X-FG-Threshold", "240"))
    bg_threshold = int(request.headers.get("X-BG-Threshold", "10"))
    erode_size = int(request.headers.get("X-Erode-Size", "10"))
    color_remove = request.headers.get("X-Color-Remove", "false") == "true"
    color_tolerance = int(request.headers.get("X-Color-Tolerance", "20"))
    edge_smooth = request.headers.get("X-Edge-Smooth", "false") == "true"
    edge_strength = int(request.headers.get("X-Edge-Strength", "50"))
    edge_trim = int(request.headers.get("X-Edge-Trim", "0"))
    # Multiple colors as JSON array: [[r,g,b], [r,g,b], ...]
    colors_json = request.headers.get("X-Colors", "[]")
    try:
        colors = [tuple(c) for c in json.loads(colors_json)]
    except (json.JSONDecodeError, TypeError):
        colors = []
    # Fallback: single color from old headers
    if not colors:
        color_r = int(request.headers.get("X-Color-R", "255"))
        color_g = int(request.headers.get("X-Color-G", "255"))
        color_b = int(request.headers.get("X-Color-B", "255"))
        colors = [(color_r, color_g, color_b)]

    session = get_session(model_id)

    kwargs = dict(session=session)
    if alpha_matting:
        kwargs["alpha_matting"] = True
        kwargs["alpha_matting_foreground_threshold"] = fg_threshold
        kwargs["alpha_matting_background_threshold"] = bg_threshold
        kwargs["alpha_matting_erode_size"] = erode_size

    result = remove(data, **kwargs)

    # Post-process: remove remaining color
    print(f"Settings: model={model_id}, alpha_matting={alpha_matting}, "
          f"color_remove={color_remove}, tolerance={color_tolerance}", flush=True)
    if color_remove and colors:
        result = remove_colors(result, colors, color_tolerance)

    if edge_smooth:
        result = smooth_edges(result, edge_strength, edge_trim)

    return Response(result, mimetype="image/png")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5123
    app.run(host="127.0.0.1", port=port, debug=False)
