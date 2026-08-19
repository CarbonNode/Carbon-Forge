import sys
import json
import os
import base64
import traceback
from flask import Flask, request, Response, jsonify

import pixel_art
import processing
from processing import (
    PipelineOptions, run_pipeline, run_split_pipeline, parse_colors,
    split_sprites, get_session, DEFAULT_MODEL, AVAILABLE_MODELS,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


def _safe_json(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def _opts_from_headers(req) -> PipelineOptions:
    return PipelineOptions(
        model=req.headers.get("X-Model", DEFAULT_MODEL),
        skip_bg=req.headers.get("X-Skip-Bg", "false") == "true",
        alpha_matting=req.headers.get("X-Alpha-Matting", "false") == "true",
        fg_threshold=int(req.headers.get("X-FG-Threshold", "240")),
        bg_threshold=int(req.headers.get("X-BG-Threshold", "10")),
        erode_size=int(req.headers.get("X-Erode-Size", "10")),
        color_remove=req.headers.get("X-Color-Remove", "false") == "true",
        colors=parse_colors(_safe_json(req.headers.get("X-Colors", "[]"))),
        color_auto_detect=req.headers.get("X-Color-Auto-Detect", "false") == "true",
        color_tolerance=int(req.headers.get("X-Color-Tolerance", "20")),
        edge_smooth=req.headers.get("X-Edge-Smooth", "false") == "true",
        edge_strength=int(req.headers.get("X-Edge-Strength", "50")),
        edge_trim=int(req.headers.get("X-Edge-Trim", "0")),
        auto_trim=req.headers.get("X-Auto-Trim", "false") == "true",
        watermark_remove=req.headers.get("X-Watermark-Remove", "false") == "true",
        watermark_position=req.headers.get("X-Watermark-Position", "bottom-right"),
        watermark_size_pct=int(req.headers.get("X-Watermark-Size", "15")),
    )


# --- Startup ---

print("Loading AI model...", flush=True)
get_session(DEFAULT_MODEL)
print("MODEL_READY", flush=True)


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return jsonify(AVAILABLE_MODELS)


@app.get("/watermark-status")
def watermark_status():
    return jsonify({
        "onnxAvailable": processing.HAS_ONNX,
        "modelExists": os.path.exists(processing.LAMA_MODEL_PATH),
        "modelLoaded": processing.lama_session is not None,
        "modelPath": processing.LAMA_MODEL_PATH,
    })


@app.errorhandler(Exception)
def _json_error(e):
    if hasattr(e, "code") and isinstance(getattr(e, "code", None), int):
        return jsonify({"error": str(e)}), e.code
    print(f"Unhandled exception: {e}", flush=True)
    traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.post("/remove-bg")
def remove_bg():
    data = request.get_data()
    if not data:
        return {"error": "No image data"}, 400

    opts = _opts_from_headers(request)
    # Historical behavior: with no explicit colors, the bg color is auto-detected
    # (used only if color_remove is on, but always detected/logged).
    opts.color_auto_detect = opts.color_auto_detect or not opts.colors

    return Response(run_pipeline(data, opts), mimetype="image/png")


@app.post("/split-sprites")
def split_sprites_endpoint():
    data = request.get_data()
    if not data:
        return jsonify({"error": "No image data"}), 400

    opts = _opts_from_headers(request)
    min_area = int(request.headers.get("X-Min-Sprite-Area", "400"))

    sprites = run_split_pipeline(data, opts, min_area)
    encoded = [base64.b64encode(s).decode("ascii") for s in sprites]
    return jsonify({"count": len(encoded), "sprites": encoded})


@app.post("/pixel-refine")
def pixel_refine_endpoint():
    """Refine AI 'fake' pixel art into true low-res pixels (pixel_art engine).
    Options ride in the X-Refine-Options header as JSON (same keyword names as
    pixel_art.refine_pixel_art); the analysis report returns in X-Refine-Report."""
    data = request.get_data()
    if not data:
        return {"error": "No image data"}, 400
    opts = _safe_json(request.headers.get("X-Refine-Options", "{}"))
    if not isinstance(opts, dict):
        opts = {}
    out, report = pixel_art.refine_pixel_art(data, **opts)
    resp = Response(out, mimetype="image/png")
    resp.headers["X-Refine-Report"] = json.dumps(report)
    return resp


@app.post("/split-only")
def split_only_endpoint():
    """Takes an already-transparent PNG and splits into individual sprites."""
    data = request.get_data()
    if not data:
        return {"error": "No image data"}, 400

    min_area = int(request.headers.get("X-Min-Sprite-Area", "400"))
    sprites = split_sprites(data, min_area)
    encoded = [base64.b64encode(s).decode("ascii") for s in sprites]
    return jsonify({"count": len(encoded), "sprites": encoded})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5123
    app.run(host="127.0.0.1", port=port, debug=False)
