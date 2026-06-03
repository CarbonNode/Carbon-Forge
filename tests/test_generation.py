import base64

import pytest

from forge_mcp import generation as g


def test_model_aliases():
    assert g.resolve_image_model("imagen-4") == "imagen-4.0-generate-001"
    assert g.resolve_image_model("imagen-4-fast") == "imagen-4.0-fast-generate-001"
    assert g.resolve_image_model("imagen-4-ultra") == "imagen-4.0-ultra-generate-001"
    assert g.resolve_video_model("veo-3") == "veo-3.0-generate-001"
    assert g.resolve_video_model("veo-3-fast") == "veo-3.0-fast-generate-001"
    assert g.resolve_video_model("veo-2") == "veo-2.0-generate-001"
    with pytest.raises(g.GenerationError):
        g.resolve_image_model("dalle-3")


def test_max_batch():
    assert g.imagen_max_batch("imagen-4.0-ultra-generate-001") == 1
    assert g.imagen_max_batch("imagen-4.0-generate-001") == 4


def test_parse_imagen_predictions():
    b64 = base64.b64encode(b"PNGDATA").decode()
    assert g.parse_imagen_predictions({"predictions": [{"bytesBase64Encoded": b64}]}) == [b"PNGDATA"]
    assert g.parse_imagen_predictions({"generatedImages": [{"image": {"imageBytes": b64}}]}) == [b"PNGDATA"]
    assert g.parse_imagen_predictions({}) == []


def test_parse_gemini_parts():
    b64 = base64.b64encode(b"IMG").decode()
    json_resp = {"candidates": [{"content": {"parts": [
        {"text": "here you go"},
        {"inlineData": {"mimeType": "image/png", "data": b64}},
    ]}}]}
    assert g.parse_gemini_parts(json_resp) == [b"IMG"]


def test_parse_veo_done_extracts_sample():
    done = {"done": True, "response": {"generateVideoResponse": {"generatedSamples": [
        {"video": {"uri": "https://dl/v.mp4"}}]}}}
    assert g.parse_veo_done(done) == {"uri": "https://dl/v.mp4", "inline_b64": None}


def test_parse_veo_done_safety_filtered():
    with pytest.raises(g.GenerationError, match="safety"):
        g.parse_veo_done({"done": True, "response": {}})


def test_veo_request_body_audio_only_for_veo3():
    b3 = g.build_veo_body("veo-3.0-generate-001", "a dog", None, "16:9", 8, True)
    assert b3["parameters"]["generateAudio"] is True
    b2 = g.build_veo_body("veo-2.0-generate-001", "a dog", None, "16:9", 8, True)
    assert "generateAudio" not in b2["parameters"]
