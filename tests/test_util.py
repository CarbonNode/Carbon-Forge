import json
import struct
from io import BytesIO

import pytest

from forge_mcp import storage
from forge_mcp import video as v
from forge_mcp.assets3d import ModelError, glb_stats


def test_sniff_audio_and_model_magics():
    assert storage.sniff_mime(b"ID3\x04\x00" + b"\x00" * 20) == "audio/mpeg"
    assert storage.sniff_mime(b"\xff\xfb\x90\x00" + b"\x00" * 20) == "audio/mpeg"
    assert storage.sniff_mime(b"OggS" + b"\x00" * 20) == "audio/ogg"
    assert storage.sniff_mime(b"fLaC" + b"\x00" * 20) == "audio/flac"
    assert storage.sniff_mime(b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 8) == "audio/wav"
    assert storage.sniff_mime(b"RIFF\x24\x00\x00\x00WEBP" + b"\x00" * 8) == "image/webp"
    assert storage.sniff_mime(b"RIFF\x24\x00\x00\x00AVI " + b"\x00" * 8) == "video/x-msvideo"
    assert storage.sniff_mime(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 8) == "audio/mp4"
    assert storage.sniff_mime(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 8) == "video/mp4"
    assert storage.sniff_mime(b"glTF" + struct.pack("<II", 2, 32) + b"\x00" * 20) == "model/gltf-binary"
    # untouched: existing image formats still sniff
    assert storage.sniff_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == "image/png"


def test_audio_convert_cmd():
    cmd = v.build_audio_convert_cmd("in.m4a", "out.mp3", "mp3", bitrate_kbps=128)
    assert "libmp3lame" in cmd and "128k" in cmd and "-vn" in cmd
    wav = v.build_audio_convert_cmd("in.mp3", "out.wav", "wav")
    assert "pcm_s16le" in wav and "-b:a" not in wav  # no bitrate for lossless
    flac = v.build_audio_convert_cmd("in.mp3", "out.flac", "flac", bitrate_kbps=320)
    assert "-b:a" not in flac
    opus = v.build_audio_convert_cmd("a.wav", "o.opus", "opus", sample_rate_hz=48000, channels=1)
    assert "libopus" in opus and "48000" in opus and "-ac" in opus
    with pytest.raises(v.VideoError):
        v.build_audio_convert_cmd("a.mp3", "o.aiff", "aiff")


def test_audio_trim_cmd():
    copy_cmd = v.build_audio_trim_cmd("a.mp3", "o.mp3", 1.0, 5.0, codec=None)
    assert "copy" in copy_cmd and "-vn" in copy_cmd
    re_cmd = v.build_audio_trim_cmd("a.mp4", "o.mp3", 1.0, 5.0, codec="libmp3lame")
    assert "libmp3lame" in re_cmd


def _tiny_glb(extensions=None):
    doc = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}, {}]}], "nodes": [{}]}
    if extensions:
        doc["extensionsUsed"] = extensions
    payload = json.dumps(doc).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    return (b"glTF" + struct.pack("<II", 2, 20 + len(payload))
            + struct.pack("<I", len(payload)) + b"JSON" + payload)


def test_glb_stats():
    stats = glb_stats(_tiny_glb())
    assert stats["meshes"] == 1 and stats["primitives"] == 2 and stats["nodes"] == 1
    assert not stats["draco_compressed"]
    assert glb_stats(_tiny_glb(["KHR_draco_mesh_compression"]))["draco_compressed"]
    with pytest.raises(ModelError):
        glb_stats(b"not a glb at all....")


def test_summarize_probe():
    raw = {"format": {"format_name": "mp3", "duration": "12.345", "bit_rate": "192000"},
           "streams": [{"codec_type": "audio", "codec_name": "mp3",
                        "sample_rate": "44100", "channels": 2}]}
    s = v.summarize_probe(raw)
    assert s["container"] == "mp3" and s["duration_s"] == 12.35 and s["bit_rate"] == 192000
    assert s["streams"][0] == {"type": "audio", "codec": "mp3", "sample_rate": 44100, "channels": 2}
    vid = v.summarize_probe({"format": {}, "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
         "avg_frame_rate": "30/1"}]})
    assert vid["streams"][0]["fps"] == 30.0


def test_convert_image_roundtrip():
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    from forge_mcp.imaging import ImagingError, convert_image

    buf = BytesIO()
    Image.new("RGBA", (64, 32), (255, 0, 0, 128)).save(buf, format="PNG")
    png = buf.getvalue()

    jpg = convert_image(png, "jpg", quality=85)
    assert storage.sniff_mime(jpg) == "image/jpeg"
    webp = convert_image(png, "webp", max_dimension=32)
    with Image.open(BytesIO(webp)) as im:
        assert im.format == "WEBP" and max(im.size) == 32
    ico = convert_image(png, "ico")
    assert ico[:4] == b"\x00\x00\x01\x00"  # ICO header
    with pytest.raises(ImagingError):
        convert_image(png, "tiff")
    with pytest.raises(ImagingError):
        convert_image(b"garbage-not-an-image", "png")
