"""Replicate helpers: schema summarizing, output walking, extension picking."""
from forge_mcp import replicate_api as R
from forge_mcp.storage import sniff_mime


def test_summarize_inputs_orders_resolves_enums_and_flags_required():
    version = {"openapi_schema": {"components": {"schemas": {
        "Input": {
            "required": ["prompt"],
            "properties": {
                "output_format": {"allOf": [{"$ref": "#/components/schemas/output_format"}],
                                  "default": "webp", "x-order": 2,
                                  "description": "Format of the output images"},
                "prompt": {"type": "string", "x-order": 0, "description": "Prompt"},
                "num_outputs": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4,
                                "x-order": 1},
            },
        },
        "output_format": {"type": "string", "enum": ["webp", "jpg", "png"]},
    }}}}
    rows = R.summarize_inputs(version)
    assert [r["name"] for r in rows] == ["prompt", "num_outputs", "output_format"]
    assert rows[0]["required"] is True and "required" not in rows[1]
    assert rows[1]["min"] == 1 and rows[1]["max"] == 4
    assert rows[2]["enum"] == ["webp", "jpg", "png"] and rows[2]["default"] == "webp"


def test_collect_urls_and_output_text():
    assert R.collect_file_urls("https://replicate.delivery/x/out.webp") == [
        "https://replicate.delivery/x/out.webp"]
    nested = {"images": ["https://a/1.png", "https://a/2.png"], "seed": 7,
              "extra": {"video": "https://a/v.mp4"}}
    assert R.collect_file_urls(nested) == ["https://a/1.png", "https://a/2.png",
                                           "https://a/v.mp4"]
    assert R.collect_file_urls({"n": 3}) == []
    # LLM token streams join to text; URL outputs are NOT text
    assert R.output_text(["Hel", "lo ", "world"]) == "Hello world"
    assert R.output_text("plain answer") == "plain answer"
    assert R.output_text(["https://a/1.png"]) is None
    assert R.output_text({"k": "v"}) is None


def test_ext_for_url_suffix_then_sniff():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    assert R.ext_for("https://x/y/out.webp?sig=abc", png, sniff_mime) == "webp"
    assert R.ext_for("https://x/y/out.JPEG", png, sniff_mime) == "jpg"
    assert R.ext_for("https://x/y/stream", png, sniff_mime) == "png"          # no suffix -> sniff
    assert R.ext_for("https://x/y/blob", b"\x00" * 20, sniff_mime) == "bin"   # unknown bytes


def test_model_brief():
    row = R.model_brief({"owner": "black-forest-labs", "name": "flux-schnell",
                         "description": "d" * 300, "run_count": 5, "is_official": True})
    assert row["model"] == "black-forest-labs/flux-schnell"
    assert row["official"] is True and len(row["description"]) == 201
