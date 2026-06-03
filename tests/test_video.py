import pytest

from forge_mcp import video as v


def test_to_seconds():
    assert v.to_seconds("90") == 90.0
    assert v.to_seconds("01:30") == 90.0
    assert v.to_seconds("00:01:30.5") == 90.5
    assert v.to_seconds(12.5) == 12.5
    with pytest.raises(v.VideoError):
        v.to_seconds("abc")


def test_trim_cmd_stream_copy_vs_reencode():
    copy_cmd = v.build_trim_cmd("in.mp4", "out.mp4", 1.0, 5.0, reencode=False)
    assert "-c" in copy_cmd and "copy" in copy_cmd
    re_cmd = v.build_trim_cmd("in.mp4", "out.mp4", 1.0, 5.0, reencode=True)
    assert "libx264" in re_cmd


def test_convert_cmd_formats():
    assert "libx264" in v.build_convert_cmd("a.webm", "out.mp4", "mp4", crf=23, scale=None)
    assert "libvpx-vp9" in v.build_convert_cmd("a.mp4", "out.webm", "webm", crf=32, scale=720)
    gif = v.build_convert_cmd("a.mp4", "out.gif", "gif", crf=None, scale=480)
    assert "palettegen" in " ".join(gif) and "paletteuse" in " ".join(gif)
    with pytest.raises(v.VideoError):
        v.build_convert_cmd("a.mp4", "out.avi", "avi", crf=None, scale=None)


def test_frames_cmd():
    ts = v.build_frame_cmd("a.mp4", "out.png", 2.5, "png")
    assert "-ss" in ts and "2.5" in ts
