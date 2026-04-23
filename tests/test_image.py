"""Tests for image-input handling."""
from __future__ import annotations

import base64


from vidlizer.frames import _IMAGE_EXTS, encode_frame


def test_image_ext_set_contains_common_types():
    assert ".jpg" in _IMAGE_EXTS
    assert ".png" in _IMAGE_EXTS
    assert ".webp" in _IMAGE_EXTS
    assert ".pdf" not in _IMAGE_EXTS


def test_encode_frame_returns_image_url_dict(test_image_png):
    result = encode_frame(test_image_png)
    assert result["type"] == "image_url"
    url = result["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_encode_frame_base64_is_valid(test_image_png):
    result = encode_frame(test_image_png)
    url = result["image_url"]["url"]
    b64_part = url.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert len(decoded) > 0


def test_encode_frame_jpeg(test_video, tmp_path):
    from vidlizer.frames import extract_frames
    frames = extract_frames(
        test_video, tmp_path, scale=320, max_frames=1,
        scene_threshold=0.0, fps=1.0, min_interval=0.0, verbose=False,
    )
    assert frames
    result = encode_frame(frames[0])
    assert "image/jpeg" in result["image_url"]["url"]
