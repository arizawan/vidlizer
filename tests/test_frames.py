"""Integration tests for frame extraction (requires ffmpeg)."""
from __future__ import annotations

import json


from vidlizer.core import extract_frames


def test_extract_returns_jpgs(test_video, tmp_path):
    frames = extract_frames(
        test_video, tmp_path,
        scale=320, max_frames=10,
        scene_threshold=0.1, fps=None,
        min_interval=1.0, verbose=False,
    )
    assert len(frames) > 0
    assert all(f.suffix == ".jpg" for f in frames)


def test_timestamps_sidecar_written(test_video, tmp_path):
    extract_frames(
        test_video, tmp_path,
        scale=320, max_frames=10,
        scene_threshold=0.1, fps=None,
        min_interval=1.0, verbose=False,
    )
    ts_json = tmp_path / ".timestamps.json"
    assert ts_json.exists()
    ts_map = json.loads(ts_json.read_text())
    assert isinstance(ts_map, dict)
    assert len(ts_map) > 0


def test_max_frames_cap(test_video, tmp_path):
    # fps=1 on a 5s video would give ~5 frames; cap at 2
    frames = extract_frames(
        test_video, tmp_path,
        scale=320, max_frames=2,
        scene_threshold=0.0, fps=1.0,
        min_interval=0.1, verbose=False,
    )
    assert len(frames) <= 2


def test_fps_mode_extracts_fixed_rate(test_video, tmp_path):
    frames = extract_frames(
        test_video, tmp_path,
        scale=320, max_frames=20,
        scene_threshold=0.0, fps=1.0,
        min_interval=0.0, verbose=False,
    )
    # 5s video at 1fps → ~5 frames
    assert 3 <= len(frames) <= 6


def test_start_end_range(test_video, tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "range").mkdir()

    full = extract_frames(
        test_video, tmp_path / "full",
        scale=320, max_frames=20,
        scene_threshold=0.0, fps=1.0,
        min_interval=0.0, verbose=False,
    )
    ranged = extract_frames(
        test_video, tmp_path / "range",
        scale=320, max_frames=20,
        scene_threshold=0.0, fps=1.0,
        min_interval=0.0, verbose=False,
        start=1.0, end=3.0,
    )
    # Narrower window → fewer frames
    assert len(ranged) < len(full)
    assert len(ranged) >= 1
