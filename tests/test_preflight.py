"""Tests for vidlizer.preflight — estimation helpers and probe_video."""
from __future__ import annotations

import pytest

from vidlizer.preflight import (
    _fmt_cost,
    _fmt_time,
    estimate_cost,
    estimate_frames,
    estimate_time,
    probe_video,
)

_MODELS_PAID = [
    {
        "id": "test/model",
        "input_usd_per_1m": 1.0, "output_usd_per_1m": 2.0,
        "free": False, "rate_limited": False,
        "per_req_limit_tokens": None, "context_length": 128_000,
    },
]
_MODELS_FREE = [
    {
        "id": "test/free",
        "input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0,
        "free": True, "rate_limited": True,
        "per_req_limit_tokens": None, "context_length": 64_000,
    },
]


# ---------------------------------------------------------------------------
# estimate_frames
# ---------------------------------------------------------------------------

def test_estimate_frames_with_duration():
    low, high = estimate_frames(60.0, 2.0, 50)
    assert 1 <= low <= high <= 50


def test_estimate_frames_capped_at_max():
    _, high = estimate_frames(1000.0, 1.0, 20)
    assert high <= 20


def test_estimate_frames_no_duration_returns_defaults():
    low, high = estimate_frames(None, 2.0, 60)
    assert low == 10
    assert high == 60


def test_estimate_frames_short_video():
    low, high = estimate_frames(4.0, 2.0, 60)
    assert low >= 1
    assert high >= 1


def test_estimate_frames_low_lte_high():
    for duration in [10.0, 60.0, 300.0, 1800.0]:
        low, high = estimate_frames(duration, 2.0, 60)
        assert low <= high


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_zero_pricing_returns_zero():
    low, high = estimate_cost("test/free", 5, 10, _MODELS_FREE)
    assert low == 0.0
    assert high == 0.0


def test_estimate_cost_with_pricing_positive():
    low, high = estimate_cost("test/model", 5, 10, _MODELS_PAID)
    assert low > 0
    assert high > low


def test_estimate_cost_unknown_model_returns_zero():
    low, high = estimate_cost("nobody/model", 5, 10, _MODELS_PAID)
    assert low == 0.0
    assert high == 0.0


def test_estimate_cost_more_frames_costs_more():
    low1, _ = estimate_cost("test/model", 5, 5, _MODELS_PAID)
    low2, _ = estimate_cost("test/model", 10, 10, _MODELS_PAID)
    assert low2 > low1


# ---------------------------------------------------------------------------
# estimate_time
# ---------------------------------------------------------------------------

def test_estimate_time_low_lt_high():
    low, high = estimate_time(5, 20)
    assert low < high


def test_estimate_time_both_positive():
    low, high = estimate_time(1, 1)
    assert low > 0
    assert low == high


def test_estimate_time_scales_with_frames():
    low1, _ = estimate_time(10, 10)
    low2, _ = estimate_time(20, 20)
    assert low2 > low1


# ---------------------------------------------------------------------------
# _fmt_time
# ---------------------------------------------------------------------------

def test_fmt_time_returns_seconds_below_90():
    assert _fmt_time(45.0) == "45s"
    assert _fmt_time(0.0) == "0s"
    assert _fmt_time(89.0) == "89s"


def test_fmt_time_returns_minutes_at_or_above_90():
    result = _fmt_time(90.0)
    assert "m" in result
    assert "s" not in result


def test_fmt_time_minutes_format():
    assert _fmt_time(120.0) == "2.0m"


# ---------------------------------------------------------------------------
# _fmt_cost
# ---------------------------------------------------------------------------

def test_fmt_cost_both_zero_returns_free():
    result = _fmt_cost(0.0, 0.0)
    assert "free" in result


def test_fmt_cost_tight_range_no_separator():
    result = _fmt_cost(0.1234, 0.1235)
    assert "–" not in result


def test_fmt_cost_wide_range_has_separator():
    result = _fmt_cost(0.01, 0.10)
    assert "–" in result


def test_fmt_cost_single_value_uses_high():
    result = _fmt_cost(0.0050, 0.0050)
    assert "0.0050" in result


# ---------------------------------------------------------------------------
# probe_video (integration — needs ffprobe)
# ---------------------------------------------------------------------------

def test_probe_video_returns_duration(test_video):
    info = probe_video(test_video)
    assert info["duration_s"] is not None
    assert info["duration_s"] == pytest.approx(5.0, abs=0.5)


def test_probe_video_returns_resolution(test_video):
    info = probe_video(test_video)
    assert info["width"] == 320
    assert info["height"] == 240


def test_probe_video_returns_fps(test_video):
    info = probe_video(test_video)
    assert info["fps"] is not None
    assert info["fps"] > 0


def test_probe_video_returns_size(test_video):
    info = probe_video(test_video)
    assert info["size_mb"] is not None
    assert info["size_mb"] > 0


def test_probe_video_missing_file_returns_none_fields():
    from pathlib import Path
    info = probe_video(Path("/nonexistent/does_not_exist.mp4"))
    assert info["duration_s"] is None
    assert info["width"] is None
    assert info["height"] is None
