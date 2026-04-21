"""Integration tests for the full run() pipeline with mocked OpenRouter API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import MOCK_FLOW, make_mock_post
from vidlizer import cache as _cache


@pytest.fixture(autouse=True)
def _reset_cache():
    _cache.clear()
    yield
    _cache.clear()


def _run(video, output, **kwargs):
    """Helper: call run() with the necessary mocks applied."""
    from vidlizer.core import run

    defaults = dict(
        model="google/gemini-2.5-flash",
        provider="openrouter",
        scene=0.1, min_interval=1.0, fps=1.0,
        scale=320, max_frames=5, batch_size=0,
        timeout=30, verbose=False, max_cost=10.0,
        start=None, end=None, dedup_threshold=0,
    )
    defaults.update(kwargs)

    with (
        patch("vidlizer.core.requests.post", make_mock_post(MOCK_FLOW)),
        patch("vidlizer.models.fetch_models", return_value=[]),
        patch("vidlizer.preflight.show_preflight", return_value={}),
        patch("vidlizer.transcribe.has_audio", return_value=False),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"}),
    ):
        return run(video, output, **defaults)


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------

def test_run_video_exits_zero(test_video, tmp_path):
    out = tmp_path / "result.json"
    rc = _run(test_video, out)
    assert rc == 0


def test_run_video_writes_json(test_video, tmp_path):
    out = tmp_path / "result.json"
    _run(test_video, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "flow" in data
    assert isinstance(data["flow"], list)
    assert len(data["flow"]) >= 1


def test_run_video_flow_has_required_fields(test_video, tmp_path):
    out = tmp_path / "result.json"
    _run(test_video, out)
    step = json.loads(out.read_text())["flow"][0]
    for field in ("step", "phase", "scene", "action"):
        assert field in step, f"missing field: {field}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_second_run_is_cache_hit(test_video, tmp_path):
    out = tmp_path / "result.json"
    call_count = [0]
    orig_post = make_mock_post(MOCK_FLOW)

    def _counting_post(*a, **kw):
        call_count[0] += 1
        return orig_post(*a, **kw)

    with (
        patch("vidlizer.core.requests.post", _counting_post),
        patch("vidlizer.models.fetch_models", return_value=[]),
        patch("vidlizer.preflight.show_preflight", return_value={}),
        patch("vidlizer.transcribe.has_audio", return_value=False),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"}),
    ):
        from vidlizer.core import run
        params = dict(
            model="google/gemini-2.5-flash",
            provider="openrouter",
            scene=0.1, min_interval=1.0, fps=1.0,
            scale=320, max_frames=5, batch_size=0,
            timeout=30, verbose=False, max_cost=10.0,
            start=None, end=None, dedup_threshold=0,
        )
        run(test_video, out, **params)
        run(test_video, out, **params)

    assert call_count[0] == 1  # second run served from cache


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

def test_run_image_exits_zero(test_image_png, tmp_path):
    out = tmp_path / "result.json"
    rc = _run(test_image_png, out)
    assert rc == 0


def test_run_image_writes_json(test_image_png, tmp_path):
    out = tmp_path / "result.json"
    _run(test_image_png, out)
    data = json.loads(out.read_text())
    assert "flow" in data


# ---------------------------------------------------------------------------
# PDF input
# ---------------------------------------------------------------------------

def test_run_pdf_exits_zero(test_pdf, tmp_path):
    out = tmp_path / "result.json"
    rc = _run(test_pdf, out)
    assert rc == 0


def test_run_pdf_writes_json(test_pdf, tmp_path):
    out = tmp_path / "result.json"
    _run(test_pdf, out)
    data = json.loads(out.read_text())
    assert "flow" in data


# ---------------------------------------------------------------------------
# analyse_moment (start/end)
# ---------------------------------------------------------------------------

def test_run_with_start_end(test_video, tmp_path):
    out = tmp_path / "result.json"
    rc = _run(test_video, out, start=1.0, end=3.0)
    assert rc == 0
    assert out.exists()
