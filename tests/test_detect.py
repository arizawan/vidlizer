"""Tests for vidlizer.detect — pure detection helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vidlizer.detect import (
    OL_PREFS,
    check_ffmpeg,
    check_ollama,
    check_openrouter,
    is_vision_model,
    pick_best_vision,
)


# ---------------------------------------------------------------------------
# is_vision_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id,expected", [
    ("qwen2.5vl:3b",                            True),
    ("minicpm-v:8b",                             True),
    ("llava:7b",                                 True),
    ("llava-onevision:7b",                       True),
    ("mlx-community/Qwen2.5-VL-3B-Instruct",    True),
    ("google/gemini-2.5-flash",                  False),
    ("llama3:8b",                                False),
    ("mistral:7b",                               False),
    ("openai/gpt-4o",                            False),
])
def test_is_vision_model(model_id, expected):
    assert is_vision_model(model_id) is expected


# ---------------------------------------------------------------------------
# pick_best_vision
# ---------------------------------------------------------------------------

def test_pick_best_vision_prefers_pref_over_hint():
    models = ["llava:7b", "qwen2.5vl:3b"]
    result = pick_best_vision(models, OL_PREFS)
    assert result == "qwen2.5vl:3b"


def test_pick_best_vision_falls_back_to_hint():
    models = ["llava:7b", "llama3:8b"]
    result = pick_best_vision(models, OL_PREFS)
    assert result == "llava:7b"


def test_pick_best_vision_returns_none_when_no_vision():
    models = ["llama3:8b", "mistral:7b"]
    assert pick_best_vision(models, OL_PREFS) is None


def test_pick_best_vision_empty_list():
    assert pick_best_vision([], OL_PREFS) is None


def test_pick_best_vision_pref_prefix_match():
    models = ["qwen2.5vl:7b-instruct-q4"]
    result = pick_best_vision(models, OL_PREFS)
    assert result == "qwen2.5vl:7b-instruct-q4"


# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------

def test_check_ffmpeg_found():
    with patch("vidlizer.detect.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("vidlizer.detect.subprocess.run") as mock_run:
        mock_run.return_value.stdout = "ffmpeg version 6.1 Copyright...\n"
        ok, ver = check_ffmpeg()
    assert ok is True
    assert ver == "6.1"


def test_check_ffmpeg_not_found():
    with patch("vidlizer.detect.shutil.which", return_value=None):
        ok, ver = check_ffmpeg()
    assert ok is False
    assert ver == "not found"


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------

def test_check_ollama_reachable():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"models": [{"name": "qwen2.5vl:3b"}]}
    with patch("vidlizer.detect.requests.get", return_value=mock_resp):
        ok, host, models = check_ollama("http://localhost:11434")
    assert ok is True
    assert "qwen2.5vl:3b" in models


def test_check_ollama_not_reachable():
    with patch("vidlizer.detect.requests.get", side_effect=ConnectionError("refused")):
        ok, host, models = check_ollama("http://localhost:11434")
    assert ok is False
    assert models == []


def test_check_ollama_default_host_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://custom:9999")
    with patch("vidlizer.detect.requests.get", side_effect=ConnectionError):
        ok, host, _ = check_ollama()
    assert "9999" in host


# ---------------------------------------------------------------------------
# check_openrouter
# ---------------------------------------------------------------------------

def test_check_openrouter_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ok, detail, model = check_openrouter(key="")
    assert ok is False
    assert detail == "not set"


def test_check_openrouter_key_valid_candidate_found():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "data": [
            {"id": "google/gemma-3-27b-it:free"},
            {"id": "openai/gpt-4o"},
        ]
    }
    with patch("vidlizer.detect.requests.get", return_value=mock_resp):
        ok, detail, model = check_openrouter(key="sk-test-1234")
    assert ok is True
    assert model == "google/gemma-3-27b-it:free"
    assert "1234" in detail


def test_check_openrouter_network_error_returns_true_with_fallback():
    with patch("vidlizer.detect.requests.get", side_effect=ConnectionError):
        ok, detail, model = check_openrouter(key="sk-test-abcd")
    assert ok is True
    assert model == "google/gemma-3-27b-it:free"


def test_check_openrouter_no_free_candidates_falls_through_to_image_model():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "data": [
            {
                "id": "some/vision-model:free",
                "architecture": {"input_modalities": ["text", "image"]},
            }
        ]
    }
    with patch("vidlizer.detect.requests.get", return_value=mock_resp):
        ok, detail, model = check_openrouter(key="sk-test-zzzz")
    assert ok is True
    assert model == "some/vision-model:free"
