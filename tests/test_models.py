"""Tests for vidlizer.models — pure functions (no network)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vidlizer.models import (
    _is_vision,
    _parse_per_req_limit,
    _parse_price,
    fetch_ollama_models,
    fetch_openai_compat_models,
    format_model_line,
    format_price_label,
    get_cheapest_paid,
    get_ollama_fallback_sequence,
    get_openai_fallback_sequence,
    get_pricing,
)

_MODELS = [
    {
        "id": "vendor/cheap",
        "input_usd_per_1m": 0.10, "output_usd_per_1m": 0.30,
        "free": False, "rate_limited": False,
        "per_req_limit_tokens": None, "context_length": 128_000,
    },
    {
        "id": "vendor/expensive",
        "input_usd_per_1m": 5.00, "output_usd_per_1m": 15.0,
        "free": False, "rate_limited": False,
        "per_req_limit_tokens": None, "context_length": 32_000,
    },
    {
        "id": "vendor/free-model:free",
        "input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0,
        "free": True, "rate_limited": True,
        "per_req_limit_tokens": 8192, "context_length": 64_000,
    },
]


# ---------------------------------------------------------------------------
# _parse_price
# ---------------------------------------------------------------------------

def test_parse_price_string_per_token():
    assert _parse_price("0.00000015") == pytest.approx(0.15)


def test_parse_price_float():
    assert _parse_price(0.000001) == pytest.approx(1.0)


def test_parse_price_none_returns_zero():
    assert _parse_price(None) == 0.0


def test_parse_price_zero_string():
    assert _parse_price("0") == 0.0


def test_parse_price_invalid_string():
    assert _parse_price("not-a-number") == 0.0


# ---------------------------------------------------------------------------
# _parse_per_req_limit
# ---------------------------------------------------------------------------

def test_per_req_limit_returns_min_of_values():
    assert _parse_per_req_limit({"prompt": "8192", "completion": "4096"}) == 4096


def test_per_req_limit_single_value():
    assert _parse_per_req_limit({"prompt": "16384"}) == 16384


def test_per_req_limit_none_returns_none():
    assert _parse_per_req_limit(None) is None


def test_per_req_limit_empty_dict_returns_none():
    assert _parse_per_req_limit({}) is None


# ---------------------------------------------------------------------------
# _is_vision
# ---------------------------------------------------------------------------

def test_is_vision_list_modality_with_image():
    m = {"architecture": {"input_modalities": ["text", "image"]}}
    assert _is_vision(m) is True


def test_is_vision_list_modality_text_only():
    m = {"architecture": {"input_modalities": ["text"]}}
    assert _is_vision(m) is False


def test_is_vision_string_modality_with_image():
    m = {"architecture": {"modality": "text+image"}}
    assert _is_vision(m) is True


def test_is_vision_string_modality_text_only():
    m = {"architecture": {"modality": "text"}}
    assert _is_vision(m) is False


def test_is_vision_missing_architecture():
    assert _is_vision({}) is False


# ---------------------------------------------------------------------------
# get_pricing
# ---------------------------------------------------------------------------

def test_get_pricing_exact_match():
    inp, out = get_pricing("vendor/cheap", _MODELS)
    assert inp == pytest.approx(0.10)
    assert out == pytest.approx(0.30)


def test_get_pricing_prefix_match():
    inp, out = get_pricing("vendor/cheap:variant", _MODELS)
    assert inp == pytest.approx(0.10)


def test_get_pricing_unknown_returns_zero():
    inp, out = get_pricing("nobody/model", _MODELS)
    assert inp == 0.0
    assert out == 0.0


def test_get_pricing_none_models_uses_fallback():
    inp, out = get_pricing("google/gemini-2.5-flash", None)
    assert inp > 0


# ---------------------------------------------------------------------------
# get_cheapest_paid
# ---------------------------------------------------------------------------

def test_get_cheapest_paid_returns_cheapest():
    assert get_cheapest_paid(_MODELS) == "vendor/cheap"


def test_get_cheapest_paid_ignores_free_models():
    free_only = [{"id": "f:free", "input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0, "free": True}]
    result = get_cheapest_paid(free_only)
    assert result == "google/gemini-2.5-flash"


def test_get_cheapest_paid_none_uses_fallback():
    result = get_cheapest_paid(None)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# format_model_line
# ---------------------------------------------------------------------------

def test_format_model_line_free_model():
    line = format_model_line(_MODELS[2])
    assert "free" in line
    assert "⚡" in line
    assert "vendor/free-model:free" in line


def test_format_model_line_paid_model():
    line = format_model_line(_MODELS[0])
    assert "$0.100/$0.300" in line
    assert "128K ctx" in line


def test_format_model_line_million_ctx():
    m = {**_MODELS[0], "context_length": 1_048_576}
    assert "1M ctx" in format_model_line(m)


def test_format_model_line_rate_limited_with_token_limit():
    m = {**_MODELS[2], "per_req_limit_tokens": 8192}
    assert "8K/req" in format_model_line(m)


def test_format_model_line_no_ctx():
    m = {**_MODELS[0], "context_length": 0}
    line = format_model_line(m)
    assert "ctx" not in line


# ---------------------------------------------------------------------------
# format_price_label
# ---------------------------------------------------------------------------

def test_format_price_label_free():
    m = {"free": True, "input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0}
    assert format_price_label(m) == "free"


def test_format_price_label_cheap_uses_per_M():
    m = {"free": False, "input_usd_per_1m": 0.05, "output_usd_per_1m": 0.20}
    label = format_price_label(m)
    assert "0.050" in label


def test_format_price_label_expensive_uses_dollar():
    m = {"free": False, "input_usd_per_1m": 2.50, "output_usd_per_1m": 10.0}
    label = format_price_label(m)
    assert "$2.50" in label


# ---------------------------------------------------------------------------
# get_ollama_fallback_sequence
# ---------------------------------------------------------------------------

def test_ollama_fallback_preferred_order():
    installed = ["qwen2.5vl:7b", "qwen2.5vl:3b", "minicpm-v:8b"]
    seq = get_ollama_fallback_sequence(installed, exclude="qwen2.5vl:7b")
    assert "qwen2.5vl:7b" not in seq
    assert seq[0] == "qwen2.5vl:3b"


def test_ollama_fallback_excludes_current_model():
    installed = ["qwen2.5vl:7b"]
    seq = get_ollama_fallback_sequence(installed, exclude="qwen2.5vl:7b")
    assert seq == []


def test_ollama_fallback_no_match_returns_empty():
    seq = get_ollama_fallback_sequence(["totally-unknown:latest"])
    assert seq == []


def test_ollama_fallback_empty_installed():
    assert get_ollama_fallback_sequence([]) == []


def test_ollama_fallback_no_duplicates():
    installed = ["qwen2.5vl:7b", "qwen2.5vl:7b"]
    seq = get_ollama_fallback_sequence(installed)
    assert seq.count("qwen2.5vl:7b") == 1


# ---------------------------------------------------------------------------
# get_openai_fallback_sequence
# ---------------------------------------------------------------------------

def test_openai_fallback_fragment_order():
    available = ["provider/llava-7b", "provider/qwen2.5-vl-7b-instruct", "other-model"]
    seq = get_openai_fallback_sequence(available)
    qwen_idx = seq.index("provider/qwen2.5-vl-7b-instruct")
    llava_idx = seq.index("provider/llava-7b")
    assert qwen_idx < llava_idx


def test_openai_fallback_excludes_current():
    available = ["model-a", "model-b"]
    seq = get_openai_fallback_sequence(available, exclude="model-a")
    assert "model-a" not in seq


def test_openai_fallback_unmatched_appended():
    seq = get_openai_fallback_sequence(["totally-unknown"])
    assert "totally-unknown" in seq


def test_openai_fallback_no_duplicates():
    seq = get_openai_fallback_sequence(["qwen2.5-vl-7b-model", "qwen2.5-vl-7b-model"])
    assert seq.count("qwen2.5-vl-7b-model") == 1


def test_openai_fallback_empty_returns_empty():
    assert get_openai_fallback_sequence([]) == []


# ---------------------------------------------------------------------------
# fetch_ollama_models / fetch_openai_compat_models (mocked network)
# ---------------------------------------------------------------------------

def test_fetch_ollama_models_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "qwen2.5vl:7b"}, {"name": "llava:7b"}]}
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr("vidlizer.models.requests.get", MagicMock(return_value=mock_resp))
    result = fetch_ollama_models("http://localhost:11434")
    assert result == ["qwen2.5vl:7b", "llava:7b"]


def test_fetch_ollama_models_unreachable(monkeypatch):
    import requests
    monkeypatch.setattr("vidlizer.models.requests.get", MagicMock(side_effect=requests.ConnectionError))
    assert fetch_ollama_models("http://localhost:11434") == []


def test_fetch_ollama_models_empty_response(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": []}
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr("vidlizer.models.requests.get", MagicMock(return_value=mock_resp))
    assert fetch_ollama_models() == []


def test_fetch_openai_compat_models_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "qwen2.5-vl-7b"}, {"id": "gemma-4"}]}
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr("vidlizer.models.requests.get", MagicMock(return_value=mock_resp))
    result = fetch_openai_compat_models("http://localhost:1234/v1", "lm-studio")
    assert "qwen2.5-vl-7b" in result
    assert "gemma-4" in result


def test_fetch_openai_compat_models_failure(monkeypatch):
    import requests
    monkeypatch.setattr("vidlizer.models.requests.get", MagicMock(side_effect=requests.ConnectionError))
    assert fetch_openai_compat_models("http://localhost:1234/v1", "key") == []
