"""Tests for vidlizer.http — CostTracker, ImageLimitError, post()."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vidlizer.http import CostCapExceeded, CostTracker, ImageLimitError, post


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

def test_cost_tracker_initial_state():
    t = CostTracker()
    assert t.prompt_tokens == 0
    assert t.completion_tokens == 0
    assert t.cost_usd == 0.0
    assert t.batches_done == 0
    assert t.max_cost == 0.0


def test_cost_tracker_negative_max_clamps_to_zero():
    t = CostTracker(max_cost=-5.0)
    assert t.max_cost == 0.0


def test_cost_tracker_add_accumulates_tokens():
    t = CostTracker()
    t.add("unknown/model", {"prompt_tokens": 100, "completion_tokens": 50})
    assert t.prompt_tokens == 100
    assert t.completion_tokens == 50
    assert t.batches_done == 1


def test_cost_tracker_add_twice_accumulates():
    t = CostTracker()
    t.add("unknown/model", {"prompt_tokens": 10, "completion_tokens": 5})
    t.add("unknown/model", {"prompt_tokens": 20, "completion_tokens": 10})
    assert t.prompt_tokens == 30
    assert t.completion_tokens == 15
    assert t.batches_done == 2


def test_cost_tracker_none_usage_values_treated_as_zero():
    t = CostTracker()
    t.add("unknown/model", {"prompt_tokens": None, "completion_tokens": None})
    assert t.prompt_tokens == 0
    assert t.completion_tokens == 0


def test_cost_tracker_cost_cap_exceeded(monkeypatch):
    monkeypatch.setattr("vidlizer.http._model_cost", lambda *_: 1.0)
    t = CostTracker(max_cost=0.001)
    with pytest.raises(CostCapExceeded, match="cost cap"):
        t.add("any/model", {"prompt_tokens": 1000, "completion_tokens": 100})


def test_cost_tracker_zero_cap_never_raises(monkeypatch):
    monkeypatch.setattr("vidlizer.http._model_cost", lambda *_: 999.0)
    t = CostTracker(max_cost=0.0)
    t.add("any/model", {"prompt_tokens": 1, "completion_tokens": 1})  # must not raise


def test_cost_tracker_summary_free():
    t = CostTracker()
    summary = t.summary()
    assert "free" in summary


def test_cost_tracker_summary_with_tokens():
    t = CostTracker()
    t.prompt_tokens = 1000
    t.completion_tokens = 500
    t.cost_usd = 0.0123
    summary = t.summary()
    assert "1,000" in summary
    assert "500" in summary
    assert "0.0123" in summary


# ---------------------------------------------------------------------------
# post() — OpenAI SSE path
# ---------------------------------------------------------------------------

def _make_sse_mock(content: str, status: int = 200) -> MagicMock:
    content_escaped = json.dumps(content)
    lines = [
        f'data: {{"choices":[{{"delta":{{"content":{content_escaped}}}}}],"usage":null}}'.encode(),
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
        b'data: [DONE]',
    ]
    mock_resp = MagicMock()
    mock_resp.ok = (status == 200)
    mock_resp.status_code = status
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.text = "error body"
    return mock_resp


def test_post_returns_content(monkeypatch):
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=_make_sse_mock('{"flow":[]}')))
    result = post("key", "m", {"model": "m", "messages": []}, 30, False)
    assert result["choices"][0]["message"]["content"] == '{"flow":[]}'


def test_post_tracks_tokens(monkeypatch):
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=_make_sse_mock("{}")))
    tracker = CostTracker()
    post("key", "m", {"model": "m", "messages": []}, 30, False, tracker=tracker)
    assert tracker.prompt_tokens == 10
    assert tracker.completion_tokens == 5


def test_post_429_raises_rate_limited(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 429
    mock_resp.text = "too many requests"
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(RuntimeError, match="rate_limited"):
        post("key", "m", {"model": "m", "messages": []}, 30, False)


def test_post_non_ok_raises_runtime(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.text = "internal server error"
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(RuntimeError, match="API 500"):
        post("key", "m", {"model": "m", "messages": []}, 30, False)


def test_post_http_image_limit_raises(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 400
    mock_resp.text = "image limit — most images exceeded"
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(ImageLimitError):
        post("key", "m", {"model": "m", "messages": []}, 30, False)


def test_post_stream_error_chunk_image_limit(monkeypatch):
    lines = [
        b'data: {"error": {"message": "image limit most requests exceeded"}}',
        b'data: [DONE]',
    ]
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(ImageLimitError):
        post("key", "m", {"model": "m", "messages": []}, 30, False)


def test_post_stream_error_chunk_generic_raises(monkeypatch):
    lines = [
        b'data: {"error": {"message": "some generic api error"}}',
        b'data: [DONE]',
    ]
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(RuntimeError, match="API error"):
        post("key", "m", {"model": "m", "messages": []}, 30, False)


def test_post_no_stream_opts_flag(monkeypatch):
    captured = {}

    def _capture(*args, **kwargs):
        captured["data"] = json.loads(kwargs.get("data", b"{}"))
        return _make_sse_mock("{}")

    monkeypatch.setattr("vidlizer.http.requests.post", _capture)
    post("key", "m", {"model": "m", "messages": []}, 30, False, no_stream_opts=True)
    assert "stream_options" not in captured.get("data", {})


# ---------------------------------------------------------------------------
# post() — Ollama path
# ---------------------------------------------------------------------------

def test_post_ollama_returns_content(monkeypatch):
    lines = [
        json.dumps({"message": {"content": '{"flow":'}, "done": False}).encode(),
        json.dumps({"message": {"content": "[]}"}, "done": False}).encode(),
        json.dumps({"done": True, "prompt_eval_count": 10, "eval_count": 5}).encode(),
    ]
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    result = post("key", "llava:7b", {"model": "llava:7b", "messages": []}, 30, False, is_ollama=True)
    assert '{"flow":' in result["choices"][0]["message"]["content"]


def test_post_ollama_non_ok_raises(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 503
    mock_resp.text = "service unavailable"
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(RuntimeError, match="Ollama 503"):
        post("key", "ollama-model", {"model": "ollama-model", "messages": []}, 30, False, is_ollama=True)


def test_post_ollama_error_in_stream_raises(monkeypatch):
    lines = [
        json.dumps({"error": "model not found"}).encode(),
    ]
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    monkeypatch.setattr("vidlizer.http.requests.post", MagicMock(return_value=mock_resp))
    with pytest.raises(RuntimeError, match="Ollama error"):
        post("key", "m", {"model": "m", "messages": []}, 30, False, is_ollama=True)
