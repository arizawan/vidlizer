"""Tests for vidlizer.batch — parse_json and call_model logic."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.conftest import MOCK_FLOW, make_mock_post
from vidlizer.batch import call_model, parse_json


def _fresh_mock_post(flow_data: dict = MOCK_FLOW) -> MagicMock:
    """Like make_mock_post but creates a fresh iter_lines iterator per call.

    make_mock_post uses return_value=iter(lines) which is exhausted after the
    first chunk. side_effect=lambda: iter(lines) creates a new iterator each time
    iter_lines() is invoked, which is required for multi-chunk tests.
    """
    flow_json = json.dumps(flow_data)
    content_escaped = json.dumps(flow_json)
    lines = [
        f'data: {{"choices":[{{"delta":{{"content":{content_escaped}}}}}],"usage":null}}'.encode(),
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
        b'data: [DONE]',
    ]
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_lines.side_effect = lambda: iter(lines)
    return MagicMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------

def test_parse_json_plain_object():
    raw = json.dumps({"flow": [{"step": 1}]})
    assert parse_json(raw) == {"flow": [{"step": 1}]}


def test_parse_json_passthrough_dict():
    d = {"flow": []}
    assert parse_json(d) is d


def test_parse_json_strips_think_tags():
    raw = "<think>internal reasoning goes here</think>\n{\"flow\": []}"
    assert parse_json(raw) == {"flow": []}


def test_parse_json_strips_pipe_think_tags():
    raw = "<|think|>reasoning<|/think|>\n{\"flow\": []}"
    assert parse_json(raw) == {"flow": []}


def test_parse_json_strips_code_fence_json():
    raw = "```json\n{\"flow\": []}\n```"
    assert parse_json(raw) == {"flow": []}


def test_parse_json_strips_plain_code_fence():
    raw = "```\n{\"flow\": []}\n```"
    assert parse_json(raw) == {"flow": []}


def test_parse_json_wraps_list_in_flow():
    raw = json.dumps([{"step": 1}, {"step": 2}])
    result = parse_json(raw)
    assert "flow" in result
    assert len(result["flow"]) == 2


def test_parse_json_empty_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json("")


def test_parse_json_only_think_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json("<think>only thinking, no json follows</think>")


def test_parse_json_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json("not json at all {broken{{")


def test_parse_json_whitespace_only_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json("   \n  ")


# ---------------------------------------------------------------------------
# call_model — single-chunk (no batching)
# ---------------------------------------------------------------------------

def test_call_model_returns_flow(monkeypatch):
    monkeypatch.setattr("vidlizer.http.requests.post", make_mock_post(MOCK_FLOW))
    result = call_model(
        api_key="k", model="m", frames=[], timeout=30,
        verbose=False, batch_size=0,
    )
    assert "flow" in result


def test_call_model_single_chunk_step_count(monkeypatch):
    monkeypatch.setattr("vidlizer.http.requests.post", make_mock_post(MOCK_FLOW))
    result = call_model(
        api_key="k", model="m", frames=[], timeout=30,
        verbose=False, batch_size=0,
    )
    assert len(result["flow"]) == len(MOCK_FLOW["flow"])


# ---------------------------------------------------------------------------
# call_model — batched serial path
# ---------------------------------------------------------------------------

def _make_frames(tmp_path, n: int) -> list:
    from pathlib import Path
    frames = []
    for i in range(n):
        f = tmp_path / f"frame_{i:04d}.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")
        frames.append(f)
    return frames


def test_call_model_batched_sequential_steps(monkeypatch, tmp_path):
    frames = _make_frames(tmp_path, 3)
    monkeypatch.setattr("vidlizer.http.requests.post", _fresh_mock_post(MOCK_FLOW))
    result = call_model(
        api_key="k", model="m", frames=frames, timeout=30,
        verbose=False, batch_size=1,
    )
    steps = result.get("flow", [])
    assert len(steps) == 3
    for i, step in enumerate(steps):
        assert step["step"] == i + 1


def test_call_model_json_repair_retry(monkeypatch, tmp_path):
    """Bad JSON on first chunk triggers repair; repaired result used."""
    frames = _make_frames(tmp_path, 2)
    call_count = 0
    good = {
        "flow": [{
            "step": 1, "phase": "Repaired", "scene": "", "subjects": [],
            "action": "", "text_visible": "", "context": "",
            "observations": "", "next_scene": None,
        }]
    }

    def _mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        content = "BROKEN{{not-json" if call_count == 1 else json.dumps(good)
        content_escaped = json.dumps(content)
        lines = [
            f'data: {{"choices":[{{"delta":{{"content":{content_escaped}}}}}],"usage":null}}'.encode(),
            b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":5}}',
            b'data: [DONE]',
        ]
        mock_resp.iter_lines.return_value = iter(lines)
        return mock_resp

    monkeypatch.setattr("vidlizer.http.requests.post", _mock_post)
    result = call_model(
        api_key="k", model="m", frames=frames, timeout=30,
        verbose=False, batch_size=1,
    )
    # call 1: chunk 1 original (bad), call 2: chunk 1 repair (good), call 3: chunk 2
    assert call_count == 3
    assert result["flow"][0]["phase"] == "Repaired"


def test_call_model_repair_fails_skips_chunk(monkeypatch, tmp_path):
    """If both original and repair return bad JSON, chunk is skipped."""
    frames = _make_frames(tmp_path, 2)

    def _mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        content_escaped = json.dumps("still broken json {{{")
        lines = [
            f'data: {{"choices":[{{"delta":{{"content":{content_escaped}}}}}],"usage":null}}'.encode(),
            b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":5}}',
            b'data: [DONE]',
        ]
        mock_resp.iter_lines.return_value = iter(lines)
        return mock_resp

    monkeypatch.setattr("vidlizer.http.requests.post", _mock_post)
    result = call_model(
        api_key="k", model="m", frames=frames, timeout=30,
        verbose=False, batch_size=1,
    )
    assert result["flow"] == []


# ---------------------------------------------------------------------------
# call_model — parallel path
# ---------------------------------------------------------------------------

def test_call_model_parallel_mode(monkeypatch, tmp_path):
    # 4 frames, batch_size=2 → 2 parallel chunks, each returning 1 step from MOCK_FLOW
    frames = _make_frames(tmp_path, 4)
    monkeypatch.setattr("vidlizer.http.requests.post", _fresh_mock_post(MOCK_FLOW))
    result = call_model(
        api_key="k", model="m", frames=frames, timeout=30,
        verbose=False, batch_size=2, concurrency=2,
    )
    assert "flow" in result
    assert len(result["flow"]) == 2


def test_call_model_parallel_steps_renumbered(monkeypatch, tmp_path):
    frames = _make_frames(tmp_path, 4)
    monkeypatch.setattr("vidlizer.http.requests.post", _fresh_mock_post(MOCK_FLOW))
    result = call_model(
        api_key="k", model="m", frames=frames, timeout=30,
        verbose=False, batch_size=2, concurrency=2,
    )
    steps = result["flow"]
    assert len(steps) == 2  # 2 chunks × 1 step each
    for i, step in enumerate(steps):
        assert step["step"] == i + 1
