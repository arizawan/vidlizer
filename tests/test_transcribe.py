"""Tests for vidlizer.transcribe — audio detection and transcript merging."""
from __future__ import annotations

import pytest

from vidlizer.batch import merge_transcript as _merge_transcript
from vidlizer.transcribe import has_audio


# ---------------------------------------------------------------------------
# has_audio — integration (needs real ffprobe + test fixtures)
# ---------------------------------------------------------------------------

def test_has_audio_true(test_video):
    assert has_audio(test_video) is True


def test_has_audio_false(test_video_silent):
    assert has_audio(test_video_silent) is False


# ---------------------------------------------------------------------------
# _merge_transcript — unit tests (no external deps)
# ---------------------------------------------------------------------------

def test_merge_basic_assignment():
    flow = [
        {"step": 1, "timestamp_s": 0.0},
        {"step": 2, "timestamp_s": 5.0},
        {"step": 3, "timestamp_s": 10.0},
    ]
    segs = [
        {"start": 1.0, "end": 3.0, "text": "Hello"},
        {"start": 6.0, "end": 8.0, "text": "World"},
    ]
    _merge_transcript(flow, segs)
    assert flow[0].get("speech") == "Hello"
    assert flow[1].get("speech") == "World"
    assert "speech" not in flow[2]


def test_merge_no_duplicates():
    """A segment spanning two step windows goes into exactly one step."""
    flow = [
        {"step": 1, "timestamp_s": 0.0},
        {"step": 2, "timestamp_s": 2.0},
    ]
    segs = [{"start": 1.0, "end": 3.5, "text": "Long segment"}]
    _merge_transcript(flow, segs)

    speeches = [s.get("speech") for s in flow]
    assert speeches.count("Long segment") == 1


def test_merge_segment_before_first_frame():
    flow = [{"step": 1, "timestamp_s": 5.0}]
    segs = [{"start": 0.5, "end": 3.0, "text": "Intro speech"}]
    _merge_transcript(flow, segs)
    assert flow[0]["speech"] == "Intro speech"


def test_merge_empty_segments_no_speech_key():
    flow = [{"step": 1, "timestamp_s": 0.0}]
    _merge_transcript(flow, [])
    assert "speech" not in flow[0]


def test_merge_multiple_segments_in_one_step():
    flow = [
        {"step": 1, "timestamp_s": 0.0},
        {"step": 2, "timestamp_s": 10.0},
    ]
    segs = [
        {"start": 1.0, "end": 3.0, "text": "First"},
        {"start": 4.0, "end": 6.0, "text": "Second"},
    ]
    _merge_transcript(flow, segs)
    assert flow[0]["speech"] == "First Second"
    assert "speech" not in flow[1]


def test_merge_empty_flow():
    _merge_transcript([], [{"start": 0.0, "end": 1.0, "text": "ignored"}])


def test_merge_null_timestamp_treated_as_zero():
    flow = [
        {"step": 1, "timestamp_s": None},
        {"step": 2, "timestamp_s": 5.0},
    ]
    segs = [{"start": 1.0, "end": 3.0, "text": "Hi"}]
    _merge_transcript(flow, segs)
    assert flow[0].get("speech") == "Hi"
