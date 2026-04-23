"""Tests for vidlizer.formatter output modes."""
from __future__ import annotations

import json


from vidlizer.formatter import format_output, to_json, to_markdown, to_summary


SAMPLE = {
    "flow": [
        {
            "step": 1,
            "timestamp_s": 0.0,
            "phase": "Introduction",
            "scene": "Title screen.",
            "subjects": ["Logo"],
            "action": "Logo fades in.",
            "text_visible": "vidlizer",
            "context": "Opening.",
            "observations": "Clean design.",
            "next_scene": "Demo screen.",
        },
        {
            "step": 2,
            "timestamp_s": 5.0,
            "phase": "Demo",
            "scene": "Terminal window.",
            "subjects": ["Terminal"],
            "action": "User types command.",
            "text_visible": "vidlizer demo.mp4",
            "context": "CLI demo.",
            "observations": "",
            "next_scene": None,
            "speech": "Let's run vidlizer.",
        },
    ],
    "transcript": [
        {"start": 5.0, "end": 7.0, "text": "Let's run vidlizer."},
    ],
}


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

def test_to_json_is_valid():
    out = to_json(SAMPLE)
    parsed = json.loads(out)
    assert "flow" in parsed
    assert len(parsed["flow"]) == 2


def test_to_json_pretty_indented():
    out = to_json(SAMPLE)
    assert "\n" in out  # indented


def test_format_output_json_default():
    out = format_output(SAMPLE, "json")
    assert json.loads(out)["flow"][0]["step"] == 1


# ---------------------------------------------------------------------------
# to_summary
# ---------------------------------------------------------------------------

def test_to_summary_contains_phases():
    out = to_summary(SAMPLE)
    assert "Introduction" in out
    assert "Demo" in out


def test_to_summary_contains_actions():
    out = to_summary(SAMPLE)
    assert "Logo fades in" in out
    assert "User types command" in out


def test_to_summary_includes_transcript():
    out = to_summary(SAMPLE)
    assert "Let's run vidlizer" in out


def test_to_summary_empty_flow():
    out = to_summary({"flow": []})
    assert out == "No steps found."


def test_format_output_summary():
    out = format_output(SAMPLE, "summary")
    assert "Introduction" in out
    assert "{" not in out  # not raw JSON


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

def test_to_markdown_has_h1():
    out = to_markdown(SAMPLE)
    assert out.startswith("# Video Analysis")


def test_to_markdown_has_step_headers():
    out = to_markdown(SAMPLE)
    assert "## Step 1" in out
    assert "## Step 2" in out


def test_to_markdown_includes_timestamps():
    out = to_markdown(SAMPLE)
    assert "0.0s" in out
    assert "5.0s" in out


def test_to_markdown_includes_phases():
    out = to_markdown(SAMPLE)
    assert "Introduction" in out
    assert "Demo" in out


def test_to_markdown_includes_speech():
    out = to_markdown(SAMPLE)
    assert "Let's run vidlizer" in out


def test_to_markdown_includes_transcript_section():
    out = to_markdown(SAMPLE)
    assert "Full Transcript" in out


def test_to_markdown_null_timestamp():
    data = {"flow": [{"step": 1, "timestamp_s": None, "phase": "X",
                      "scene": "s", "action": "a"}]}
    out = to_markdown(data)
    assert "## Step 1 — X" in out
    assert "None" not in out


def test_format_output_markdown():
    out = format_output(SAMPLE, "markdown")
    assert out.startswith("#")
