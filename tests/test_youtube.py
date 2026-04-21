"""E2E tests requiring real network access — opt-in with -m e2e."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_YT_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" — 19s, public


# ---------------------------------------------------------------------------
# Download only
# ---------------------------------------------------------------------------

def test_youtube_download(tmp_path):
    from vidlizer.downloader import download

    video = download(_YT_URL, tmp_path)
    assert video.exists()
    assert video.suffix in (".mp4", ".mkv", ".webm", ".mov")
    assert video.stat().st_size > 10_000


def test_youtube_metadata(tmp_path):
    from vidlizer.downloader import get_metadata

    meta = get_metadata(_YT_URL)
    assert isinstance(meta.get("title"), str)
    assert len(meta["title"]) > 0


# ---------------------------------------------------------------------------
# Full CLI pipeline (requires OPENROUTER_API_KEY in env)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_full_pipeline_youtube_url(tmp_path):
    out = tmp_path / "yt-result.json"
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli",
         _YT_URL, "-o", str(out), "--no-transcript",
         "--max-frames", "5", "--fps", "0.5"],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "PROVIDER": "openrouter", "OPENROUTER_MODEL": "google/gemini-2.5-flash"},
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr[-3000:]}"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "flow" in data
    assert len(data["flow"]) >= 1
