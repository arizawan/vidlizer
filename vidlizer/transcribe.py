#!/usr/bin/env python3
"""Audio transcription via Apple MLX Whisper (Neural Engine / GPU on M-series).

Install: pip install mlx-whisper  (auto-installed on first audio video)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

_MLX_REPO = "mlx-community/whisper-base-mlx"


def has_audio(video: Path) -> bool:
    """Return True if the video file contains an audio stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def _extract_audio(video: Path, out: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-y", "-i", str(video), "-vn", "-ar", "16000", "-ac", "1", str(out)],
        capture_output=True,
    )
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def is_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(video: Path, repo: str = _MLX_REPO) -> list[dict] | None:
    """
    Return [{start, end, text}, ...] or None if mlx-whisper unavailable.
    Uses Apple Neural Engine / GPU via MLX — fast on M-series Macs.
    Model (~150 MB) is downloaded once and cached in ~/.cache/huggingface/.
    """
    try:
        import mlx_whisper
    except ImportError:
        return None

    with tempfile.TemporaryDirectory(prefix="vidlizer_audio_") as tmp:
        wav = Path(tmp) / "audio.wav"
        if not _extract_audio(video, wav):
            return None

        result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, verbose=False)
        return [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result.get("segments", [])
            if s["text"].strip()
        ]
