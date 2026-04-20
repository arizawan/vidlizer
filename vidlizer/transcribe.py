#!/usr/bin/env python3
"""Optional audio transcription via faster-whisper.

Install: pip install vidlizer[transcribe]  or  pip install faster-whisper
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


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
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(video: Path, model_size: str = "base") -> list[dict] | None:
    """
    Return [{start, end, text}, ...] segments or None if faster-whisper unavailable.
    First call downloads the model (~150 MB for 'base').
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    with tempfile.TemporaryDirectory(prefix="vidlizer_audio_") as tmp:
        wav = Path(tmp) / "audio.wav"
        if not _extract_audio(video, wav):
            return None

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(wav), beam_size=1, vad_filter=True)
        return [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]
