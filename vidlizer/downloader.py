#!/usr/bin/env python3
"""URL video downloader via yt-dlp (YouTube, Loom, Vimeo, Twitter/X)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_SUPPORTED_DOMAINS = (
    "youtube.com", "youtu.be",
    "loom.com",
    "vimeo.com",
    "twitter.com", "x.com",
)


def is_url(s: str) -> bool:
    return bool(_URL_RE.match(s))


def is_supported(url: str) -> bool:
    return is_url(url) and any(d in url for d in _SUPPORTED_DOMAINS)


def _ydl():
    try:
        import yt_dlp
        return yt_dlp.YoutubeDL
    except ImportError:
        if shutil.which("yt-dlp"):
            return None  # signal: use CLI fallback
        raise RuntimeError(
            "yt-dlp not found — install with: pip install yt-dlp  or  brew install yt-dlp"
        )


def download(url: str, out_dir: Path) -> Path:
    """Download video to out_dir. Returns path to the downloaded file."""
    YDL = _ydl()
    tmpl = str(out_dir / "%(id)s.%(ext)s")
    opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": tmpl,
        "quiet": True,
        "no_playlist": True,
        "noplaylist": True,
        # android_vr is JS-less: no n-challenge, no PO token needed
        "extractor_args": {"youtube": {"player_client": ["android_vr"]}},
    }

    if YDL is not None:
        with YDL(opts) as ydl:
            ydl.download([url])
    else:
        import subprocess
        r = subprocess.run(
            ["yt-dlp", "--no-playlist", "-f", "bestvideo+bestaudio/best",
             "--merge-output-format", "mp4",
             "--extractor-args", "youtube:player_client=android_vr",
             "-o", tmpl, url],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"yt-dlp: {r.stderr[:500]}")

    files = [f for f in out_dir.iterdir() if f.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    if not files:
        raise RuntimeError("Download produced no recognizable video file")
    return max(files, key=lambda f: f.stat().st_size)


def get_metadata(url: str) -> dict:
    """Fetch title/duration/uploader without downloading."""
    try:
        YDL = _ydl()
        if YDL is None:
            return {"url": url}
        with YDL({"skip_download": True, "quiet": True, "no_playlist": True,
                   "extractor_args": {"youtube": {"player_client": ["android_vr"]}}}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
            return {
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "duration": info.get("duration"),
                "url": url,
            }
    except Exception:
        return {"url": url}
