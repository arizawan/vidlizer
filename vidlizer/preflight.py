#!/usr/bin/env python3
"""Pre-run video analysis: estimate frame count, cost, and processing time."""
from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vidlizer.models import get_pricing

_console = Console(stderr=True, highlight=False)

# Rough token estimates per frame at 512px width (empirical)
_TOKENS_PER_FRAME_PROMPT = 1200   # image tokens + system prompt amortised
_TOKENS_PER_FRAME_COMPLETION = 80  # JSON output per frame

# API latency estimate (seconds per batch of N frames)
_LATENCY_BASE_S = 3.0   # cold-start / network
_LATENCY_PER_FRAME_S = 0.8


def probe_video(video: Path) -> dict:
    """Return dict with duration_s, width, height, fps, size_mb."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of", "json",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info: dict = {"duration_s": None, "width": None, "height": None, "fps": None, "size_mb": None}
    try:
        import json
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [{}])
        s = streams[0] if streams else {}

        info["duration_s"] = float(fmt.get("duration") or 0) or None
        info["size_mb"] = int(fmt.get("size") or 0) / 1_048_576 or None
        info["width"] = s.get("width")
        info["height"] = s.get("height")

        rfr = s.get("r_frame_rate", "")
        if "/" in str(rfr):
            n, d = rfr.split("/")
            if float(d):
                info["fps"] = round(float(n) / float(d), 2)
    except Exception:
        pass
    return info


def estimate_frames(duration_s: float | None, min_interval: float, max_frames: int) -> tuple[int, int]:
    """Return (low_estimate, high_estimate) for frame count."""
    if not duration_s:
        return 10, max_frames
    interval_based = int(duration_s / min_interval) + 1
    low = min(max(1, interval_based // 2), max_frames)
    high = min(interval_based, max_frames)
    return low, high


def estimate_cost(
    model_id: str,
    low_frames: int,
    high_frames: int,
    models: list[dict] | None = None,
) -> tuple[float, float]:
    """Return (low_usd, high_usd) cost estimate."""
    inp_rate, out_rate = get_pricing(model_id, models)
    if inp_rate == 0 and out_rate == 0:
        return 0.0, 0.0

    def _cost(n: int) -> float:
        prompt_tok = n * _TOKENS_PER_FRAME_PROMPT
        completion_tok = n * _TOKENS_PER_FRAME_COMPLETION
        return (prompt_tok * inp_rate + completion_tok * out_rate) / 1_000_000

    return _cost(low_frames), _cost(high_frames)


def estimate_time(low_frames: int, high_frames: int) -> tuple[float, float]:
    """Return (low_s, high_s) wall-clock estimate."""
    def _t(n: int) -> float:
        return _LATENCY_BASE_S + n * _LATENCY_PER_FRAME_S
    return _t(low_frames), _t(high_frames)


def _fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    return f"{seconds / 60:.1f}m"


def _fmt_cost(low: float, high: float) -> str:
    if low == 0 and high == 0:
        return "[cyan]free[/cyan]"
    if abs(high - low) < 0.0005:
        return f"[green]~${high:.4f}[/green]"
    return f"[green]~${low:.4f} – ${high:.4f}[/green]"


def show_preflight(
    video: Path,
    model_id: str,
    min_interval: float,
    max_frames: int,
    models: list[dict] | None = None,
) -> dict:
    """Print pre-run summary panel. Returns probe info dict."""
    info = probe_video(video)
    low_f, high_f = estimate_frames(info["duration_s"], min_interval, max_frames)
    low_cost, high_cost = estimate_cost(model_id, low_f, high_f, models)
    low_t, high_t = estimate_time(low_f, high_f)

    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=14)
    t.add_column()

    # Video info
    if info["duration_s"]:
        dur_m, dur_s = divmod(int(info["duration_s"]), 60)
        dur_str = f"{dur_m}m {dur_s}s" if dur_m else f"{dur_s}s"
        t.add_row("duration", f"[white]{dur_str}[/white]")
    if info["width"] and info["height"]:
        t.add_row("resolution", f"[white]{info['width']}×{info['height']}[/white]")
    if info["fps"]:
        t.add_row("source fps", f"[white]{info['fps']}[/white]")
    if info["size_mb"]:
        t.add_row("file size", f"[white]{info['size_mb']:.1f} MB[/white]")

    t.add_row("", "")  # spacer

    # Estimates
    frame_range = f"[white]{low_f}[/white]" if low_f == high_f else f"[white]{low_f}–{high_f}[/white]"
    t.add_row("est. frames", frame_range)
    t.add_row("est. cost", _fmt_cost(low_cost, high_cost))

    time_low_str = _fmt_time(low_t)
    time_high_str = _fmt_time(high_t)
    time_range = f"[white]{time_low_str}[/white]" if time_low_str == time_high_str else f"[white]{time_low_str} – {time_high_str}[/white]"
    t.add_row("est. time", time_range)

    title = Text()
    title.append("pre-run estimate", style="dim")

    _console.print(Panel(t, title=title, border_style="blue", padding=(0, 1)))
    _console.print()

    return info
