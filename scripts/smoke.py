#!/usr/bin/env python3
"""
vidlizer smoke test — runs actual analyses against a real provider,
reports which features work and which don't.

Usage:
    python scripts/smoke.py                          # auto-detect provider
    python scripts/smoke.py --provider openrouter
    python scripts/smoke.py --model qwen2.5vl:7b
    python scripts/smoke.py --provider openai --model qwen/qwen2.5-vl-7b-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    status: str = "skip"   # "pass" | "fail" | "skip"
    detail: str = ""
    duration_s: float = 0.0


_checks: list[Check] = []


def _record(name: str, status: str, detail: str = "", duration_s: float = 0.0) -> None:
    _checks.append(Check(name, status, detail, duration_s))
    icon = {"pass": "[green]✓[/green]", "fail": "[red]✗[/red]", "skip": "[dim]−[/dim]"}[status]
    dur = f"  [dim]{duration_s:.1f}s[/dim]" if duration_s else ""
    console.print(f"  {icon}  {name}  [dim]{detail}[/dim]{dur}")


# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "not found"
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        line = r.stdout.split("\n")[0]
        ver = line.split("version ")[1].split(" ")[0] if "version " in line else "?"
        return True, ver
    except Exception:
        return True, "?"


def _check_ollama() -> tuple[bool, str, list[str]]:
    try:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, host, models
    except Exception:
        return False, os.getenv("OLLAMA_HOST", "http://localhost:11434"), []


def _check_openrouter() -> tuple[bool, str]:
    key = os.getenv("OPENROUTER_API_KEY", "")
    return bool(key), (f"key set (...{key[-4:]})" if key else "OPENROUTER_API_KEY not set")


def _check_whisper() -> tuple[bool, str]:
    try:
        import mlx_whisper  # noqa: F401
        return True, "installed"
    except ImportError:
        return False, "not installed  (pip install 'vidlizer[transcribe]')"


# ---------------------------------------------------------------------------
# Test asset creation
# ---------------------------------------------------------------------------

def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


def _make_video(path: Path, duration: int = 5) -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", str(path),
    )


def _make_image(path: Path) -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=green:size=320x240",
        "-frames:v", "1", str(path),
    )


def _make_pdf(path: Path) -> None:
    import fitz  # type: ignore[import]
    doc = fitz.open()
    for i, color in enumerate([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(page.rect, color=color, fill=color)
        page.insert_text((100, 420), f"Page {i + 1}", fontsize=24, color=(1, 1, 1))
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run(
    video: Path,
    output: Path,
    *,
    model: str,
    provider: str,
    batch_size: int = 1,
    max_frames: int = 6,
    output_format: str = "json",
    no_transcript: bool = True,
    start: float | None = None,
    end: float | None = None,
) -> tuple[int, float]:
    from vidlizer.core import run
    t0 = time.time()
    rc = run(
        video=video, output=output, model=model, provider=provider,
        batch_size=batch_size, max_frames=max_frames,
        output_format=output_format, no_transcript=no_transcript,
        verbose=False, timeout=120,
        start=start, end=end,
    )
    return rc, time.time() - t0


def _step_count(output: Path) -> int:
    try:
        return len(json.loads(output.read_text()).get("flow", []))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

def test_video_json(video: Path, out: Path, model: str, provider: str) -> None:
    rc, t = _run(video, out / "video.json", model=model, provider=provider)
    n = _step_count(out / "video.json")
    if rc == 0 and n > 0:
        _record("Video → JSON", "pass", f"{n} steps", t)
    else:
        _record("Video → JSON", "fail", f"rc={rc}", t)


def test_video_markdown(video: Path, out: Path, model: str, provider: str) -> None:
    p = out / "video.md"
    rc, t = _run(video, p, model=model, provider=provider, output_format="markdown")
    if rc == 0 and p.exists() and p.stat().st_size > 0:
        _record("Video → Markdown", "pass", f"{p.stat().st_size // 1024} KB", t)
    else:
        _record("Video → Markdown", "fail", f"rc={rc}", t)


def test_video_summary(video: Path, out: Path, model: str, provider: str) -> None:
    p = out / "video-summary.txt"
    rc, t = _run(video, p, model=model, provider=provider, output_format="summary")
    if rc == 0 and p.exists() and p.stat().st_size > 0:
        _record("Video → Summary", "pass", f"{p.stat().st_size} bytes", t)
    else:
        _record("Video → Summary", "fail", f"rc={rc}", t)


def test_image(image: Path, out: Path, model: str, provider: str) -> None:
    p = out / "image.json"
    rc, t = _run(image, p, model=model, provider=provider)
    n = _step_count(p)
    if rc == 0 and n > 0:
        _record("Image → JSON", "pass", f"{n} step(s)", t)
    else:
        _record("Image → JSON", "fail", f"rc={rc}", t)


def test_pdf(pdf: Path, out: Path, model: str, provider: str) -> None:
    p = out / "pdf.json"
    rc, t = _run(pdf, p, model=model, provider=provider)
    n = _step_count(p)
    if rc == 0 and n > 0:
        _record("PDF → JSON", "pass", f"{n} steps", t)
    else:
        _record("PDF → JSON", "fail", f"rc={rc}", t)


def test_cache(video: Path, out: Path, model: str, provider: str) -> None:
    p = out / "cached.json"
    rc1, _ = _run(video, p, model=model, provider=provider)
    if rc1 != 0:
        _record("Cache hit", "fail", "first run failed")
        return
    _, t2 = _run(video, p, model=model, provider=provider)
    if t2 < 1.0:
        _record("Cache hit", "pass", f"second run {t2:.2f}s (instant)", t2)
    else:
        _record("Cache hit", "fail", f"second run {t2:.1f}s — expected <1s", t2)


def test_start_end(video: Path, out: Path, model: str, provider: str) -> None:
    p = out / "trimmed.json"
    rc, t = _run(video, p, model=model, provider=provider,
                 max_frames=4, start=1.0, end=4.0)
    n = _step_count(p)
    if rc == 0 and n > 0:
        _record("Start/end trim", "pass", f"1–4s slice → {n} steps", t)
    else:
        _record("Start/end trim", "fail", f"rc={rc}", t)


def test_transcription(video: Path, out: Path, model: str, provider: str) -> None:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        _record("Transcription", "skip", "mlx-whisper not installed")
        return

    p = out / "transcript.json"
    from vidlizer.core import run
    t0 = time.time()
    rc = run(
        video=video, output=p, model=model, provider=provider,
        batch_size=1, max_frames=6, verbose=False,
        timeout=180, no_transcript=False,
    )
    t = time.time() - t0
    if rc == 0 and p.exists():
        data = json.loads(p.read_text())
        has_speech = any("speech" in step for step in data.get("flow", []))
        if has_speech:
            _record("Transcription", "pass", "speech merged into flow steps", t)
        else:
            _record("Transcription", "fail", "audio present but no speech field merged", t)
    else:
        _record("Transcription", "fail", f"rc={rc}", t)


def test_batch(video: Path, out: Path, model: str, provider: str) -> None:
    p = out / "batched.json"
    rc, t = _run(video, p, model=model, provider=provider,
                 batch_size=2, max_frames=6)
    n = _step_count(p)
    if rc == 0 and n > 0:
        _record("Batched (batch_size=2)", "pass", f"{n} steps across chunks", t)
    else:
        _record("Batched (batch_size=2)", "fail", f"rc={rc}", t)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _print_table() -> None:
    t = Table(
        show_header=True, header_style="bold cyan",
        border_style="dim", show_lines=True, expand=False,
    )
    t.add_column("Feature", style="white", min_width=26)
    t.add_column("Result", justify="center", min_width=9)
    t.add_column("Time", justify="right", min_width=7)
    t.add_column("Detail", style="dim")

    for c in _checks:
        if c.status == "pass":
            badge = "[bold green]✓  PASS[/bold green]"
        elif c.status == "fail":
            badge = "[bold red]✗  FAIL[/bold red]"
        else:
            badge = "[dim]—  SKIP[/dim]"
        dur = f"{c.duration_s:.1f}s" if c.duration_s else "—"
        t.add_row(c.name, badge, dur, c.detail)

    console.print(t)


def _save_html(model: str, provider: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = ROOT / "reports" / f"smoke-{stamp}.html"
    path.parent.mkdir(exist_ok=True)

    rows = "".join(
        f"<tr>"
        f"<td>{c.name}</td>"
        f"<td class='{c.status}'>"
        f"{'✓ PASS' if c.status == 'pass' else '✗ FAIL' if c.status == 'fail' else '— SKIP'}"
        f"</td>"
        f"<td>{f'{c.duration_s:.1f}s' if c.duration_s else '—'}</td>"
        f"<td>{c.detail}</td>"
        f"</tr>\n"
        for c in _checks
    )

    passed = sum(1 for c in _checks if c.status == "pass")
    failed = sum(1 for c in _checks if c.status == "fail")
    skipped = sum(1 for c in _checks if c.status == "skip")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vidlizer Smoke Test</title>
<style>
  :root {{ --bg:#0d1117; --surface:#161b22; --border:#30363d; --text:#c9d1d9;
           --blue:#58a6ff; --green:#56d364; --red:#f85149; --muted:#8b949e; }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:48px 64px }}
  h1 {{ color:var(--blue); font-size:1.6rem; margin-bottom:4px }}
  .meta {{ color:var(--muted); font-size:0.85rem; margin-bottom:32px }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:24px }}
  th {{ background:var(--surface); color:var(--blue); padding:12px 16px; text-align:left;
        border-bottom:2px solid var(--border); font-size:0.8rem; text-transform:uppercase; letter-spacing:.05em }}
  td {{ padding:11px 16px; border-bottom:1px solid var(--border); font-size:0.9rem }}
  tr:hover td {{ background:var(--surface) }}
  td:nth-child(2) {{ font-weight:600; font-size:0.82rem }}
  .pass {{ color:var(--green) }} .fail {{ color:var(--red) }} .skip {{ color:var(--muted) }}
  .summary {{ font-size:1rem; margin-top:8px }}
  .summary .p {{ color:var(--green); font-weight:700 }}
  .summary .f {{ color:var(--red); font-weight:700 }}
  .summary .s {{ color:var(--muted) }}
</style>
</head>
<body>
<h1>vidlizer — Smoke Test Report</h1>
<p class="meta">
  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp;
  Provider: <strong>{provider}</strong> &nbsp;|&nbsp;
  Model: <strong>{model}</strong>
</p>
<table>
  <thead><tr><th>Feature</th><th>Result</th><th>Time</th><th>Detail</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<div class="summary">
  <span class="p">{passed} passed</span>&ensp;
  <span class="f">{failed} failed</span>&ensp;
  <span class="s">{skipped} skipped</span>
</div>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="vidlizer smoke test")
    parser.add_argument("--provider", default="", help="ollama | openai | openrouter")
    parser.add_argument("--model", default="", help="model ID (auto-detected if omitted)")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold cyan]vidlizer — Smoke Test[/bold cyan]\n"
        "[dim]Runs the real pipeline. Reports what actually works.[/dim]",
        border_style="cyan", padding=(0, 2),
    ))
    console.print()

    # ── Prerequisites ─────────────────────────────────────────────
    console.print("[bold]Prerequisites[/bold]")
    ffmpeg_ok, ffmpeg_ver = _check_ffmpeg()
    ollama_ok, ollama_host, ollama_models = _check_ollama()
    or_ok, or_detail = _check_openrouter()
    whisper_ok, whisper_detail = _check_whisper()

    _tick = lambda ok: "[green]✓[/green]" if ok else "[red]✗[/red]"  # noqa: E731
    console.print(f"  {_tick(ffmpeg_ok)}  ffmpeg       {ffmpeg_ver}")
    if ollama_ok:
        shown = ", ".join(ollama_models[:4]) + (" …" if len(ollama_models) > 4 else "")
        console.print(f"  {_tick(True)}  Ollama       {ollama_host}  [dim][{shown}][/dim]")
    else:
        console.print(f"  {_tick(False)}  Ollama       {ollama_host}  [dim](not reachable)[/dim]")
    console.print(f"  {_tick(or_ok)}  OpenRouter   {or_detail}")
    console.print(f"  {'[green]✓[/green]' if whisper_ok else '[dim]−[/dim]'}  mlx-whisper  {whisper_detail}")
    console.print()

    if not ffmpeg_ok:
        console.print("[red]ffmpeg required — install it first.[/red]")
        return 1

    # ── Provider / model selection ────────────────────────────────
    provider = args.provider
    model = args.model

    if not provider:
        if ollama_ok and ollama_models:
            provider = "ollama"
        elif or_ok:
            provider = "openrouter"
        else:
            console.print("[red]No provider available.[/red]\n"
                          "Start Ollama or set [bold]OPENROUTER_API_KEY[/bold].")
            return 1

    if not model:
        if provider == "ollama" and ollama_models:
            preferred = ["qwen2.5vl:7b", "qwen2.5vl:3b", "minicpm-v:8b", "llava-onevision:7b", "llava"]
            model = next(
                (m for pref in preferred for m in ollama_models if m.startswith(pref.split(":")[0])),
                ollama_models[0],
            )
        elif provider == "openrouter":
            model = "google/gemini-2.5-flash-lite"
        elif provider == "openai":
            model = "qwen/qwen2.5-vl-7b-instruct"
        else:
            console.print("[red]Cannot select model — use --model.[/red]")
            return 1

    console.print(f"[bold]Running against:[/bold]  "
                  f"provider=[magenta]{provider}[/magenta]  "
                  f"model=[magenta]{model}[/magenta]")
    console.print()

    # ── Create test assets and run feature tests ──────────────────
    with tempfile.TemporaryDirectory(prefix="vidlizer_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        assets = tmp / "assets"
        out = tmp / "out"
        assets.mkdir()
        out.mkdir()

        console.print("[dim]Creating test assets (ffmpeg + pymupdf)…[/dim]")
        try:
            video = assets / "test.mp4"
            image = assets / "test.png"
            pdf = assets / "test.pdf"
            _make_video(video)
            _make_image(image)
            _make_pdf(pdf)
        except Exception as exc:
            console.print(f"[red]Asset creation failed: {exc}[/red]")
            return 1

        console.print("[dim]Running feature tests…[/dim]\n")

        test_video_json(video, out, model, provider)
        test_video_markdown(video, out, model, provider)
        test_video_summary(video, out, model, provider)
        test_image(image, out, model, provider)
        test_pdf(pdf, out, model, provider)
        test_cache(video, out, model, provider)
        test_start_end(video, out, model, provider)
        test_batch(video, out, model, provider)
        test_transcription(video, out, model, provider)

    # ── Summary table ─────────────────────────────────────────────
    console.print()
    _print_table()
    console.print()

    passed = sum(1 for c in _checks if c.status == "pass")
    failed = sum(1 for c in _checks if c.status == "fail")
    skipped = sum(1 for c in _checks if c.status == "skip")
    color = "green" if failed == 0 else "red"
    console.print(
        f"[{color}][bold]{passed} passed[/bold][/{color}]  "
        f"[red]{failed} failed[/red]  "
        f"[dim]{skipped} skipped[/dim]"
    )

    report = _save_html(model, provider)
    console.print(f"\n[dim]Report:[/dim] [cyan]{report}[/cyan]")
    console.print(f"[dim]Open: [/dim] [cyan]open {report}[/cyan]")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
