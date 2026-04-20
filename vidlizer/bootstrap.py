#!/usr/bin/env python3
"""Auto-install missing dependencies on macOS."""
from __future__ import annotations

import shutil
import subprocess
import sys


def _brew_install(pkg: str, console) -> bool:
    if not shutil.which("brew"):
        console.print(f"[red]✗[/red]  Homebrew not found — install from https://brew.sh then re-run")
        return False
    console.print(f"[cyan]→[/cyan] installing [bold]{pkg}[/bold] via Homebrew…")
    r = subprocess.run(["brew", "install", pkg], capture_output=True, text=True)
    if r.returncode != 0:
        console.print(f"[red]✗[/red]  brew install {pkg} failed:\n{r.stderr[:400]}")
        return False
    console.print(f"[green]✓[/green]  {pkg} installed")
    return True


def _pip_install(pkg: str, console) -> bool:
    console.print(f"[cyan]→[/cyan] installing [bold]{pkg}[/bold] via pip…")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", pkg],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        console.print(f"[red]✗[/red]  pip install {pkg} failed:\n{r.stderr[:400]}")
        return False
    console.print(f"[green]✓[/green]  {pkg} installed")
    return True


def ensure_ffmpeg(console) -> bool:
    """Check for ffmpeg; brew-install if missing."""
    if shutil.which("ffmpeg"):
        return True
    return _brew_install("ffmpeg", console)


def ensure_faster_whisper(console) -> bool:
    """Check for faster-whisper; pip-install if missing."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        ok = _pip_install("faster-whisper", console)
        if ok:
            # Reload so the freshly installed package is importable this session
            import importlib
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                pass
        return ok


def run_checks(console) -> None:
    """Run all startup dependency checks. Called once from _main()."""
    ensure_ffmpeg(console)
