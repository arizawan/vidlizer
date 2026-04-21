#!/usr/bin/env python3
"""Interactive CLI entry point for vidlizer."""
from __future__ import annotations

import contextlib
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

_console = Console(stderr=True, highlight=False)


_CURATED_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-pro",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-4-31b-it:free",
]

_MODEL_NOTES = {
    "google/gemini-2.5-flash":             "Recommended — fast, accurate",
    "google/gemini-2.5-flash-lite":        "Cheaper, slightly less accurate",
    "google/gemini-2.5-pro":               "Best quality, expensive",
    "openai/gpt-4o-mini":                  "OpenAI budget option",
    "openai/gpt-4o":                       "OpenAI flagship, expensive",
    "nvidia/nemotron-nano-12b-v2-vl:free": "Free ⚡ rate-limited, 10-image cap (auto-batched)",
    "google/gemma-4-31b-it:free":          "Free ⚡ rate-limited, may be slow",
}


def _prompt_model() -> str:
    """Curated OpenRouter model picker with live pricing."""
    from vidlizer.models import fetch_models, format_price_label

    with _console.status("[dim]fetching prices…[/dim]", spinner="dots2"):
        all_models = fetch_models(os.getenv("OPENROUTER_API_KEY"))

    price_by_id = {m["id"]: format_price_label(m) for m in all_models}

    choices = []
    for mid in _CURATED_MODELS:
        price = price_by_id.get(mid, "")
        note = _MODEL_NOTES.get(mid, "")
        desc = f"{note}  [{price}]" if price else note
        choices.append((mid, desc))
    choices.append(("custom", "Enter a model slug manually"))

    choice = _prompt_select("Select model", choices)
    if choice == "custom":
        choice = _prompt_str("Model slug (e.g. openai/gpt-4o)")
    return choice


def _prompt_provider() -> str:
    """Ask user to choose a provider."""
    choices = [
        ("ollama",      "Local — Ollama (no API key, no cost)"),
        ("openai",      "Local — OpenAI-compatible (LM Studio, vLLM, LocalAI, real OpenAI)"),
        ("openrouter",  "Cloud — OpenRouter (API key required, pay-per-use)"),
    ]
    return _prompt_select("Select provider", choices)


def _prompt_ollama_model() -> str:
    """Curated local model picker. Shows install status for each model."""
    from vidlizer.models import OLLAMA_CURATED_MODELS, fetch_ollama_models

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    with _console.status("[dim]checking Ollama…[/dim]", spinner="dots2"):
        installed = set(fetch_ollama_models(ollama_host))

    choices = []
    for m in OLLAMA_CURATED_MODELS:
        mid = m["id"]
        base = mid.split(":")[0]
        is_installed = any(i.split(":")[0] == base for i in installed)
        ram = m.get("ram_gb", "?")
        disk = m.get("size_gb", "?")
        if is_installed:
            status = "[green]✓ installed[/green]"
        elif m["recommended"]:
            status = f"~{disk} GB disk / ~{ram} GB RAM — [cyan]ollama pull {mid}[/cyan]  ★ recommended"
        else:
            status = f"~{disk} GB disk / ~{ram} GB RAM — [cyan]ollama pull {mid}[/cyan]"
        choices.append((mid, f"{m['desc']}  [{status}]"))
    choices.append(("custom", "Enter model name manually"))

    choice = _prompt_select("Select local model", choices)
    if choice == "custom":
        choice = _prompt_str("Model name (e.g. qwen2.5vl:3b)")
    return choice


def _print_banner() -> None:
    title = Text()
    title.append("vid", style="bold cyan")
    title.append("lizer", style="bold white")
    subtitle = Text("frame-by-frame video  →  JSON user journey", style="dim")
    _console.print(Panel.fit(
        f"{title}\n{subtitle}",
        border_style="cyan",
        padding=(0, 3),
    ))
    _console.print()


def _print_config(args: dict) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=12)
    t.add_column(style="bold white")
    t.add_row("video", str(args["video"]))
    if args.get("provider") == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        t.add_row("provider", f"[bold green]ollama[/bold green]  [dim]{ollama_host}[/dim]")
    t.add_row("model", f"[magenta]{args['model']}[/magenta]")
    t.add_row("output", f"[cyan]{args['output']}[/cyan]")
    frames_row = f"max {args['max_frames']}  ·  scene>{args['scene']}  ·  interval {args['min_interval']}s"
    if args.get("start") is not None or args.get("end") is not None:
        s = args.get("start", 0)
        e = args.get("end", "end")
        frames_row += f"  ·  [yellow]{s}s → {e}s[/yellow]"
    t.add_row("frames", frames_row)
    t.add_row("cost cap", f"[yellow]${args['max_cost']:.2f}[/yellow]  [dim](abort if exceeded)[/dim]")
    _console.print(Panel(t, title="[dim]config[/dim]", border_style="dim", padding=(0, 1)))
    _console.print()


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_str(label: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default else ""
    try:
        import questionary
        result = questionary.text(f"{label}{hint}:", default=default or "").ask()
        if result is None:
            raise KeyboardInterrupt
        return result.strip() or (default or "")
    except ImportError:
        val = input(f"{label}{hint}: ").strip()
        return val or (default or "")


def _prompt_select(label: str, choices: list[tuple[str, str]]) -> str:
    try:
        import questionary
        options = [questionary.Choice(title=f"{v}  — {desc}", value=v) for v, desc in choices]
        result = questionary.select(label, choices=options).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    except ImportError:
        print(f"\n{label}")
        for i, (v, desc) in enumerate(choices, 1):
            print(f"  {i}) {v}  — {desc}")
        raw = input("Choose [1]: ").strip()
        idx = int(raw) - 1 if raw.isdigit() else 0
        return choices[max(0, min(idx, len(choices) - 1))][0]


def _prompt_confirm(label: str, default: bool = False) -> bool:
    try:
        import questionary
        result = questionary.confirm(label, default=default).ask()
        if result is None:
            raise KeyboardInterrupt
        return result
    except ImportError:
        hint = "Y/n" if default else "y/N"
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if not raw:
            return default
        return raw.startswith("y")


def _prompt_float(label: str, default: float) -> float:
    raw = _prompt_str(label, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _prompt_int(label: str, default: int) -> int:
    raw = _prompt_str(label, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


_MEDIA_EXTS = (
    "*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.flv *.wmv *.ts *.mts *.mpg *.mpeg "
    "*.jpg *.jpeg *.png *.gif *.webp *.bmp *.tiff *.tif *.pdf"
)


def _pick_file_gui() -> Path | None:
    """Open a native OS file-picker dialog. Returns None if unavailable or cancelled."""
    # macOS: use AppleScript — always available, no Python GUI deps needed
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose file with prompt "Select video file:")'],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                p = r.stdout.strip()
                return Path(p) if p else None
        except Exception:
            pass

    # Other platforms: tkinter fallback
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video & image files", _MEDIA_EXTS), ("All files", "*.*")],
        )
        root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def interactive_args(video: Path | None, output_format: str = "json") -> dict:
    """Ask for any missing configuration interactively."""
    load_dotenv()
    interactive = _is_interactive()
    args: dict = {}

    # --- Video file ---
    if video is None:
        if not interactive:
            _console.print("[red]error:[/red] video path required as argument")
            sys.exit(2)
        _console.print("[dim]opening file picker…[/dim]")
        video = _pick_file_gui()
        if video is None:
            raw = _prompt_str("Video file path")
            video = Path(os.path.expanduser(raw))
    if not video.is_file():
        _console.print(f"[red]error:[/red] file not found: {video}")
        sys.exit(2)
    args["video"] = video

    # --- Output path ---
    safe_stem = re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-") or "output"
    import tempfile as _tf
    out_dir = Path.cwd() if str(video).startswith(_tf.gettempdir()) else video.parent
    _ext = ".analysis.md" if output_format == "markdown" else ".analysis.txt" if output_format == "summary" else ".analysis.json"
    default_output = out_dir / f"{safe_stem}{_ext}"
    if interactive:
        out_raw = _prompt_str("Output path", str(default_output))
        args["output"] = Path(os.path.expanduser(out_raw))
    else:
        args["output"] = default_output

    # --- Provider & model ---
    env_provider = os.getenv("PROVIDER", "").lower()
    if env_provider:
        _provider = env_provider
    elif interactive:
        _provider = _prompt_provider()
    else:
        _provider = "ollama"
    args["provider"] = _provider

    if _provider == "ollama":
        env_model = os.getenv("OLLAMA_MODEL")
        if env_model:
            args["model"] = env_model
        elif interactive:
            args["model"] = _prompt_ollama_model()
        else:
            args["model"] = "qwen2.5vl:3b"
    elif _provider == "openai":
        env_model = os.getenv("OPENAI_MODEL")
        if env_model:
            args["model"] = env_model
        elif interactive:
            base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
            _console.print(f"  [dim]endpoint: {base}[/dim]")
            args["model"] = _prompt_str("Model name (as shown in LM Studio / your server)", "")
        else:
            args["model"] = os.getenv("OPENAI_MODEL", "")
    else:
        env_model = os.getenv("OPENROUTER_MODEL")
        if env_model:
            args["model"] = env_model
        elif interactive:
            args["model"] = _prompt_model()
        else:
            args["model"] = "google/gemini-2.5-flash"

    # --- Advanced settings ---
    if interactive:
        advanced = _prompt_confirm("Customize advanced settings?", default=False)
    else:
        advanced = False

    if advanced:
        args["scene"] = _prompt_float("Scene-change threshold (0..1, lower=more frames)", float(os.getenv("SCENE_THRESHOLD", "0.1")))
        args["min_interval"] = _prompt_float("Min seconds between frames (fallback interval)", float(os.getenv("MIN_INTERVAL", "2")))
        args["scale"] = _prompt_int("Frame width in pixels", int(os.getenv("FRAME_WIDTH", "512")))
        args["max_frames"] = _prompt_int("Max frames to send", int(os.getenv("MAX_FRAMES", "60")))
        args["batch_size"] = _prompt_int("Batch size (0=auto, use 8 for Nvidia free)", int(os.getenv("BATCH_SIZE", "0")))
    else:
        args["scene"] = float(os.getenv("SCENE_THRESHOLD", "0.1"))
        args["min_interval"] = float(os.getenv("MIN_INTERVAL", "2"))
        args["scale"] = int(os.getenv("FRAME_WIDTH", "512"))
        args["max_frames"] = int(os.getenv("MAX_FRAMES", "60"))
        args["batch_size"] = int(os.getenv("BATCH_SIZE", "0"))

    args["timeout"] = int(os.getenv("REQUEST_TIMEOUT", "600"))
    args["max_cost"] = float(os.getenv("MAX_COST_USD", "1.00"))
    args["verbose"] = False
    args["fps"] = None
    args["start"] = None
    args["end"] = None
    args["dedup_threshold"] = 8
    args["no_transcript"] = False
    args["output_format"] = "json"

    return args


def main() -> int:
    try:
        return _main()
    except KeyboardInterrupt:
        _console.print("\n[dim]cancelled[/dim]")
        return 130


def _main() -> int:
    load_dotenv()

    import argparse
    p = argparse.ArgumentParser(
        description="vidlizer — analyze video/image/PDF → JSON user-journey map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any omitted options will be asked interactively when running in a terminal.",
    )
    p.add_argument("video", nargs="?", type=str, help="Path to file or URL (YouTube, Loom, Vimeo, Twitter)")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--scene", type=float, default=None)
    p.add_argument("--min-interval", type=float, default=None, dest="min_interval")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--max-cost", type=float, default=None, dest="max_cost",
                   help="Abort mid-run if spend exceeds this many USD (default 1.00)")
    p.add_argument("--start", type=float, default=None,
                   help="Analyze from this timestamp in seconds (analyze_moment)")
    p.add_argument("--end", type=float, default=None,
                   help="Analyze up to this timestamp in seconds (analyze_moment)")
    p.add_argument("--no-transcript", action="store_true", dest="no_transcript",
                   help="Skip audio transcription even if faster-whisper is installed")
    p.add_argument("--dedup-threshold", type=int, default=None, dest="dedup_threshold",
                   help="Perceptual dedup Hamming threshold (default 8, 0=off)")
    p.add_argument("--format", choices=["json", "summary", "markdown"], default="json",
                   dest="output_format", metavar="FORMAT",
                   help="Output format: json (default), summary (plain text), markdown")
    p.add_argument("--provider", choices=["openrouter", "ollama", "openai"], default=None,
                   help="AI provider: openrouter (cloud), ollama (local), openai (LM Studio/vLLM/any OpenAI-compat)")
    cli = p.parse_args()

    if cli.provider:
        os.environ["PROVIDER"] = cli.provider

    _print_banner()

    from vidlizer.bootstrap import run_checks
    run_checks(_console)

    # Handle URL input — download to a managed temp dir before interactive_args
    video_raw = cli.video
    url_ctx: contextlib.AbstractContextManager = contextlib.nullcontext(None)
    video_path: Path | None = Path(video_raw) if video_raw else None

    if video_raw:
        from vidlizer.downloader import is_url, download, get_metadata
        if is_url(video_raw):
            url_tmp = tempfile.TemporaryDirectory(prefix="vidlizer_dl_")
            url_ctx = url_tmp
            _console.print(f"[cyan]→[/cyan] [bold]{video_raw}[/bold]")
            with _console.status("[dim]fetching metadata…[/dim]", spinner="dots2"):
                meta = get_metadata(video_raw)
            if meta.get("title"):
                _console.print(f"   [dim]{meta['title']}[/dim]")
            with _console.status("[dim]downloading…[/dim]", spinner="dots2"):
                try:
                    video_path = download(video_raw, Path(url_tmp.name))
                except RuntimeError as e:
                    _console.print(f"[red]✗[/red]  [red]{e}[/red]")
                    return 2
            _console.print()

    with url_ctx:
        # Build args: interactive fills missing values
        iargs = interactive_args(video_path, output_format=cli.output_format)

        # CLI flags override interactive answers
        if cli.output:                       iargs["output"] = cli.output
        if cli.model:                        iargs["model"] = cli.model
        if cli.scene is not None:            iargs["scene"] = cli.scene
        if cli.min_interval is not None:     iargs["min_interval"] = cli.min_interval
        if cli.fps is not None:              iargs["fps"] = cli.fps
        if cli.scale is not None:            iargs["scale"] = cli.scale
        if cli.max_frames is not None:       iargs["max_frames"] = cli.max_frames
        if cli.batch_size is not None:       iargs["batch_size"] = cli.batch_size
        if cli.timeout is not None:          iargs["timeout"] = cli.timeout
        if cli.max_cost is not None:         iargs["max_cost"] = cli.max_cost
        if cli.start is not None:            iargs["start"] = cli.start
        if cli.end is not None:              iargs["end"] = cli.end
        if cli.dedup_threshold is not None:  iargs["dedup_threshold"] = cli.dedup_threshold
        if cli.verbose:                      iargs["verbose"] = True
        if cli.no_transcript:                iargs["no_transcript"] = True
        iargs["output_format"] = cli.output_format

        _print_config(iargs)
        _console.print(Rule(style="dim"))
        _console.print()

        from vidlizer.core import run
        return run(**iargs)


if __name__ == "__main__":
    sys.exit(main())
