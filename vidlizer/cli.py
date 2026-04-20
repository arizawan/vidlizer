#!/usr/bin/env python3
"""Interactive CLI entry point for vidlizer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

_console = Console(stderr=True, highlight=False)


# Curated list: reliable paid vision models only (no free-tier rate limits)
_CURATED_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-pro",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
]

_MODEL_NOTES = {
    "google/gemini-2.5-flash":      "Recommended — fast, accurate",
    "google/gemini-2.5-flash-lite": "Cheaper, slightly less accurate",
    "google/gemini-2.5-pro":        "Best quality, expensive",
    "openai/gpt-4o-mini":           "OpenAI budget option",
    "openai/gpt-4o":                "OpenAI flagship, expensive",
}


def _prompt_model() -> str:
    """Simple curated model picker with live pricing."""
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
    t.add_row("model", f"[magenta]{args['model']}[/magenta]")
    t.add_row("output", f"[cyan]{args['output']}[/cyan]")
    t.add_row("frames", f"max {args['max_frames']}  ·  scene>{args['scene']}  ·  interval {args['min_interval']}s")
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


def interactive_args(video: Path | None) -> dict:
    """Ask for any missing configuration interactively."""
    load_dotenv()
    interactive = _is_interactive()
    args: dict = {}

    # --- Video file ---
    if video is None:
        if not interactive:
            _console.print("[red]error:[/red] video path required as argument")
            sys.exit(2)
        raw = _prompt_str("Video file path")
        video = Path(os.path.expanduser(raw))
    if not video.is_file():
        _console.print(f"[red]error:[/red] file not found: {video}")
        sys.exit(2)
    args["video"] = video

    # --- Output path ---
    default_output = video.with_suffix(video.suffix + ".analysis.json")
    if interactive:
        out_raw = _prompt_str("Output JSON path", str(default_output))
        args["output"] = Path(os.path.expanduser(out_raw))
    else:
        args["output"] = default_output

    # --- Model ---
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
        args["min_interval"] = _prompt_float("Min seconds between frames (fallback interval)", float(os.getenv("MIN_INTERVAL", "5")))
        args["scale"] = _prompt_int("Frame width in pixels", int(os.getenv("FRAME_WIDTH", "512")))
        args["max_frames"] = _prompt_int("Max frames to send", int(os.getenv("MAX_FRAMES", "60")))
        args["batch_size"] = _prompt_int("Batch size (0=auto, use 8 for Nvidia free)", int(os.getenv("BATCH_SIZE", "0")))
    else:
        args["scene"] = float(os.getenv("SCENE_THRESHOLD", "0.1"))
        args["min_interval"] = float(os.getenv("MIN_INTERVAL", "5"))
        args["scale"] = int(os.getenv("FRAME_WIDTH", "512"))
        args["max_frames"] = int(os.getenv("MAX_FRAMES", "60"))
        args["batch_size"] = int(os.getenv("BATCH_SIZE", "0"))

    args["timeout"] = int(os.getenv("REQUEST_TIMEOUT", "600"))
    args["max_cost"] = float(os.getenv("MAX_COST_USD", "1.00"))
    args["verbose"] = False
    args["fps"] = None

    return args


def main() -> int:
    load_dotenv()

    import argparse
    p = argparse.ArgumentParser(
        description="vidlizer — analyze a video frame-by-frame → JSON user-journey map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any omitted options will be asked interactively when running in a terminal.",
    )
    p.add_argument("video", nargs="?", type=Path, help="Path to video file")
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
    cli = p.parse_args()

    _print_banner()

    # Build args: interactive fills missing values
    iargs = interactive_args(cli.video)

    # CLI flags override interactive answers
    if cli.output:              iargs["output"] = cli.output
    if cli.model:               iargs["model"] = cli.model
    if cli.scene is not None:   iargs["scene"] = cli.scene
    if cli.min_interval is not None: iargs["min_interval"] = cli.min_interval
    if cli.fps is not None:     iargs["fps"] = cli.fps
    if cli.scale is not None:   iargs["scale"] = cli.scale
    if cli.max_frames is not None: iargs["max_frames"] = cli.max_frames
    if cli.batch_size is not None: iargs["batch_size"] = cli.batch_size
    if cli.timeout is not None: iargs["timeout"] = cli.timeout
    if cli.max_cost is not None: iargs["max_cost"] = cli.max_cost
    if cli.verbose:             iargs["verbose"] = True

    _print_config(iargs)
    _console.print(Rule(style="dim"))
    _console.print()

    from vidlizer.core import run
    return run(**iargs)


if __name__ == "__main__":
    sys.exit(main())
