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

def _config_env_path() -> Path:
    """Canonical per-user config: ~/.config/vidlizer/.env (override via VIDLIZER_CONFIG_DIR)."""
    d = os.getenv("VIDLIZER_CONFIG_DIR")
    return (Path(d) if d else Path.home() / ".config" / "vidlizer") / ".env"


def _load_dotenv() -> None:
    """Load .env with precedence: system env > cwd/.env > ~/.config/vidlizer/.env"""
    cwd_env = Path.cwd() / ".env"
    cfg_env = _config_env_path()
    load_dotenv(cwd_env, override=False)   # project override (highest after system env)
    load_dotenv(cfg_env, override=False)   # user config fallback


_CURATED_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-pro",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-27b-it:free",
]

_MODEL_NOTES = {
    "google/gemini-2.5-flash":             "Recommended — fast, accurate",
    "google/gemini-2.5-flash-lite":        "Cheaper, slightly less accurate",
    "google/gemini-2.5-pro":               "Best quality, expensive",
    "openai/gpt-4o-mini":                  "OpenAI budget option",
    "openai/gpt-4o":                       "OpenAI flagship, expensive",
    "nvidia/nemotron-nano-12b-v2-vl:free": "Free ⚡ rate-limited, 10-image cap (auto-batched)",
    "google/gemma-3-27b-it:free":          "Free ⚡ rate-limited, may be slow",
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
    prov = args.get("provider", "")
    if prov == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        t.add_row("provider", f"[bold green]ollama[/bold green]  [dim]{ollama_host}[/dim]")
    elif prov == "openai":
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        t.add_row("provider", f"[bold cyan]openai-compat[/bold cyan]  [dim]{base}[/dim]")
    elif prov == "openrouter":
        t.add_row("provider", "[bold magenta]openrouter[/bold magenta]  [dim]cloud[/dim]")
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


def interactive_args(video: Path | None, output_format: str = "json", output: Path | None = None, skip_advanced: bool = False) -> dict:
    """Ask for any missing configuration interactively."""
    _load_dotenv()
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
    if output is not None:
        args["output"] = output
    elif interactive:
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
    if interactive and not skip_advanced:
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
    args["concurrency"] = 0  # 0 = auto-detect from provider

    return args


def _cmd_setup() -> int:
    """Interactive first-run wizard: detect providers, configure .env."""
    from vidlizer.detect import (
        check_ollama, check_lmstudio, check_omlx, check_openrouter,
        pick_best_vision, OL_PREFS, OLLAMA_MINIMAL,
    )
    _load_dotenv()
    _print_banner()
    _console.print("[bold]Setup Wizard[/bold]  [dim]— configures your .env[/dim]\n")

    # ── ffmpeg ───────────────────────────────────────────────────────────────
    from vidlizer.bootstrap import ensure_ffmpeg
    if not ensure_ffmpeg(_console):
        _console.print("[red]ffmpeg is required. Install from https://brew.sh then re-run.[/red]")
        return 1

    # ── Detect providers ─────────────────────────────────────────────────────
    _console.print("\n[dim]Scanning for local providers…[/dim]")
    ollama_ok, ol_host, ol_mdls    = check_ollama()
    lms_ok, lms_base, lms_mdls     = check_lmstudio()
    omlx_ok, omlx_base, omlx_mdls  = check_omlx()
    or_ok, or_detail, or_model     = check_openrouter()

    # Custom port retry for unreachable local providers
    def _retry_port(label: str, default_port: int) -> int | None:
        try:
            raw = input(f"  {label} not reachable at :{default_port}. Custom port? (Enter to skip): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 65535:
                return int(raw)
        except (EOFError, KeyboardInterrupt):
            pass
        return None

    if not ollama_ok:
        port = _retry_port("Ollama", 11434)
        if port:
            ollama_ok, ol_host, ol_mdls = check_ollama(f"http://localhost:{port}")
    if not lms_ok:
        port = _retry_port("LM Studio", 1234)
        if port:
            lms_ok, lms_base, lms_mdls = check_lmstudio(f"http://localhost:{port}/v1")
    if not omlx_ok:
        port = _retry_port("oMLX", 8000)
        if port:
            omlx_ok, omlx_base, omlx_mdls = check_omlx(f"http://localhost:{port}/v1")

    # Build candidate list — auto-detected + always-available manual options
    # name, base_url, detected_model
    candidates: list[tuple[str, str, str]] = []
    if ollama_ok:   candidates.append(("Ollama",              ol_host,   pick_best_vision(ol_mdls, OL_PREFS) or ""))
    if lms_ok:      candidates.append(("LM Studio",           lms_base,  pick_best_vision(lms_mdls, OL_PREFS) or ""))
    if omlx_ok:     candidates.append(("oMLX",                omlx_base, pick_best_vision(omlx_mdls, OL_PREFS) or ""))
    if or_ok:       candidates.append(("OpenRouter",           "",        or_model))
    # Manual options always available (even if not auto-detected)
    if not or_ok:   candidates.append(("OpenRouter",           "",        ""))
    candidates.append(    ("Custom OpenAI-compatible",  "",        ""))

    # Pick primary provider
    _console.print("\n[bold]Available providers:[/bold]")
    for i, (name, url, model) in enumerate(candidates, 1):
        hint = f"  [dim]{url}[/dim]" if url else ""
        if model:
            model_hint = f"  → {model}"
        elif name in ("OpenRouter", "Custom OpenAI-compatible"):
            model_hint = "  [dim](will configure)[/dim]"
        else:
            model_hint = "  [yellow]→ no vision model[/yellow]"
        detected = "  [green dim]detected[/green dim]" if url else ""
        _console.print(f"  [bold]{i}[/bold].  [magenta]{name:<24}[/magenta]{hint}[dim]{model_hint}[/dim]{detected}")

    primary_idx = None
    while primary_idx is None:
        try:
            raw = input(f"\n  Primary provider (1–{len(candidates)}): ").strip()
            n = int(raw)
            if 1 <= n <= len(candidates):
                primary_idx = n - 1
            else:
                _console.print(f"  [yellow]Enter 1–{len(candidates)}.[/yellow]")
        except ValueError:
            _console.print("  [yellow]Enter a number.[/yellow]")
        except (EOFError, KeyboardInterrupt):
            _console.print("\n  [dim]Cancelled.[/dim]")
            return 130

    primary_name, primary_url, primary_model = candidates[primary_idx]

    # Ollama: offer pull if no vision model
    if primary_name == "Ollama" and not primary_model:
        _console.print("\n  [yellow]No vision model found.[/yellow]")
        if _prompt_confirm(f"Download {OLLAMA_MINIMAL} (~3.2 GB)?", default=True):
            import subprocess as _sp
            env = {**os.environ, "OLLAMA_HOST": ol_host}
            _console.print(f"  [cyan]↓ ollama pull {OLLAMA_MINIMAL}[/cyan]")
            r = _sp.run(["ollama", "pull", OLLAMA_MINIMAL], env=env)
            primary_model = OLLAMA_MINIMAL if r.returncode == 0 else ""

    # Custom OpenAI-compatible: prompt for URL, key, model
    custom_openai_key = ""
    if primary_name == "Custom OpenAI-compatible":
        _console.print("\n  [bold]Custom OpenAI-compatible server[/bold]")
        _console.print("  [dim]Supports: LM Studio, vLLM, LocalAI, Ollama /v1, real OpenAI, any compatible server[/dim]")
        try:
            primary_url = input("  Base URL (e.g. http://localhost:1234/v1): ").strip() or "http://localhost:1234/v1"
            custom_openai_key = input("  API key (Enter for 'lm-studio'): ").strip() or "lm-studio"
        except (EOFError, KeyboardInterrupt):
            primary_url = "http://localhost:1234/v1"
            custom_openai_key = "lm-studio"
        # Try to detect models from the given URL
        from vidlizer.detect import _probe_openai_compat
        ok, mdls = _probe_openai_compat(primary_url, custom_openai_key)
        if ok and mdls:
            vision = pick_best_vision(mdls, OL_PREFS)
            primary_model = vision or mdls[0]
            _console.print(f"  [green]✓[/green]  Detected model: [magenta]{primary_model}[/magenta]")
        else:
            _console.print("  [yellow]Could not reach server — enter model name manually.[/yellow]")
            try:
                primary_model = input("  Model name: ").strip()
            except (EOFError, KeyboardInterrupt):
                primary_model = ""

    # OpenRouter: always prompt for key and model when selected
    or_api_key = os.getenv("OPENROUTER_API_KEY", "")
    or_model_choice = or_model  # detected free model, used as default
    _or_involved = primary_name == "OpenRouter"

    # Pick fallback (optional) — exclude primary from list
    remaining = [(i, c) for i, c in enumerate(candidates) if i != primary_idx]
    fallback: tuple[str, str, str] | None = None
    if remaining:
        _console.print("\n  Fallback provider (Enter to skip):")
        for n, (orig_i, (name, url, model)) in enumerate(remaining, 1):
            hint = f"  [dim]{url}[/dim]" if url else ""
            _console.print(f"    [bold]{n}[/bold].  [magenta]{name:<24}[/magenta]{hint}")
        try:
            raw = input(f"  Fallback (1–{len(remaining)}, Enter to skip): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(remaining):
                fallback = remaining[int(raw) - 1][1]
                if fallback[0] == "OpenRouter":
                    _or_involved = True
        except (EOFError, KeyboardInterrupt):
            pass

    # OpenRouter: always prompt for key + model when involved
    if _or_involved:
        _console.print("\n  [bold]OpenRouter configuration[/bold]")
        key_hint = f" [dim](current: ...{or_api_key[-4:]})[/dim]" if or_api_key else ""
        _console.print(f"  API key{key_hint}")
        try:
            raw = input("  sk-or-v1-… (Enter to keep current): ").strip()
            if raw:
                or_api_key = raw
        except (EOFError, KeyboardInterrupt):
            pass

        _console.print("\n  Model (curated list — Enter for recommended):")
        _OR_MODELS = [
            ("google/gemini-2.5-flash",             "Recommended — fast, accurate, cheap"),
            ("google/gemini-2.5-flash-lite",         "Cheaper, slightly less accurate"),
            ("google/gemini-2.5-pro",                "Best quality, expensive"),
            ("openai/gpt-4o-mini",                   "OpenAI budget option"),
            ("nvidia/nemotron-nano-12b-v2-vl:free",  "Free — rate-limited"),
            ("google/gemma-3-27b-it:free",           "Free — rate-limited"),
            ("custom",                               "Enter model slug manually"),
        ]
        for n, (mid, note) in enumerate(_OR_MODELS, 1):
            _console.print(f"    [bold]{n}[/bold].  {mid:<42}[dim]{note}[/dim]")
        try:
            raw = input(f"  Model (1–{len(_OR_MODELS)}, Enter=1): ").strip()
            idx = (int(raw) - 1) if raw.isdigit() and 1 <= int(raw) <= len(_OR_MODELS) else 0
            or_model_choice = _OR_MODELS[idx][0]
            if or_model_choice == "custom":
                or_model_choice = input("  Model slug: ").strip() or "google/gemini-2.5-flash"
        except (EOFError, KeyboardInterrupt):
            or_model_choice = "google/gemini-2.5-flash"

        if primary_name == "OpenRouter":
            primary_model = or_model_choice

    # ── Build .env values ────────────────────────────────────────────────────
    def _prov_key(name: str) -> str:
        return {"Ollama": "ollama", "LM Studio": "openai", "oMLX": "openai",
                "Custom OpenAI-compatible": "openai",
                "OpenRouter": "openrouter"}.get(name, "ollama")

    env_values: dict[str, str] = {
        "PROVIDER": _prov_key(primary_name),
    }
    if primary_name == "Ollama":
        env_values["OLLAMA_HOST"]  = primary_url or "http://localhost:11434"
        env_values["OLLAMA_MODEL"] = primary_model or OLLAMA_MINIMAL
    elif primary_name in ("LM Studio", "oMLX", "Custom OpenAI-compatible"):
        env_values["OPENAI_BASE_URL"] = primary_url or "http://localhost:1234/v1"
        env_values["OPENAI_MODEL"]    = primary_model or ""
        env_values["OPENAI_API_KEY"]  = custom_openai_key or os.getenv("OPENAI_API_KEY", "lm-studio")
    elif primary_name == "OpenRouter":
        env_values["OPENROUTER_API_KEY"] = or_api_key
        env_values["OPENROUTER_MODEL"]   = primary_model or "google/gemini-2.5-flash"

    if fallback:
        fb_name, fb_url, fb_model = fallback
        env_values["FALLBACK_PROVIDER"] = _prov_key(fb_name)
        env_values["FALLBACK_MODEL"]    = fb_model or ""
        if fb_url:
            env_values["FALLBACK_BASE_URL"] = fb_url
        if fb_name == "OpenRouter":
            env_values["FALLBACK_API_KEY"] = or_api_key

    # ── Write .env ────────────────────────────────────────────────────────────
    env_path = _config_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # packaged install: env.sample lives inside the package directory
    # git clone: also check repo root as fallback
    env_sample = Path(__file__).parent / "env.sample"
    if not env_sample.exists():
        env_sample = Path(__file__).parent.parent / "env.sample"

    if env_sample.exists():
        lines = env_sample.read_text().splitlines()
        out_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                out_lines.append(line)
                continue
            key = stripped.split("=", 1)[0]
            if key in env_values:
                out_lines.append(f"{key}={env_values[key]}")
            else:
                out_lines.append(line)
        env_content = "\n".join(out_lines) + "\n"
    else:
        env_content = "".join(f"{k}={v}\n" for k, v in env_values.items())

    if env_path.exists():
        if not _prompt_confirm(f".env already exists at {env_path}. Overwrite?", default=False):
            _console.print("  [dim]Skipped .env write.[/dim]")
            return 0

    env_path.write_text(env_content)
    env_path.chmod(0o600)  # secrets: owner-read only
    _console.print(f"\n[green]✓[/green]  .env written → [cyan]{env_path}[/cyan]")
    _console.print("\n[dim]Run [bold]vidlizer doctor[/bold] to verify your setup.[/dim]")
    _console.print("[dim]Run [bold]vidlizer video.mp4[/bold] to analyze a file.[/dim]\n")
    return 0


def _cmd_doctor() -> int:
    """Read-only health check — shows provider and dependency status."""
    from vidlizer.detect import (
        check_ffmpeg, check_ollama, check_lmstudio, check_omlx,
        check_openrouter, check_whisper, pick_best_vision, OL_PREFS,
    )
    _load_dotenv()
    _print_banner()
    _console.print("[bold]Doctor[/bold]  [dim]— system health check[/dim]\n")

    from rich.table import Table as _Table
    t = _Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Check", style="white", min_width=18)
    t.add_column("Status", min_width=8)
    t.add_column("Detail", style="dim")

    def _row(label: str, ok: bool | None, detail: str = "") -> None:
        if ok is True:
            badge = "[bold green]✓ OK[/bold green]"
        elif ok is False:
            badge = "[bold red]✗ FAIL[/bold red]"
        else:
            badge = "[dim]— SKIP[/dim]"
        t.add_row(label, badge, detail)

    # ffmpeg
    ffmpeg_ok, ffmpeg_ver = check_ffmpeg()
    _row("ffmpeg", ffmpeg_ok, ffmpeg_ver if ffmpeg_ok else "not found — run: brew install ffmpeg")

    # .env — check cwd first (project override), then user config
    cwd_env = Path.cwd() / ".env"
    env_path = cwd_env if cwd_env.exists() else _config_env_path()
    _row(".env file", env_path.exists(),
         str(env_path) if env_path.exists() else "missing — run: vidlizer setup")

    # OpenRouter key
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    _row("OPENROUTER_API_KEY", bool(or_key),
         f"...{or_key[-4:]}" if or_key else "not set (required for cloud provider)")

    # Ollama
    ollama_ok, ol_host, ol_mdls = check_ollama()
    if ollama_ok:
        vision = pick_best_vision(ol_mdls, OL_PREFS)
        _row("Ollama", True,
             f"{ol_host}  →  {vision or 'no vision model'}"
             + ("" if vision else "  — run: ollama pull qwen2.5vl:3b"))
    else:
        _row("Ollama", None, f"{ol_host} not reachable")

    # LM Studio
    lms_ok, lms_base, lms_mdls = check_lmstudio()
    if lms_ok:
        vision = pick_best_vision(lms_mdls, OL_PREFS)
        _row("LM Studio", True, f"{lms_base}  →  {vision or 'no vision model'}")
    else:
        _row("LM Studio", None, f"{lms_base} not reachable")

    # oMLX
    omlx_ok, omlx_base, omlx_mdls = check_omlx()
    if omlx_ok:
        vision = pick_best_vision(omlx_mdls, OL_PREFS)
        _row("oMLX", True, f"{omlx_base}  →  {vision or 'no vision model'}")
    else:
        _row("oMLX", None, f"{omlx_base} not reachable")

    # OpenRouter
    or_ok, or_detail, or_model = check_openrouter()
    _row("OpenRouter", or_ok if or_key else None,
         f"{or_detail}  →  {or_model}" if or_ok else or_detail)

    # mlx-whisper
    whisper_ok, whisper_det = check_whisper()
    _row("mlx-whisper", whisper_ok,
         whisper_det if whisper_ok else "optional — pip install 'vidlizer[transcribe]'")

    _console.print(t)
    _console.print()

    any_fail = not ffmpeg_ok or not env_path.exists()
    if any_fail:
        _console.print("[dim]Fix issues above, then run [bold]vidlizer setup[/bold] or re-check.[/dim]\n")
    else:
        _console.print("[green]All core checks passed.[/green]\n")
    return 0 if not any_fail else 1


def main() -> int:
    try:
        return _main()
    except KeyboardInterrupt:
        _console.print("\n[dim]cancelled[/dim]")
        return 130


def _main() -> int:
    # Subcommands — intercept before argparse so "vidlizer setup" isn't
    # parsed as a video path.
    if sys.argv[1:2] == ["setup"]:
        return _cmd_setup()
    if sys.argv[1:2] == ["doctor"]:
        return _cmd_doctor()

    _load_dotenv()

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
    p.add_argument("--concurrency", type=int, default=None,
                   help="Parallel batch workers (default: 4 for OpenRouter, 1 for local providers)")
    p.add_argument("--stats", action="store_true",
                   help="Show token and cost usage statistics across all runs, then exit")
    p.add_argument("--clear-stats", action="store_true", dest="clear_stats",
                   help="Reset usage statistics (delete usage log), then exit")
    cli = p.parse_args()

    if cli.stats:
        from vidlizer.usage import get_stats
        stats = get_stats()
        _console.print(f"\n[bold]Usage statistics[/bold]  ([dim]{stats['log_path']}[/dim])\n")
        _console.print(f"  Total runs:       [cyan]{stats['total_runs']}[/cyan]")
        _console.print(f"  Total tokens in:  [cyan]{stats['total_tokens_in']:,}[/cyan]")
        _console.print(f"  Total tokens out: [cyan]{stats['total_tokens_out']:,}[/cyan]")
        cost = f"[bold green]~${stats['total_cost_usd']:.4f}[/bold green]" if stats['total_cost_usd'] else "[bold cyan]free[/bold cyan]"
        _console.print(f"  Total cost:       {cost}")
        _console.print(f"  Total steps:      [cyan]{stats['total_steps']:,}[/cyan]\n")
        if stats["by_model"]:
            from rich.table import Table
            t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
            t.add_column("Model")
            t.add_column("Provider")
            t.add_column("Runs", justify="right")
            t.add_column("Tokens in", justify="right")
            t.add_column("Tokens out", justify="right")
            t.add_column("Cost USD", justify="right")
            for row in stats["by_model"]:
                c = f"~${row['cost_usd']:.4f}" if row["cost_usd"] else "free"
                t.add_row(
                    row["model"], row["provider"], str(row["runs"]),
                    f"{row['tokens_in']:,}", f"{row['tokens_out']:,}", c,
                )
            _console.print(t)
        _console.print()
        return 0

    if cli.clear_stats:
        from vidlizer.usage import clear_stats
        n = clear_stats()
        _console.print(f"[green]✓[/green] Usage log cleared ({n} records deleted)")
        return 0

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
        # Build args: interactive fills missing values (skip prompts for already-specified flags)
        _has_advanced = any(v is not None for v in [
            cli.scene, cli.min_interval, cli.fps, cli.scale, cli.max_frames,
            cli.batch_size, cli.timeout, cli.max_cost, cli.start, cli.end,
        ])
        iargs = interactive_args(video_path, output_format=cli.output_format,
                                 output=cli.output, skip_advanced=_has_advanced)

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
        if cli.concurrency is not None:      iargs["concurrency"] = cli.concurrency
        iargs["output_format"] = cli.output_format

        _print_config(iargs)
        _console.print(Rule(style="dim"))
        _console.print()

        from vidlizer.core import run
        return run(**iargs)


if __name__ == "__main__":
    sys.exit(main())
