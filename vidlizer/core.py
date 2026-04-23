"""Video analysis orchestrator: provider setup, frame pipeline, fallback, result writing."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from vidlizer import cache as _cache
from vidlizer.batch import call_model, merge_transcript
from vidlizer.dedup import DEFAULT_THRESHOLD as _DEDUP_DEFAULT, dedup_frames
from vidlizer.frames import _IMAGE_EXTS, _PDF_EXTS, extract_frames, pdf_to_frames
from vidlizer.http import CostCapExceeded, CostTracker
from vidlizer.models import get_pricing

_console = Console(stderr=True, highlight=False)

_live_models: list[dict] = []

_EXPENSIVE_THRESHOLD_USD_PER_1M = 1.00
_PREVIEW_MAX_LINES = 24


def _info(msg: str) -> None:
    _console.print(f"[cyan]→[/cyan] {msg}")


def _warn(msg: str) -> None:
    _console.print(f"[yellow]⚠[/yellow]  {msg}")


def _err(msg: str) -> None:
    _console.print(f"[red]✗[/red]  [red]{msg}[/red]")


def _dbg(msg: str) -> None:
    _console.print(f"[dim]{msg}[/dim]", markup=False)


def log(msg: str, verbose: bool = False, always: bool = False) -> None:
    if always or verbose:
        _console.print(msg, markup=False)


def _show_result_preview(data: dict) -> None:
    steps = data.get("flow", [])
    n = len(steps)
    if n == 0:
        return
    preview_str = json.dumps({"flow": steps[:2]}, indent=2, ensure_ascii=False)
    lines = preview_str.splitlines()
    truncated = len(lines) > _PREVIEW_MAX_LINES
    visible = "\n".join(lines[:_PREVIEW_MAX_LINES])
    if truncated:
        visible += f"\n  // … {len(lines) - _PREVIEW_MAX_LINES} more lines ({n} steps total)"
    syntax = Syntax(visible, "json", theme="monokai", line_numbers=True, word_wrap=False)
    _console.print(Panel(
        syntax,
        title=f"[dim]preview  [white]{n} steps[/white][/dim]",
        border_style="dim",
        padding=(0, 1),
    ))


def _setup_provider(
    provider: str, model: str,
) -> tuple[str | None, str | None, dict | None, bool, bool, list[str], str, str, list[dict]]:
    """Return (api_key, endpoint, req_headers, is_ollama, is_openai_compat,
               installed_models, base_url, local_api_key, live_models)."""
    global _live_models
    _is_ollama = provider == "ollama"
    _is_openai = provider == "openai"

    if _is_ollama:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        endpoint: str | None = f"{host}/api/chat"
        req_headers: dict | None = None
        api_key = None
        _live_models = []
        _info(f"provider: [bold]ollama[/bold]  [dim]{host}[/dim]")
        from vidlizer.models import fetch_ollama_models
        with _console.status("[dim]checking Ollama…[/dim]", spinner="dots2"):
            installed = fetch_ollama_models(host)
        if not installed:
            _warn(f"Ollama not reachable at {host} — start with: [cyan]ollama serve[/cyan]")
        elif not any(i.split(":")[0] == model.split(":")[0] for i in installed):
            _warn(f"model [bold]{model}[/bold] not installed — run: [cyan]ollama pull {model}[/cyan]")
        return api_key, endpoint, req_headers, True, False, installed, host, "", []

    if _is_openai:
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        endpoint = f"{base}/chat/completions"
        local_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        req_headers = {"Authorization": f"Bearer {local_key}", "Content-Type": "application/json"}
        _live_models = []
        _info(f"provider: [bold]openai-compat[/bold]  [dim]{base}[/dim]")
        return local_key, endpoint, req_headers, False, True, [], base, local_key, []

    # openrouter
    endpoint = None
    req_headers = None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing (set in .env or export in shell)")
    from vidlizer.models import fetch_models
    with _console.status("[dim]fetching model pricing…[/dim]", spinner="dots2"):
        _live_models = fetch_models(api_key)
    import vidlizer.http as _http_mod
    _http_mod._live_models = _live_models
    return api_key, endpoint, req_headers, False, False, [], "", "", _live_models


def run(
    video: Path,
    output: Path,
    model: str,
    scene: float = 0.1,
    min_interval: float = 2.0,
    fps: float | None = None,
    scale: int = 512,
    max_frames: int = 60,
    batch_size: int = 0,
    timeout: int = 600,
    verbose: bool = False,
    max_cost: float = 1.00,
    start: float | None = None,
    end: float | None = None,
    dedup_threshold: int = _DEDUP_DEFAULT,
    no_transcript: bool = False,
    output_format: str = "json",
    provider: str = "",
    concurrency: int = 0,
) -> int:
    v = verbose
    _provider = (provider or os.getenv("PROVIDER", "ollama")).lower()

    if not shutil.which("ffmpeg"):
        _err("ffmpeg not found on PATH")
        return 2

    try:
        (api_key, _endpoint, _req_headers,
         _is_ollama, _is_openai_compat,
         _installed, _base, _local_api_key, _) = _setup_provider(_provider, model)
    except ValueError as e:
        _err(str(e))
        return 2

    if batch_size == 0:
        batch_size = 1
        _info("auto-set [bold]batch_size=1[/bold] (override with --batch-size)")

    # Provider-aware concurrency: local providers serialize server-side anyway
    _is_local = _is_ollama or _is_openai_compat
    _default_concurrency = 1 if _is_local else 4
    if concurrency == 0:
        concurrency = int(os.getenv("CONCURRENCY", str(_default_concurrency)))
    if concurrency > 1 and _is_local:
        _warn(f"concurrency={concurrency} ignored for local providers (serialized server-side) — using 1")
        concurrency = 1
    if concurrency > 1 and (
        model.endswith(":free") or
        next((m.get("free", False) for m in _live_models if m["id"] == model), False)
    ):
        _warn("free model detected — forcing concurrency=1 to avoid rate limits")
        concurrency = 1
    if concurrency > 1:
        _info(f"[dim]concurrency: [bold]{concurrency}[/bold] parallel chunks[/dim]")

    # Build fallback sequence
    _fallback_models: list[str] = []
    _fallback_model_env = os.getenv("FALLBACK_MODEL", "").strip()
    _fallback_provider_env = os.getenv("FALLBACK_PROVIDER", "").lower().strip()
    _is_cross_provider_fallback = bool(_fallback_provider_env and _fallback_provider_env != _provider)
    if _fallback_model_env and _fallback_model_env != model and not _is_cross_provider_fallback:
        _fallback_models = [_fallback_model_env]
        _info(f"[dim]fallback model (env): {_fallback_model_env}[/dim]")
    elif _fallback_model_env and _is_cross_provider_fallback:
        _info(f"[dim]fallback: {_fallback_model_env} via {_fallback_provider_env}[/dim]")
    elif _is_ollama and _installed:
        from vidlizer.models import get_ollama_fallback_sequence
        _fallback_models = get_ollama_fallback_sequence(_installed, exclude=model)
        if _fallback_models:
            _info(f"[dim]fallback candidates: {', '.join(_fallback_models[:3])}[/dim]")
    elif _is_openai_compat and _base:
        from vidlizer.models import fetch_openai_compat_models, get_openai_fallback_sequence
        _avail = fetch_openai_compat_models(_base, _local_api_key)
        if len(_avail) > 1:
            _fallback_models = get_openai_fallback_sequence(_avail, exclude=model)
            _info(f"[dim]fallback candidates: {', '.join(_fallback_models[:3])}[/dim]")

    if max_frames > 200:
        _warn(f"max_frames={max_frames} is high — runaway-cost risk. Capping to 200.")
        max_frames = 200
    if not _is_ollama and not _is_openai_compat:
        inp_rate, out_rate = get_pricing(model, _live_models)
        if inp_rate > _EXPENSIVE_THRESHOLD_USD_PER_1M:
            _warn(f"[bold]{model}[/bold] is an expensive model "
                  f"(${inp_rate:.2f}/M input, ${out_rate:.2f}/M output). "
                  f"Cost cap: [bold]${max_cost:.2f}[/bold]")

    import contextlib

    is_image = video.suffix.lower() in _IMAGE_EXTS
    is_pdf = video.suffix.lower() in _PDF_EXTS

    if is_image:
        _info(f"image input: [bold]{video.name}[/bold]  [dim]({video.stat().st_size // 1024} KB)[/dim]")
    elif is_pdf:
        _info(f"PDF input: [bold]{video.name}[/bold]  [dim]({video.stat().st_size // 1024} KB)[/dim]")
    else:
        from vidlizer.preflight import show_preflight
        show_preflight(video, model, min_interval, max_frames, _live_models)

    tmp_ctx = contextlib.nullcontext(None) if is_image else tempfile.TemporaryDirectory(prefix="vidframes_")

    with tmp_ctx as tmp:
        frame_timestamps: list[float] | None = None
        if is_image:
            frames: list[Path] = [video]
        elif is_pdf:
            tmp_path = Path(tmp)  # type: ignore[arg-type]
            frames = pdf_to_frames(video, tmp_path, scale, max_frames)
            if not frames:
                _err("no pages extracted from PDF")
                return 1
        else:
            tmp_path = Path(tmp)  # type: ignore[arg-type]
            if v:
                _dbg(f"[debug] temp dir: {tmp_path}")
            frames = extract_frames(video, tmp_path, scale, max_frames, scene, fps, min_interval, v, start, end)
            if not frames:
                _err("no frames extracted — try --fps 0.5 or lower --scene threshold")
                return 1
            before = len(frames)
            frames = dedup_frames(frames, dedup_threshold)
            if len(frames) < before:
                _info(f"dedup: [bold]{before - len(frames)}[/bold] duplicate frames removed ({len(frames)} remain)")

            ts_json = tmp_path / ".timestamps.json"
            if ts_json.exists():
                try:
                    ts_map = json.loads(ts_json.read_text())
                    frame_timestamps = [ts_map[f.name] for f in frames if f.name in ts_map]
                    if len(frame_timestamps) != len(frames):
                        frame_timestamps = None
                except Exception:
                    frame_timestamps = None

        if is_image:
            label = "image"
        elif is_pdf:
            label = f"{len(frames)} pages"
        else:
            label = f"{len(frames)} frames"
        _info(
            f"[bold]{label}[/bold]  [dim]→[/dim]  [magenta]{model}[/magenta]  "
            f"[dim](batch={batch_size or 'auto'}, output: {output})[/dim]"
        )
        _console.print()

        cache_params = {
            "model": model, "max_frames": max_frames, "scene": scene,
            "min_interval": min_interval, "fps": str(fps), "scale": scale,
            "batch_size": batch_size, "start": str(start), "end": str(end),
            "dedup": dedup_threshold,
        }
        cached = _cache.get(video, cache_params)
        if cached is not None:
            _info("[dim]cache hit[/dim]")
            from vidlizer.formatter import format_output
            output.write_text(format_output(cached, output_format))
            _show_result_preview(cached)
            steps_c = len(cached.get("flow", []))
            _console.print(Panel(
                f"[green]✓[/green] [bold]{steps_c} steps[/bold] (cached) → [cyan]{output}[/cyan]",
                border_style="green", padding=(0, 1),
            ))
            return 0

        def _try_model_with(
            m: str, trk: CostTracker,
            key: str | None, ep: str | None, hdrs: dict | None,
            is_olla: bool, is_oai: bool,
        ) -> tuple[dict | None, int]:
            for attempt in range(1, 4):
                try:
                    return call_model(
                        key, m, frames, timeout, v, batch_size, trk,
                        is_image=is_image, timestamps=frame_timestamps,
                        endpoint=ep, req_headers=hdrs, is_ollama=is_olla,
                        no_stream_opts=is_oai, no_json_format=is_oai,
                        concurrency=concurrency,
                    ), 0
                except CostCapExceeded as e:
                    _err(str(e))
                    _warn("partial run — no output written. Raise MAX_COST_USD or use a cheaper model.")
                    return None, 3
                except RuntimeError as e:
                    if "rate_limited" in str(e) and attempt < 3:
                        wait = 15 * attempt
                        _warn(f"rate limited — retrying in [bold]{wait}s[/bold] (attempt {attempt}/3)")
                        time.sleep(wait)
                    else:
                        _err(f"[bold]{m}[/bold] failed: {e}")
                        return None, -1
                except (json.JSONDecodeError, requests.RequestException) as e:
                    _err(f"[bold]{m}[/bold] failed: {e}")
                    return None, -1
            return None, -1

        def _try_model(m: str, trk: CostTracker) -> tuple[dict | None, int]:
            return _try_model_with(m, trk, api_key, _endpoint, _req_headers, _is_ollama, _is_openai_compat)

        tracker = CostTracker(max_cost=max_cost)
        data, rc = _try_model(model, tracker)

        if data is None and rc == -1:
            if _is_cross_provider_fallback and _fallback_model_env:
                _fb_is_ollama = _fallback_provider_env == "ollama"
                _fb_is_oai = _fallback_provider_env == "openai"
                if _fb_is_ollama:
                    _fb_host = os.getenv("FALLBACK_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
                    _fb_ep: str | None = f"{_fb_host}/api/chat"
                    _fb_key: str | None = None
                    _fb_hdrs: dict | None = None
                elif _fb_is_oai:
                    _fb_base = os.getenv("FALLBACK_BASE_URL",
                                         os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")).rstrip("/")
                    _fb_ep = f"{_fb_base}/chat/completions"
                    _fb_key = os.getenv("FALLBACK_API_KEY", os.getenv("OPENAI_API_KEY", "lm-studio"))
                    _fb_hdrs = {"Authorization": f"Bearer {_fb_key}", "Content-Type": "application/json"}
                else:
                    _fb_ep = None
                    _fb_key = os.getenv("FALLBACK_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
                    _fb_hdrs = None
                _warn(f"primary failed — fallback [bold]{_fallback_model_env}[/bold] via [bold]{_fallback_provider_env}[/bold]")
                data, rc = _try_model_with(
                    _fallback_model_env, tracker,
                    _fb_key, _fb_ep, _fb_hdrs, _fb_is_ollama, _fb_is_oai,
                )
                if data is not None:
                    model = _fallback_model_env

            elif _fallback_models:
                for _fb in _fallback_models:
                    _warn(f"trying fallback: [bold]{_fb}[/bold]")
                    data, rc = _try_model(_fb, tracker)
                    if data is not None:
                        model = _fb
                        break

            elif not _is_ollama and not _is_openai_compat:
                is_free = next((m["free"] for m in _live_models if m["id"] == model), model.endswith(":free"))
                if is_free:
                    from vidlizer.models import get_cheapest_paid
                    fallback = get_cheapest_paid(_live_models)
                    _warn(f"free model failed — falling back to [bold]{fallback}[/bold]")
                    model = fallback
                    data, rc = _try_model(model, tracker)

        if data is None:
            _err("all retries exhausted" if rc == -1 else "")
            return max(rc, 1)

    steps = len(data.get("flow", []))
    if steps == 0:
        _warn("model returned 0 steps — check --verbose output for clues")

    if not is_image and not is_pdf and not no_transcript:
        from vidlizer.transcribe import has_audio, is_available, transcribe
        if has_audio(video):
            if not is_available():
                from vidlizer.bootstrap import ensure_transcriber
                ensure_transcriber(_console)
            if is_available():
                with _console.status("[dim]transcribing audio…[/dim]", spinner="dots2"):
                    segments = transcribe(video)
                if segments:
                    data["transcript"] = segments
                    merge_transcript(data["flow"], segments)
                    _info(f"transcript: [bold]{len(segments)} segments[/bold]  (merged into flow steps)")

    data["model_used"] = model
    data["provider_used"] = _provider

    _cache.put(video, cache_params, data)

    from vidlizer.formatter import format_output
    output.write_text(format_output(data, output_format))

    from vidlizer.usage import record_run
    record_run(
        model=model, provider=_provider,
        tokens_in=tracker.prompt_tokens, tokens_out=tracker.completion_tokens,
        cost_usd=tracker.cost_usd, source=str(video), steps=steps,
    )

    _console.print()
    _show_result_preview(data)
    _console.print()
    t = Table.grid(padding=(0, 2))
    t.add_column(style="green bold")
    t.add_column()
    t.add_row("✓", f"[bold]{steps} steps[/bold] written to [cyan]{output}[/cyan]")
    tok_text = Text()
    tok_text.append(f"{tracker.prompt_tokens:,}", style="cyan")
    tok_text.append("↑  ", style="dim")
    tok_text.append(f"{tracker.completion_tokens:,}", style="cyan")
    tok_text.append("↓  tokens", style="dim")
    if tracker.cost_usd > 0:
        tok_text.append(f"  |  ~${tracker.cost_usd:.4f}", style="bold green")
    else:
        tok_text.append("  |  free", style="bold cyan")
    t.add_row("", tok_text)
    _console.print(Panel(t, border_style="green", padding=(0, 1)))
    return 0
