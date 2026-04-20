#!/usr/bin/env python3
"""Frame-by-frame video analyzer → structured JSON event map via OpenRouter vision models."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import sys

import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from vidlizer.models import get_pricing

_console = Console(stderr=True, highlight=False)

# Runtime-populated by run() after fetching live models
_live_models: list[dict] = []


def _model_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = get_pricing(model, _live_models or None)
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class CostCapExceeded(RuntimeError):
    """Raised when accumulated cost crosses the configured cap."""


class CostTracker:
    def __init__(self, max_cost: float = 0.0) -> None:
        # max_cost=0 means no cap
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.batches_done = 0
        self.batches_total = 0
        self.max_cost = max(0.0, max_cost)

    def add(self, model: str, usage: dict) -> float:
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        c = _model_cost(model, pt, ct)
        self.cost_usd += c
        self.batches_done += 1
        if self.max_cost > 0 and self.cost_usd > self.max_cost:
            raise CostCapExceeded(
                f"cost cap ${self.max_cost:.2f} exceeded "
                f"(spent ~${self.cost_usd:.4f}) — aborting"
            )
        return c

    def summary(self) -> str:
        tok = f"{self.prompt_tokens:,}↑  {self.completion_tokens:,}↓"
        cost = f"~${self.cost_usd:.4f}" if self.cost_usd > 0 else "free"
        return f"tokens: {tok}  |  cost: {cost}"


PROMPT = """Role: Act as an expert Video Analyst.

Task: Analyze the provided frames and map the complete sequence of events into a single exhaustive JSON object. This applies to ANY type of video: software/app recordings, tutorials, interviews, product demos, marketing content, physical activities, presentations, or anything else.

Observational Rules:
- Zero Compression: Every meaningful change between frames is a step. Do not skip.
- Adapt to Content: Infer the video type and use appropriate terminology — UI interaction, physical action, narrative event, speech, etc.
- Precision: Only describe what is visually evident. Note uncertainty where present.
- Text: Capture all visible text — UI labels, captions, subtitles, titles, overlays, on-screen code.
- Issues: Flag errors, anomalies, quality problems, or anything unexpected.
- Persistence: Track any ongoing state — timer, score, progress bar, speaker identity, topic, brand, etc.

JSON Schema:
Produce a single JSON object with a `flow` array. Each element MUST include:
- step: Sequential integer.
- phase: Logical section (e.g. "Introduction", "Demo", "Action", "Conclusion", "Navigation", "Dialogue").
- scene: What is currently visible — the setting, screen, environment, or context.
- subjects: Key people, objects, UI elements, or entities present.
- action: What is happening — interaction, physical movement, narration, transition, or event.
- text_visible: All readable text on screen (UI labels, captions, code, titles, overlays). Empty string if none.
- context: Persistent state or background information relevant to this moment.
- observations: Notable details — emotions, errors, quality issues, key facts, or anomalies.
- next_scene: Brief description of what follows.

Output: Provide ONLY a valid JSON object with the `flow` array. No prose, no code fences. 100% accuracy based on visual evidence."""


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


def extract_frames(
    video: Path,
    out_dir: Path,
    scale: int,
    max_frames: int,
    scene_threshold: float,
    fps: float | None,
    min_interval: float,
    verbose: bool,
) -> list[Path]:
    """Hybrid extraction: scene-change OR time-based minimum interval."""
    if fps is not None:
        vf = f"fps={fps},scale={scale}:-2:flags=lanczos,format=yuvj420p"
        mode_desc = f"fixed fps={fps}"
    else:
        # Select frame if: scene changed OR enough time has passed since last selected frame
        # isnan(prev_selected_t) catches the very first frame
        vf = (
            f"select='gt(scene\\,{scene_threshold})+isnan(prev_selected_t)"
            f"+gte(t-prev_selected_t\\,{min_interval})',"
            f"scale={scale}:-2:flags=lanczos,format=yuvj420p"
        )
        mode_desc = f"scene>{scene_threshold} or every {min_interval}s"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning" if verbose else "error",
        "-y", "-i", str(video),
        "-vf", vf,
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        str(out_dir / "f_%05d.jpg"),
    ]

    _info(f"ffmpeg mode: [bold]{mode_desc}[/bold]")
    if verbose:
        _dbg(f"[ffmpeg] cmd: {' '.join(cmd)}")

    with _console.status("[dim]extracting frames…[/dim]", spinner="dots2"):
        subprocess.run(cmd, check=True)

    frames = sorted(out_dir.glob("f_*.jpg"))
    if verbose:
        total_kb = sum(f.stat().st_size for f in frames) / 1024
        _dbg(f"[frames] extracted {len(frames)} frames, total {total_kb:.1f} KB")
        for i, f in enumerate(frames):
            _dbg(f"  [{i+1:03d}] {f.name}  {f.stat().st_size/1024:.1f} KB")
    return frames


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}


def encode_frame(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = _MIME.get(path.suffix.lower(), "image/jpeg")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


PROMPT_CONTINUE = """Role: Act as an expert Video Analyst.

These frames continue a video analysis already in progress.
Start step numbering from {step_offset}. Current phase context: {phase_context}.

Apply all the same rules: Zero Compression, adapt to content type, capture all text, flag issues, track persistent state.

Output: Provide ONLY a valid JSON object with the `flow` array continuing from step {step_offset}. No prose, no code fences."""


PROMPT_IMAGE = """Role: Act as an expert Image Analyst.

Task: Analyze this single image and describe it completely in exactly ONE step.

JSON Schema:
Return a JSON object with a `flow` array containing exactly one element:
- step: 1
- phase: "Image Analysis"
- scene: Full description of what is visible — setting, environment, layout.
- subjects: All people, objects, UI elements, or entities present.
- action: Any activity, state, or interaction evident in the image.
- text_visible: All readable text (labels, captions, code, titles, overlays). Empty string if none.
- context: Inferred purpose, brand, topic, or background information.
- observations: Notable details — quality, errors, emotions, anomalies, key facts.
- next_scene: null

Output: Provide ONLY a valid JSON object with the `flow` array containing exactly one step. No prose, no code fences."""


class ImageLimitError(RuntimeError):
    """Raised when the model rejects the request due to too many images."""
    pass


def _post(
    api_key: str,
    model: str,
    payload: dict,
    timeout: int,
    verbose: bool,
    tracker: CostTracker | None = None,
    label: str = "",
    n_frames: int = 0,
) -> dict:
    stream_payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    payload_bytes = json.dumps(stream_payload).encode()
    kb = len(payload_bytes) // 1024

    frame_info = f"  [white]{n_frames}f[/white]" if n_frames else ""
    _console.print(
        f"[cyan]→[/cyan] [bold]{label}[/bold]{frame_info}  "
        f"[dim]{kb} KB[/dim]  [dim]→[/dim]  [magenta]{model}[/magenta]"
    )
    if verbose:
        _dbg(f"[debug] POST stream=True payload={kb} KB")

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/arizawan/vidlizer",
            "X-Title": "vidlizer",
        },
        data=payload_bytes,
        timeout=timeout,
        stream=True,
    )

    if r.status_code == 429:
        raise RuntimeError(f"rate_limited: {r.text[:500]}")
    if not r.ok:
        err_text = r.text[:1000]
        if "image" in err_text.lower() and ("most" in err_text.lower() or "limit" in err_text.lower()):
            raise ImageLimitError(err_text)
        raise RuntimeError(f"OpenRouter {r.status_code}: {err_text}")

    full_content = ""
    usage: dict = {}

    sys.stderr.write("   \033[2m")  # indent + dim ANSI
    sys.stderr.flush()

    for raw_line in r.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if chunk.get("usage"):
            usage = chunk["usage"]

        if "error" in chunk:
            sys.stderr.write("\033[0m\n")
            sys.stderr.flush()
            err = chunk["error"]
            err_str = json.dumps(err) if isinstance(err, dict) else str(err)
            if "image" in err_str.lower() and ("most" in err_str.lower() or "limit" in err_str.lower()):
                raise ImageLimitError(err_str)
            raise RuntimeError(f"OpenRouter error: {err_str}")

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                full_content += piece
                sys.stderr.write(piece)
                sys.stderr.flush()

    sys.stderr.write("\033[0m\n")  # reset + newline
    sys.stderr.flush()

    if tracker:
        batch_cost = tracker.add(model, usage)  # may raise CostCapExceeded
    else:
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        batch_cost = _model_cost(model, pt, ct)

    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    cost_str = f"[green]~${batch_cost:.4f}[/green]" if batch_cost > 0 else "[cyan]free[/cyan]"
    _console.print(
        f"   [dim]tokens:[/dim] [cyan]{pt:,}[/cyan][dim]↑[/dim]  "
        f"[cyan]{ct:,}[/cyan][dim]↓[/dim]  {cost_str}"
    )

    if verbose:
        _dbg(f"[debug] stream complete, content={len(full_content)} chars")

    return {
        "choices": [{"message": {"content": full_content}}],
        "usage": usage,
    }


_MAX_RECURSION_DEPTH = 4


def call_openrouter(
    api_key: str,
    model: str,
    frames: list[Path],
    timeout: int,
    verbose: bool,
    batch_size: int,
    tracker: CostTracker | None = None,
    _depth: int = 0,
    is_image: bool = False,
) -> dict:
    """Send frames to OpenRouter. Auto-retries with batching on image-limit errors."""
    if _depth > _MAX_RECURSION_DEPTH:
        raise RuntimeError(
            f"batching recursion depth {_depth} exceeded — "
            f"model rejects {len(frames)} frame(s) even at batch_size={batch_size}"
        )
    if tracker is None:
        tracker = CostTracker()
    if batch_size <= 0 or len(frames) <= batch_size:
        # Single request
        content = [{"type": "text", "text": PROMPT_IMAGE if is_image else PROMPT}]
        content.extend(encode_frame(f) for f in frames)
        tracker.batches_total = 1
        try:
            body = _post(api_key, model, {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }, timeout, verbose, tracker, label="[1/1]", n_frames=len(frames))
            result = parse_json(body["choices"][0]["message"]["content"])
            _console.print(
                f"  [green]✓[/green] [bold][1/1][/bold]  "
                f"[white]{len(result.get('flow', []))} steps[/white]"
            )
            return result
        except ImageLimitError:
            auto_batch = max(1, len(frames) // 5)
            _warn(f"image limit hit — auto-batching at [bold]{auto_batch}[/bold] frames/request")
            return call_openrouter(api_key, model, frames, timeout, verbose, auto_batch, tracker, _depth + 1)

    # Batched: merge flow arrays across chunks
    all_steps: list[dict] = []
    step_offset = 1
    phase_context = "Initial Run"
    total_chunks = (len(frames) + batch_size - 1) // batch_size
    tracker.batches_total = total_chunks

    for i, chunk_start in enumerate(range(0, len(frames), batch_size)):
        chunk = frames[chunk_start:chunk_start + batch_size]
        is_first = i == 0
        label = f"[{i+1}/{total_chunks}]"
        if verbose:
            _dbg(f"[batch] chunk {i+1}: frames {chunk_start+1}–{chunk_start+len(chunk)} (step offset {step_offset})")

        prompt_text = PROMPT if is_first else PROMPT_CONTINUE.format(
            step_offset=step_offset,
            phase_context=phase_context,
        )
        content = [{"type": "text", "text": prompt_text}]
        content.extend(encode_frame(f) for f in chunk)

        try:
            body = _post(api_key, model, {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }, timeout, verbose, tracker, label=label, n_frames=len(chunk))
        except ImageLimitError:
            smaller = max(1, len(chunk) // 2)
            _warn(f"chunk {i+1} hit image limit — re-splitting to batch_size={smaller}")
            sub = call_openrouter(api_key, model, chunk, timeout, verbose, smaller, tracker, _depth + 1)
            for j, step in enumerate(sub.get("flow", [])):
                step["step"] = step_offset + j
            all_steps.extend(sub.get("flow", []))
            step_offset += len(sub.get("flow", []))
            if sub.get("flow"):
                phase_context = sub["flow"][-1].get("phase", phase_context)
            continue

        if verbose:
            raw_text = body["choices"][0]["message"]["content"] or ""
            _dbg(f"[batch] raw chunk {i+1}:\n{raw_text[:300]}")

        raw_text = body["choices"][0]["message"]["content"] or ""
        try:
            chunk_data = parse_json(raw_text)
        except json.JSONDecodeError as e:
            _warn(f"chunk {i+1} parse failed ({e}) — skipping")
            continue

        steps = chunk_data.get("flow", [])
        for j, step in enumerate(steps):
            step["step"] = step_offset + j
        all_steps.extend(steps)

        _console.print(
            f"  [green]✓[/green] [bold]{label}[/bold]  "
            f"frames [white]{chunk_start+1}–{chunk_start+len(chunk)}[/white]  "
            f"[dim]|[/dim]  [white]{len(steps)} steps[/white]"
        )

        step_offset += len(steps)
        if steps:
            phase_context = steps[-1].get("phase", phase_context)

    return {"flow": all_steps}


def parse_json(text: str | dict) -> dict:
    if isinstance(text, dict):
        return text
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


_PREVIEW_MAX_LINES = 24


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


_EXPENSIVE_THRESHOLD_USD_PER_1M = 1.00  # warn if input price > this


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
) -> int:
    global _live_models
    v = verbose

    if not shutil.which("ffmpeg"):
        _err("ffmpeg not found on PATH")
        return 2
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        _err("OPENROUTER_API_KEY missing (set in .env or export in shell)")
        return 2

    # Fetch live model pricing (cached 1h)
    from vidlizer.models import fetch_models
    with _console.status("[dim]fetching model pricing…[/dim]", spinner="dots2"):
        _live_models = fetch_models(api_key)

    # Money-bleeding guardrails
    if max_frames > 200:
        _warn(f"max_frames={max_frames} is high — runaway-cost risk. Capping to 200.")
        max_frames = 200
    inp_rate, out_rate = get_pricing(model, _live_models)
    if inp_rate > _EXPENSIVE_THRESHOLD_USD_PER_1M:
        _warn(f"[bold]{model}[/bold] is an expensive model "
              f"(${inp_rate:.2f}/M input, ${out_rate:.2f}/M output). "
              f"Cost cap: [bold]${max_cost:.2f}[/bold]")

    import contextlib

    is_image = video.suffix.lower() in _IMAGE_EXTS

    if is_image:
        _info(f"image input: [bold]{video.name}[/bold]  [dim]({video.stat().st_size // 1024} KB)[/dim]")
    else:
        from vidlizer.preflight import show_preflight
        show_preflight(video, model, min_interval, max_frames, _live_models)

    # nullcontext for images (file is permanent); TemporaryDirectory for videos
    tmp_ctx = contextlib.nullcontext(None) if is_image else tempfile.TemporaryDirectory(prefix="vidframes_")

    with tmp_ctx as tmp:
        if is_image:
            frames: list[Path] = [video]
        else:
            tmp_path = Path(tmp)  # type: ignore[arg-type]
            if v:
                _dbg(f"[debug] temp dir: {tmp_path}")
            frames = extract_frames(video, tmp_path, scale, max_frames, scene, fps, min_interval, v)
            if not frames:
                _err("no frames extracted — try --fps 0.5 or lower --scene threshold")
                return 1

        label = "image" if is_image else f"{len(frames)} frames"
        _info(
            f"[bold]{label}[/bold]  [dim]→[/dim]  [magenta]{model}[/magenta]  "
            f"[dim](batch={batch_size or 'auto'}, output: {output})[/dim]"
        )
        _console.print()

        def _try_model(m: str, trk: CostTracker) -> tuple[dict | None, int]:
            """Run up to 3 attempts. Returns (data, rc) where rc=-1 means exhausted."""
            for attempt in range(1, 4):
                try:
                    return call_openrouter(api_key, m, frames, timeout, v, batch_size, trk, is_image=is_image), 0
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
            return None, -1  # retries exhausted

        tracker = CostTracker(max_cost=max_cost)
        data, rc = _try_model(model, tracker)

        if data is None and rc == -1:
            # If a free model failed, auto-fallback to cheapest paid model
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

    output.write_text(json.dumps(data, indent=2, ensure_ascii=False))

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
