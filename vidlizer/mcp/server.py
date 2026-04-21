"""MCP server for vidlizer — analyze any video/image/PDF via any MCP-compatible agent."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from vidlizer.mcp import store

_LOG_PATH = Path.home() / ".cache" / "vidlizer" / "mcp.log"

_logger = logging.getLogger("vidlizer.mcp")


def _setup_logging() -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    # Redirect stderr (Rich console from core.py) to same log file
    sys.stderr = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)  # noqa: SIM115


app = FastMCP(
    "vidlizer",
    instructions=(
        "Analyze videos, images, and PDFs into structured JSON timelines. "
        "Supports local files and URLs (YouTube, Vimeo, Loom, Twitter/X). "
        "Workflow: call analyze_video → get analysis_id → use get_summary/get_step/"
        "get_phase/search_analysis to pull specific parts without loading everything. "
        "Full data via get_full_analysis (use sparingly — can be large). "
        "Tail logs: tail -f ~/.cache/vidlizer/mcp.log"
    ),
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _resolve_model(provider: str, model: str) -> str:
    if model:
        return model
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "")
    return os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")


def _meta(rec: dict) -> dict:
    return {
        "analysis_id": rec["id"],
        "source": rec.get("source", ""),
        "step_count": rec.get("step_count", 0),
        "phases": rec.get("phases", []),
        "has_transcript": rec.get("has_transcript", False),
        "duration_s": rec.get("duration_s"),
        "created_at": rec.get("created_at", 0),
    }


_CORE_FIELDS = ("step", "timestamp_s", "phase", "action", "scene", "speech")


def _slim(step: dict, full: bool) -> dict:
    return step if full else {k: step[k] for k in _CORE_FIELDS if k in step}


def _summary_text(data: dict, level: str = "medium") -> str:
    flow = data.get("flow", [])
    if not flow:
        return "No steps found."

    if level == "brief":
        phases: dict[str, list[str]] = {}
        for s in flow:
            phases.setdefault(s.get("phase", "?"), []).append(s.get("action", ""))
        return " | ".join(
            f"{ph}: {next((a for a in acts if a), '')}"
            for ph, acts in phases.items()
        )

    if level == "full":
        return "\n".join(
            f"[{s.get('timestamp_s', '?')}s] step {s['step']}: "
            f"{s.get('action', '')} — {s.get('scene', '')}"
            for s in flow
        )

    # medium: group by phase, up to 5 steps shown per phase
    by_phase: dict[str, list[dict]] = {}
    for s in flow:
        by_phase.setdefault(s.get("phase", "Unknown"), []).append(s)

    lines = []
    for ph, steps in by_phase.items():
        lines.append(f"\n## {ph}")
        for s in steps[:5]:
            ts = s.get("timestamp_s")
            ts_str = f"[{ts:.0f}s] " if ts is not None else ""
            action = s.get("action", "")
            scene = s.get("scene", "")
            speech = s.get("speech", "")
            line = f"- {ts_str}{action}"
            if scene and scene != action:
                line += f" ({scene[:60]})"
            if speech:
                line += f'\n  > "{speech[:80]}"'
            lines.append(line)
        if len(steps) > 5:
            lines.append(f"  … {len(steps) - 5} more steps")

    transcript = data.get("transcript", [])
    if transcript:
        lines.append(f"\n_Transcript: {len(transcript)} segments_")

    return "\n".join(lines)


# ─── tools ──────────────────────────────────────────────────────────────────

@app.tool()
async def analyze_video(
    ctx: Context,
    path: str,
    provider: str = "",
    model: str = "",
    max_frames: int = 60,
    start: float | None = None,
    end: float | None = None,
    scene_threshold: float = 0.1,
    min_interval: float = 2.0,
    fps: float | None = None,
    scale: int = 512,
    batch_size: int = 0,
    dedup_threshold: int = 8,
    transcript: bool = True,
    max_cost_usd: float = 1.0,
    timeout: int = 600,
    force_rerun: bool = False,
) -> dict:
    """
    Analyze a video, image, or PDF. Returns analysis_id + metadata.

    Supported inputs:
    - Local files: /absolute/path/to/video.mp4, image.png, document.pdf
    - URLs: YouTube, Vimeo, Loom, Twitter/X

    After analysis, use get_summary/get_step/get_phase to retrieve parts
    on demand instead of loading everything at once (saves tokens).

    Args:
        path: Local file path or URL
        provider: 'ollama' (local, free) or 'openrouter' (cloud, API key needed).
                  Defaults to PROVIDER env var.
        model: Model ID — Ollama name or OpenRouter slug (e.g. google/gemini-2.5-flash).
               Defaults to OLLAMA_MODEL / OPENROUTER_MODEL env vars.
        max_frames: Max frames to extract (default 60, hard cap 200)
        start: Analyze from this timestamp in seconds
        end: Analyze up to this timestamp in seconds
        scene_threshold: Scene-change sensitivity 0–1 (lower = more frames, default 0.1)
        min_interval: Min seconds between frames (default 2.0)
        fps: Fixed FPS extraction — overrides scene-change detection
        scale: Frame width in pixels (default 512)
        batch_size: Frames per API call (0=auto; Ollama always uses 1)
        dedup_threshold: Perceptual dedup Hamming distance (0=off, default 8)
        transcript: Auto-transcribe audio via Apple MLX Whisper (default True)
        max_cost_usd: Abort if spend exceeds this in USD (default 1.00)
        timeout: Per-request timeout in seconds (default 600)
        force_rerun: Re-analyze even if cached result exists (default False)
    """
    _provider = (provider or os.getenv("PROVIDER", "")).lower()
    _model = _resolve_model(_provider, model)
    progress_log: list[str] = []

    def _log(msg: str) -> None:
        progress_log.append(msg)
        _logger.info(msg)

    params = {
        "provider": _provider, "model": _model, "max_frames": max_frames,
        "start": start, "end": end, "scene_threshold": scene_threshold,
        "min_interval": min_interval, "fps": fps, "scale": scale,
        "batch_size": batch_size, "dedup_threshold": dedup_threshold,
        "transcript": transcript,
    }
    aid = store.make_id(path, params)

    if not force_rerun and store.exists(aid):
        _log(f"Cache hit: {aid}")
        rec = store.load(aid)
        return {**_meta(rec), "cached": True, "progress_log": progress_log}

    _log(f"Starting analysis: {path}")
    _log(f"Provider: {_provider or 'from env'} | Model: {_model}")
    await ctx.report_progress(0, 100)

    # URL download
    url_ctx: contextlib.AbstractContextManager = contextlib.nullcontext(None)
    local_path = path

    from vidlizer.downloader import is_url
    if is_url(path):
        from vidlizer.downloader import download
        _log("Downloading video from URL…")
        await ctx.report_progress(10, 100)
        url_tmp = tempfile.TemporaryDirectory(prefix="vidlizer_dl_")
        url_ctx = url_tmp
        try:
            local_path = str(await asyncio.to_thread(download, path, Path(url_tmp.name)))
            _log(f"Downloaded: {Path(local_path).name}")
        except Exception as e:
            _log(f"Download failed: {e}")
            return {"error": f"Download failed: {e}", "analysis_id": None, "progress_log": progress_log}

    _log(f"Extracting frames (max {max_frames}) and sending to {_model}…")
    await ctx.report_progress(20, 100)

    def _run_analysis() -> tuple[int, dict]:
        from vidlizer.core import run
        if _provider:
            os.environ["PROVIDER"] = _provider
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_out = Path(f.name)
        rc = run(
            video=Path(local_path), output=tmp_out, model=_model, provider=_provider,
            scene=scene_threshold, min_interval=min_interval, fps=fps, scale=scale,
            max_frames=max_frames, batch_size=batch_size, timeout=timeout,
            verbose=False, max_cost=max_cost_usd, start=start, end=end,
            dedup_threshold=dedup_threshold, no_transcript=not transcript,
            output_format="json",
        )
        data: dict = {}
        if tmp_out.exists():
            try:
                data = json.loads(tmp_out.read_text())
            except Exception:
                pass
            tmp_out.unlink(missing_ok=True)
        return rc, data

    with url_ctx:
        await ctx.report_progress(25, 100)
        rc, data = await asyncio.to_thread(_run_analysis)

    await ctx.report_progress(90, 100)

    if rc != 0 or not data:
        _log(f"Analysis failed (exit code {rc})")
        return {"error": f"Analysis failed (exit code {rc})", "analysis_id": None, "progress_log": progress_log}

    steps = len(data.get("flow", []))
    _log(f"Analysis complete: {steps} steps extracted")
    if data.get("transcript"):
        _log(f"Transcript: {len(data['transcript'])} segments")

    store.save(aid, path, params, data)
    _log(f"Saved as analysis_id={aid}")
    await ctx.report_progress(100, 100)

    rec = store.load(aid)
    result = {**_meta(rec), "cached": False, "progress_log": progress_log}
    result["next_steps"] = (
        f"Call get_summary('{aid}') for an overview, "
        f"get_phase('{aid}', '<phase>') for a section, "
        f"or search_analysis('{aid}', '<keyword>') to find specific moments."
    )
    return result


@app.tool()
def list_analyses() -> list[dict]:
    """List all stored analyses. Returns lightweight meta (no data payloads)."""
    return store.list_all()


@app.tool()
def get_summary(analysis_id: str, level: str = "medium") -> str:
    """
    Get a text summary of an analysis.

    Args:
        analysis_id: ID returned by analyze_video or list_analyses
        level: 'brief' (1 paragraph), 'medium' (phase groups, default), 'full' (one-liner per step)
    """
    rec = store.load(analysis_id)
    if not rec:
        return f"Analysis {analysis_id!r} not found. Run analyze_video first."
    return _summary_text(rec["data"], level=level)


@app.tool()
def get_step(analysis_id: str, step: int, full: bool = False) -> dict:
    """
    Get a single flow step by number.

    Args:
        analysis_id: ID from analyze_video
        step: Step number (1-indexed)
        full: Return all 11 fields. Default: core fields only (step, timestamp_s, phase, action, scene, speech).
    """
    rec = store.load(analysis_id)
    if not rec:
        return {"error": f"Analysis {analysis_id!r} not found"}
    flow = rec["data"].get("flow", [])
    for s in flow:
        if s.get("step") == step:
            return _slim(s, full)
    return {"error": f"Step {step} not found (total steps: {len(flow)})"}


@app.tool()
def get_steps(
    analysis_id: str,
    start_step: int = 1,
    end_step: int | None = None,
    full: bool = False,
) -> list[dict]:
    """
    Get a range of flow steps.

    Args:
        analysis_id: ID from analyze_video
        start_step: First step (inclusive, 1-indexed, default 1)
        end_step: Last step (inclusive). Defaults to last step.
        full: Return all fields (default: core fields only)
    """
    rec = store.load(analysis_id)
    if not rec:
        return [{"error": f"Analysis {analysis_id!r} not found"}]
    flow = rec["data"].get("flow", [])
    last = end_step or len(flow)
    return [_slim(s, full) for s in flow if start_step <= s.get("step", 0) <= last]


@app.tool()
def get_phase(analysis_id: str, phase: str, full: bool = False) -> list[dict]:
    """
    Get all flow steps in a named phase.

    Args:
        analysis_id: ID from analyze_video
        phase: Phase name (e.g. 'Introduction', 'Demo') — case-insensitive
        full: Return all fields (default: core fields only)
    """
    rec = store.load(analysis_id)
    if not rec:
        return [{"error": f"Analysis {analysis_id!r} not found"}]
    flow = rec["data"].get("flow", [])
    return [_slim(s, full) for s in flow if s.get("phase", "").lower() == phase.lower()]


@app.tool()
def search_analysis(
    analysis_id: str,
    query: str,
    fields: list[str] | None = None,
) -> list[dict]:
    """
    Search flow steps for matching text (case-insensitive substring).

    Args:
        analysis_id: ID from analyze_video
        query: Text to search for
        fields: Fields to search. Default: scene, action, text_visible, speech, observations
    """
    rec = store.load(analysis_id)
    if not rec:
        return [{"error": f"Analysis {analysis_id!r} not found"}]
    flow = rec["data"].get("flow", [])
    search_in = fields or ["scene", "action", "text_visible", "speech", "observations"]
    q = query.lower()
    results = []
    for s in flow:
        for field in search_in:
            if q in str(s.get(field, "")).lower():
                results.append({
                    "step": s.get("step"),
                    "timestamp_s": s.get("timestamp_s"),
                    "phase": s.get("phase"),
                    "matched_field": field,
                    "matched_value": s.get(field),
                    "action": s.get("action"),
                })
                break
    return results


@app.tool()
def get_transcript(
    analysis_id: str,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[dict]:
    """
    Get transcript segments, optionally filtered by time range.

    Args:
        analysis_id: ID from analyze_video
        start_s: Start time in seconds (optional)
        end_s: End time in seconds (optional)
    """
    rec = store.load(analysis_id)
    if not rec:
        return [{"error": f"Analysis {analysis_id!r} not found"}]
    segments = rec["data"].get("transcript", [])
    if start_s is None and end_s is None:
        return segments
    return [
        seg for seg in segments
        if (start_s is None or seg.get("end", 0) >= start_s)
        and (end_s is None or seg.get("start", 0) <= end_s)
    ]


@app.tool()
def get_full_analysis(analysis_id: str) -> dict:
    """
    Get the complete raw analysis data (flow array + transcript).

    Use sparingly — output can be large. Prefer get_summary/get_phase/get_steps
    to retrieve only what you need.

    Args:
        analysis_id: ID from analyze_video
    """
    rec = store.load(analysis_id)
    if not rec:
        return {"error": f"Analysis {analysis_id!r} not found"}
    return rec["data"]


@app.tool()
def delete_analysis(analysis_id: str) -> dict:
    """
    Delete a stored analysis.

    Args:
        analysis_id: ID from analyze_video or list_analyses
    """
    ok = store.delete(analysis_id)
    return {"deleted": ok, "analysis_id": analysis_id}


# ─── resources ──────────────────────────────────────────────────────────────

@app.resource("vidlizer://analyses")
def resource_list() -> str:
    """All stored analyses as JSON (meta only, no data payloads)."""
    return json.dumps(store.list_all(), indent=2)


@app.resource("vidlizer://analyses/{analysis_id}")
def resource_full(analysis_id: str) -> str:
    """Full analysis data for a given id."""
    rec = store.load(analysis_id)
    if not rec:
        return json.dumps({"error": f"Not found: {analysis_id}"})
    return json.dumps(rec["data"], indent=2)


@app.resource("vidlizer://analyses/{analysis_id}/summary")
def resource_summary(analysis_id: str) -> str:
    """Medium-level text summary for a given analysis."""
    rec = store.load(analysis_id)
    if not rec:
        return f"Analysis {analysis_id!r} not found."
    return _summary_text(rec["data"], level="medium")


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    _setup_logging()
    _logger.info("vidlizer-mcp server starting")
    app.run()


if __name__ == "__main__":
    main()
