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
        "search_analysis accepts 'query' or 'keyword' parameter (both work). "
        "Track token and cost usage: get_usage_stats() returns per-model breakdown "
        "(runs, tokens in/out, cost USD). Reset with clear_usage_stats(). "
        "Tail logs: tail -f ~/.cache/vidlizer/mcp.log"
    ),
)


# ─── helpers ────────────────────────────────────────────────────────────────

import re as _re  # noqa: E402

_UNICODE_SPACES = _re.compile(
    r"[   -​  　﻿]"
)


def _normalize_spaces(s: str) -> str:
    return _UNICODE_SPACES.sub(" ", s)


def _resolve_local_path(path: str) -> str:
    """Return the real filesystem path for `path`, resolving Unicode space variants.

    macOS uses U+202F (NARROW NO-BREAK SPACE) in time-formatted filenames
    (e.g. "11-26-41 AM") while agents pass regular U+0020 spaces. This finds
    the matching entry by normalizing all Unicode spaces before comparing.
    """
    p = Path(path)
    if p.exists():
        return path
    parent = p.parent
    if not parent.exists():
        return path
    target = _normalize_spaces(p.name).lower()
    for entry in parent.iterdir():
        if _normalize_spaces(entry.name).lower() == target:
            return str(entry)
    return path  # no match — return original so ffmpeg gives the real error


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
    Analyze a video, image, or PDF and return structured JSON via the configured LLM.

    Call this tool first. It extracts frames, deduplicates near-identical ones, sends
    them to the LLM, and caches the result. Use get_summary/get_phase/get_steps to
    retrieve parts of the result without reloading everything (saves tokens). Repeated
    calls with the same path and parameters return the cached result instantly unless
    force_rerun=True.

    Supported inputs:
    - Local files: /absolute/path/to/video.mp4, image.png, document.pdf
    - URLs: YouTube, Vimeo, Loom, Twitter/X

    Provider and model are read from MCP server env vars (PROVIDER + OPENAI_MODEL /
    OLLAMA_MODEL / OPENROUTER_MODEL). Fallback models are tried automatically if the
    primary fails.

    Args:
        path: Local file path or URL to analyze.
        max_frames: Maximum frames to extract (default 60, hard cap 200).
        start: Start timestamp in seconds — analyze from this point onward.
        end: End timestamp in seconds — stop analysis here.
        scene_threshold: Scene-change sensitivity 0–1; lower = more frames captured (default 0.1).
        min_interval: Minimum seconds between consecutive frames (default 2.0).
        fps: Fixed frame rate extraction; overrides scene-change detection when set.
        scale: Frame resize width in pixels sent to the LLM (default 512).
        batch_size: Frames per LLM API call (0 = auto, resolves to 1).
        dedup_threshold: Perceptual hash Hamming distance for deduplication (0 = off, default 8).
        transcript: Transcribe audio via Apple MLX Whisper on macOS (default True).
        max_cost_usd: Abort if estimated spend exceeds this amount in USD (default 1.00).
        timeout: Per-request LLM timeout in seconds (default 600).
        force_rerun: Ignore cached result and re-analyze from scratch (default False).

    Returns:
        Dict with keys: analysis_id (str), file (str), steps (int), duration_s (float),
        model_used (str), provider_used (str), cached (bool), progress_log (list[str]),
        next_steps (str) — a suggested follow-up call. On error: {"error": str, "analysis_id": None}.
    """
    _provider = os.getenv("PROVIDER", "").lower()
    _model = _resolve_model(_provider, "")
    progress_log: list[str] = []

    def _log(msg: str) -> None:
        progress_log.append(msg)
        _logger.info(msg)

    try:
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
            return {**_meta(rec), "cached": True, "progress_log": progress_log}  # type: ignore[arg-type]

        _log(f"Starting analysis: {path}")
        _log(f"Provider: {_provider or 'from env'} | Model: {_model}")
        await ctx.report_progress(0, 100)

        # URL download
        url_ctx: contextlib.AbstractContextManager = contextlib.nullcontext(None)
        local_path = path

        from vidlizer.downloader import is_supported, is_url
        if not is_url(local_path):
            local_path = _resolve_local_path(local_path)
        if is_url(path):
            if not is_supported(path):
                return {
                    "error": "Unsupported URL. Allowed: YouTube, Vimeo, Loom, Twitter/X",
                    "analysis_id": None,
                    "progress_log": progress_log,
                }
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
            try:
                rc = run(
                    video=Path(local_path), output=tmp_out, model=_model, provider=_provider,
                    scene=scene_threshold, min_interval=min_interval, fps=fps, scale=scale,
                    max_frames=max_frames, batch_size=batch_size, timeout=timeout,
                    verbose=False, max_cost=max_cost_usd, start=start, end=end,
                    dedup_threshold=dedup_threshold, no_transcript=not transcript,
                    output_format="json",
                )
            except Exception as exc:
                _logger.exception("core.run() raised: %s", exc)
                tmp_out.unlink(missing_ok=True)
                return 1, {}
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
        result = {**_meta(rec), "cached": False, "progress_log": progress_log}  # type: ignore[arg-type]
        result["model_used"] = data.get("model_used", _model)
        result["provider_used"] = data.get("provider_used", _provider)
        result["next_steps"] = (
            f"Call get_summary('{aid}') for an overview, "
            f"get_phase('{aid}', '<phase>') for a section, "
            f"or search_analysis('{aid}', '<keyword>') to find specific moments."
        )
        return result

    except Exception as exc:
        _logger.exception("analyze_video failed")
        return {"error": str(exc), "analysis_id": None, "progress_log": progress_log}


@app.tool()
def list_analyses() -> list[dict]:
    """
    List all analyses stored in the local vidlizer cache.

    Use this to discover existing analysis_ids without re-running analyze_video.
    Returns lightweight metadata only — no flow steps or transcript data, so it
    is safe to call even when many analyses are cached.

    Returns:
        List of dicts, each with: analysis_id (str), file (str), steps (int),
        duration_s (float), created_at (str ISO timestamp). Empty list if no analyses exist.
    """
    return store.list_all()


@app.tool()
def get_summary(analysis_id: str, level: str = "medium") -> str:
    """
    Get a human-readable text summary of a stored analysis.

    Use this immediately after analyze_video to get an overview before drilling into
    individual steps or phases. Prefer 'brief' for quick answers; 'full' for detailed
    step-by-step breakdowns.

    Args:
        analysis_id: ID returned by analyze_video or list_analyses.
        level: Summary verbosity — 'brief' (1 paragraph), 'medium' (phase groups, default),
               'full' (one sentence per step).

    Returns:
        Formatted text string. Returns an error string if analysis_id is not found.
    """
    rec = store.load(analysis_id)
    if not rec:
        return f"Analysis {analysis_id!r} not found. Run analyze_video first."
    return _summary_text(rec["data"], level=level)


@app.tool()
def get_step(analysis_id: str, step: int, full: bool = False) -> dict:
    """
    Get a single flow step by its step number.

    Use when the user asks about a specific moment in the video by number.
    For ranges, use get_steps. For named sections, use get_phase.

    Args:
        analysis_id: ID from analyze_video or list_analyses.
        step: Step number (1-indexed, as returned in the flow array).
        full: Return all 11 LLM-extracted fields (default False returns core fields:
              step, timestamp_s, phase, action, scene, speech).

    Returns:
        Dict of step fields. Returns {"error": str} if analysis_id or step not found.
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
    Get a consecutive range of flow steps from a stored analysis.

    Use when the user asks about a span of the video (e.g., "steps 5 to 10").
    For a single step, use get_step. For a named phase, use get_phase.

    Args:
        analysis_id: ID from analyze_video or list_analyses.
        start_step: First step to return, inclusive and 1-indexed (default 1).
        end_step: Last step to return, inclusive. Defaults to the final step.
        full: Return all 11 LLM-extracted fields (default False returns core fields only).

    Returns:
        List of step dicts in ascending step order. Returns a single-element list with
        {"error": str} if analysis_id is not found.
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
    Get all flow steps belonging to a named phase of the analysis.

    Phases are high-level sections the LLM assigns to groups of steps (e.g.,
    'Introduction', 'Demo', 'Checkout'). Use get_summary(level='medium') first
    to discover what phase names exist, then call this tool for details.

    Args:
        analysis_id: ID from analyze_video or list_analyses.
        phase: Phase name to filter by — case-insensitive substring match
               (e.g. 'intro', 'Introduction', 'DEMO' all match 'Introduction').
        full: Return all 11 LLM-extracted fields (default False returns core fields only).

    Returns:
        List of step dicts for the matching phase. Empty list if phase name not found.
        Returns single-element list with {"error": str} if analysis_id is not found.
    """
    rec = store.load(analysis_id)
    if not rec:
        return [{"error": f"Analysis {analysis_id!r} not found"}]
    flow = rec["data"].get("flow", [])
    return [_slim(s, full) for s in flow if s.get("phase", "").lower() == phase.lower()]


@app.tool()
def search_analysis(
    analysis_id: str,
    query: str | None = None,
    keyword: str | None = None,
    fields: list[str] | None = None,
) -> list[dict]:
    """
    Search flow steps for text matching a keyword or phrase (case-insensitive substring).

    Use when the user asks "find where X appears" or "which steps mention Y".
    Returns only the steps that match, with the matched field and value highlighted.
    Either query or keyword must be provided; they are interchangeable.

    Args:
        analysis_id: ID from analyze_video or list_analyses.
        query: Text to search for across flow step fields.
        keyword: Alias for query — use either, not both.
        fields: List of step fields to search. Defaults to:
                ['scene', 'action', 'text_visible', 'speech', 'observations'].

    Returns:
        List of match dicts with: step (int), timestamp_s (float), phase (str),
        matched_field (str), matched_value (str), action (str).
        Empty list if no steps match. Returns [{"error": str}] if analysis_id not found
        or if neither query nor keyword is provided.
    """
    search_term = query or keyword
    if not search_term:
        return [{"error": "query or keyword is required"}]
    rec = store.load(analysis_id)
    if not rec:
        return [{"error": f"Analysis {analysis_id!r} not found"}]
    flow = rec["data"].get("flow", [])
    search_in = fields or ["scene", "action", "text_visible", "speech", "observations"]
    q = search_term.lower()
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
    Get audio transcript segments from a stored analysis, optionally filtered by time range.

    Transcription is performed via Apple MLX Whisper on macOS (when transcript=True in
    analyze_video). Returns an empty list if no transcript was captured.
    Use start_s/end_s to retrieve only the portion relevant to a specific video segment.

    Args:
        analysis_id: ID from analyze_video or list_analyses.
        start_s: Return segments that overlap or start at/after this timestamp (seconds).
        end_s: Return segments that overlap or end at/before this timestamp (seconds).

    Returns:
        List of segment dicts with: start (float), end (float), text (str).
        Empty list if no transcript exists or no segments fall in the given range.
        Returns [{"error": str}] if analysis_id is not found.
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
    Get the complete raw JSON output of a stored analysis (all flow steps + transcript).

    Avoid this unless you genuinely need everything — output can be very large for long
    videos (hundreds of steps). Prefer get_summary for overviews, get_phase/get_steps
    for sections, or search_analysis for targeted lookups.

    Args:
        analysis_id: ID from analyze_video or list_analyses.

    Returns:
        Dict with keys: flow (list of all step dicts), transcript (list of segment dicts),
        metadata (dict with file, model, provider, duration). Returns {"error": str}
        if analysis_id is not found.
    """
    rec = store.load(analysis_id)
    if not rec:
        return {"error": f"Analysis {analysis_id!r} not found"}
    return rec["data"]


@app.tool()
def delete_analysis(analysis_id: str) -> dict:
    """
    Delete a stored analysis from the local vidlizer cache.

    Use when an analysis is no longer needed to free disk space.
    This operation is irreversible — the analysis must be re-run to recover the data.

    Args:
        analysis_id: ID from analyze_video or list_analyses.

    Returns:
        Dict with: deleted (bool — True if found and removed, False if not found),
        analysis_id (str — echoes the input ID).
    """
    ok = store.delete(analysis_id)
    return {"deleted": ok, "analysis_id": analysis_id}


@app.tool()
def get_usage_stats() -> dict:
    """
    Return cumulative token and cost usage statistics across all vidlizer runs.

    Use to monitor spending, compare model efficiency, or audit usage before
    running an expensive analysis. Covers both CLI and MCP-initiated runs.

    Returns:
        Dict with: total_runs (int), total_cost_usd (float), total_tokens (int),
        by_model (list of dicts sorted by uses desc, each with: model, provider,
        runs, tokens, cost_usd).
    """
    from vidlizer.usage import get_stats
    return get_stats()


@app.tool()
def clear_usage_stats() -> dict:
    """
    Delete the usage log and reset all cost/token counters to zero.

    Use at the start of a new project or billing period to get clean tracking.
    This is irreversible — historical usage data cannot be recovered after clearing.

    Returns:
        Dict with: cleared (bool — always True), records_deleted (int — number of
        log entries removed).
    """
    from vidlizer.usage import clear_stats
    deleted = clear_stats()
    return {"cleared": True, "records_deleted": deleted}


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
