"""Batched frame submission and JSON merging for multi-chunk analysis."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console

from vidlizer.frames import encode_frame
from vidlizer.http import CostCapExceeded, CostTracker, ImageLimitError, post

_console = Console(stderr=True, highlight=False)

_MAX_RECURSION_DEPTH = 4

PROMPT = """Role: Act as an expert Video Analyst.

Task: Analyze the provided frames and map the complete sequence of events into a single exhaustive JSON object. This applies to ANY type of video: software/app recordings, tutorials, interviews, product demos, marketing content, physical activities, presentations, or anything else.

Observational Rules:
- Zero Compression: Every meaningful change between frames is a step. Do not skip.
- Adapt to Content: Infer the video type and use appropriate terminology — UI interaction, physical action, narrative event, speech, etc.
- Precision: Only describe what is visually evident. Note uncertainty where present.
- Text: Capture all visible text — UI labels, captions, subtitles, titles, overlays, on-screen code.
- Issues: Flag errors, anomalies, quality problems, or anything unexpected.
- Persistence: Track any ongoing state — timer, score, progress bar, speaker identity, topic, brand, etc.

Frames are labelled [t=Xs] where X is the timestamp in seconds. Use these to fill timestamp_s.

JSON Schema:
Produce a single JSON object with a `flow` array. Each element MUST include:
- step: Sequential integer.
- timestamp_s: Approximate time in seconds when this step begins (from the nearest [t=Xs] label). null if no labels present.
- phase: Logical section (e.g. "Introduction", "Demo", "Action", "Conclusion", "Navigation", "Dialogue").
- scene: What is currently visible — the setting, screen, environment, or context.
- subjects: (JSON array) Key people, objects, UI elements, or entities present.
- action: What is happening — interaction, physical movement, narration, transition, or event.
- text_visible: All readable text on screen (UI labels, captions, code, titles, overlays). Empty string if none.
- context: Persistent state or background information relevant to this moment.
- observations: Notable details — emotions, errors, quality issues, key facts, or anomalies.
- next_scene: Brief description of what follows.

Output: Provide ONLY a valid JSON object with the `flow` array. No prose, no code fences. 100% accuracy based on visual evidence."""

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
- subjects: (JSON array) All people, objects, UI elements, or entities present.
- action: Any activity, state, or interaction evident in the image.
- text_visible: All readable text (labels, captions, code, titles, overlays). Empty string if none.
- context: Inferred purpose, brand, topic, or background information.
- observations: Notable details — quality, errors, emotions, anomalies, key facts.
- next_scene: null

Output: Provide ONLY a valid JSON object with the `flow` array containing exactly one step. No prose, no code fences."""

PROMPT_REPAIR = (
    "The following is malformed JSON from a video analysis. "
    "Return ONLY the corrected JSON object with a `flow` array. "
    "No prose, no code fences, no explanations:\n\n{broken_text}"
)


def parse_json(text: str | dict) -> dict:
    if isinstance(text, dict):
        return text
    text = text.strip()
    if not text:
        raise json.JSONDecodeError(
            "model returned empty response — "
            "model may not support vision or context window exceeded",
            "", 0,
        )
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL).strip()
    if not text:
        raise json.JSONDecodeError("model returned only thinking content, no JSON found", "", 0)
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    parsed = json.loads(text.strip())
    result = {"flow": parsed} if isinstance(parsed, list) else parsed
    for step in result.get("flow", []):
        if "subjects" in step and not isinstance(step["subjects"], list):
            s = step["subjects"]
            step["subjects"] = [s] if s else []
    return result


def merge_transcript(flow: list[dict], segments: list[dict]) -> None:
    """Inject 'speech' into flow steps. Each segment goes to exactly one step."""
    if not flow or not segments:
        return
    n = len(flow)

    windows: list[tuple[float, float]] = []
    for i, step in enumerate(flow):
        try:
            t_start = float(step.get("timestamp_s") or 0.0)
        except (TypeError, ValueError):
            t_start = 0.0
        try:
            t_end = float(flow[i + 1].get("timestamp_s") or float("inf")) if i + 1 < n else float("inf")
        except (TypeError, ValueError):
            t_end = float("inf")
        windows.append((t_start, t_end))

    buckets: list[list[str]] = [[] for _ in flow]
    for seg in segments:
        seg_start = seg["start"]
        assigned = False
        for i, (t_start, t_end) in enumerate(windows):
            if t_start <= seg_start < t_end:
                buckets[i].append(seg["text"])
                assigned = True
                break
        if not assigned and buckets:
            buckets[0].append(seg["text"])

    for step, parts in zip(flow, buckets):
        if parts:
            step["speech"] = " ".join(parts).strip()


def call_model(
    api_key: str | None,
    model: str,
    frames: list[Path],
    timeout: int,
    verbose: bool,
    batch_size: int,
    tracker: CostTracker | None = None,
    _depth: int = 0,
    is_image: bool = False,
    timestamps: list[float] | None = None,
    endpoint: str | None = None,
    req_headers: dict | None = None,
    is_ollama: bool = False,
    no_stream_opts: bool = False,
    no_json_format: bool = False,
    concurrency: int = 1,
) -> dict:
    """Send frames to model. Auto-retries with batching on image-limit errors.

    concurrency > 1 dispatches chunks in parallel (OpenRouter only — local providers
    serialize anyway). Parallel mode uses PROMPT for every chunk (no phase context
    chain); slight cost-cap overshoot is accepted since in-flight requests complete
    before the cap is rechecked.
    """
    if _depth > _MAX_RECURSION_DEPTH:
        raise RuntimeError(
            f"batching recursion depth {_depth} exceeded — "
            f"model rejects {len(frames)} frame(s) even at batch_size={batch_size}"
        )
    if tracker is None:
        tracker = CostTracker()

    def _build_content(prompt: str, chunk: list[Path], ts: list[float] | None) -> list[dict]:
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for i, f in enumerate(chunk):
            if ts and i < len(ts):
                parts.append({"type": "text", "text": f"[t={ts[i]:.1f}s]"})
            parts.append(encode_frame(f))
        return parts

    def _make_payload(prompt: str, chunk: list[Path], ts: list[float] | None) -> dict:
        p: dict = {
            "model": model,
            "messages": [{"role": "user", "content": _build_content(prompt, chunk, ts)}],
            "temperature": 0.1,
        }
        if not is_ollama and not no_json_format:
            p["response_format"] = {"type": "json_object"}
        return p

    def _post_chunk(label: str, payload: dict, chunk: list[Path]) -> dict:
        return post(api_key, model, payload, timeout, verbose, tracker,
                    label=label, n_frames=len(chunk),
                    endpoint=endpoint, req_headers=req_headers,
                    is_ollama=is_ollama, no_stream_opts=no_stream_opts)

    def _repair_json(broken_text: str, chunk_label: str) -> dict:
        """One retry: ask model to fix broken JSON. Raises JSONDecodeError on failure."""
        _console.print(f"[yellow]⚠[/yellow]  {chunk_label} parse failed — asking model to repair")
        repair_payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_REPAIR.format(broken_text=broken_text)}],
            "temperature": 0.0,
        }
        if not is_ollama and not no_json_format:
            repair_payload["response_format"] = {"type": "json_object"}
        repair_body = post(api_key, model, repair_payload, timeout, verbose, tracker,
                           label=f"{chunk_label}[repair]", n_frames=0,
                           endpoint=endpoint, req_headers=req_headers,
                           is_ollama=is_ollama, no_stream_opts=no_stream_opts)
        repair_text = repair_body["choices"][0]["message"]["content"] or ""
        return parse_json(repair_text)

    # ── Single request (no batching needed) ──────────────────────────────────
    if batch_size <= 0 or len(frames) <= batch_size:
        tracker.batches_total = 1
        payload = _make_payload(PROMPT_IMAGE if is_image else PROMPT, frames, timestamps)
        try:
            body = _post_chunk("[1/1]", payload, frames)
            result = parse_json(body["choices"][0]["message"]["content"])
            _console.print(
                f"  [green]✓[/green] [bold][1/1][/bold]  "
                f"[white]{len(result.get('flow', []))} steps[/white]"
            )
            return result
        except ImageLimitError:
            auto_batch = max(1, len(frames) // 5)
            _console.print(f"[yellow]⚠[/yellow]  image limit hit — auto-batching at [bold]{auto_batch}[/bold] frames/request")
            return call_model(api_key, model, frames, timeout, verbose, auto_batch, tracker, _depth + 1,
                              timestamps=timestamps, endpoint=endpoint, req_headers=req_headers,
                              is_ollama=is_ollama, no_stream_opts=no_stream_opts,
                              no_json_format=no_json_format, concurrency=concurrency)

    # ── Build chunk list ──────────────────────────────────────────────────────
    total_chunks = (len(frames) + batch_size - 1) // batch_size
    tracker.batches_total = total_chunks

    chunk_specs = []
    for i, chunk_start in enumerate(range(0, len(frames), batch_size)):
        chunk = frames[chunk_start:chunk_start + batch_size]
        chunk_ts = timestamps[chunk_start:chunk_start + batch_size] if timestamps else None
        label = f"[{i+1}/{total_chunks}]"
        chunk_specs.append((i, chunk_start, chunk, chunk_ts, label))

    # ── Parallel path ─────────────────────────────────────────────────────────
    if concurrency > 1 and total_chunks > 1:
        _console.print(f"[dim]parallel: {total_chunks} chunks × {concurrency} workers[/dim]")

        def _run_chunk(spec: tuple) -> tuple[int, int, list[Path], list[float] | None, str, dict]:
            i, chunk_start, chunk, chunk_ts, label = spec
            payload = _make_payload(PROMPT, chunk, chunk_ts)
            body = _post_chunk(label, payload, chunk)
            return i, chunk_start, chunk, chunk_ts, label, body

        raw_results: dict[int, tuple] = {}
        had_image_limit = False

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_run_chunk, spec): spec for spec in chunk_specs}
            for future in as_completed(futures):
                try:
                    i, chunk_start, chunk, chunk_ts, label, body = future.result()
                    raw_results[i] = (chunk_start, chunk, chunk_ts, label, body)
                except ImageLimitError:
                    had_image_limit = True
                    # Cancel remaining (best-effort) and fall through to serial retry
                    for f in futures:
                        f.cancel()
                    break

        if had_image_limit:
            smaller = max(1, batch_size // 2)
            _console.print(f"[yellow]⚠[/yellow]  image limit — retrying serial batch_size={smaller}")
            return call_model(api_key, model, frames, timeout, verbose, smaller, tracker, _depth + 1,
                              timestamps=timestamps, endpoint=endpoint, req_headers=req_headers,
                              is_ollama=is_ollama, no_stream_opts=no_stream_opts,
                              no_json_format=no_json_format, concurrency=1)

        all_steps: list[dict] = []
        step_offset = 1
        for i in sorted(raw_results):
            chunk_start, chunk, chunk_ts, label, body = raw_results[i]
            if verbose:
                raw_text = body["choices"][0]["message"]["content"] or ""
                _console.print(f"[dim][batch] raw chunk {i+1}:\n{raw_text[:300]}[/dim]", markup=False)
            raw_text = body["choices"][0]["message"]["content"] or ""
            try:
                chunk_data = parse_json(raw_text)
            except json.JSONDecodeError:
                try:
                    chunk_data = _repair_json(raw_text, label)
                except CostCapExceeded:
                    raise
                except (json.JSONDecodeError, Exception) as repair_err:
                    _console.print(f"[yellow]⚠[/yellow]  {label} repair failed ({repair_err}) — skipping")
                    continue
            steps = chunk_data.get("flow", [])
            for j, step in enumerate(steps):
                step["step"] = step_offset + j
            all_steps.extend(steps)
            step_offset += len(steps)
            _console.print(
                f"  [green]✓[/green] [bold]{label}[/bold]  "
                f"frames [white]{chunk_start+1}–{chunk_start+len(chunk)}[/white]  "
                f"[dim]|[/dim]  [white]{len(steps)} steps[/white]"
            )
        return {"flow": all_steps}

    # ── Serial path ───────────────────────────────────────────────────────────
    all_steps = []
    step_offset = 1
    phase_context = "Initial Run"

    for i, chunk_start, chunk, chunk_ts, label in chunk_specs:
        is_first = i == 0
        if verbose:
            _console.print(f"[dim][batch] chunk {i+1}: frames {chunk_start+1}–{chunk_start+len(chunk)} (step offset {step_offset})[/dim]", markup=False)

        prompt_text = PROMPT if is_first else PROMPT_CONTINUE.format(
            step_offset=step_offset, phase_context=phase_context,
        )
        payload = _make_payload(prompt_text, chunk, chunk_ts)
        try:
            body = _post_chunk(label, payload, chunk)
        except ImageLimitError:
            smaller = max(1, len(chunk) // 2)
            _console.print(f"[yellow]⚠[/yellow]  chunk {i+1} hit image limit — re-splitting to batch_size={smaller}")
            sub = call_model(api_key, model, chunk, timeout, verbose, smaller, tracker, _depth + 1,
                             timestamps=chunk_ts, endpoint=endpoint, req_headers=req_headers,
                             is_ollama=is_ollama, no_stream_opts=no_stream_opts,
                             no_json_format=no_json_format, concurrency=1)
            for j, step in enumerate(sub.get("flow", [])):
                step["step"] = step_offset + j
            all_steps.extend(sub.get("flow", []))
            step_offset += len(sub.get("flow", []))
            if sub.get("flow"):
                phase_context = sub["flow"][-1].get("phase", phase_context)
            continue

        if verbose:
            raw_text = body["choices"][0]["message"]["content"] or ""
            _console.print(f"[dim][batch] raw chunk {i+1}:\n{raw_text[:300]}[/dim]", markup=False)

        raw_text = body["choices"][0]["message"]["content"] or ""
        try:
            chunk_data = parse_json(raw_text)
        except json.JSONDecodeError:
            try:
                chunk_data = _repair_json(raw_text, label)
            except CostCapExceeded:
                raise
            except (json.JSONDecodeError, Exception) as repair_err:
                _console.print(f"[yellow]⚠[/yellow]  {label} repair failed ({repair_err}) — skipping")
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
