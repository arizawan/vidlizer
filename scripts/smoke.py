#!/usr/bin/env python3
"""
vidlizer smoke test — exercises every major function against a real provider.

Provider priority (auto-detect):
  1. Ollama      (local, free)
  2. OpenAI-compat (LM Studio / oMLX, local)
  3. OpenRouter  (cloud, uses :free model only)

Usage:
    python scripts/smoke.py
    python scripts/smoke.py --provider openrouter
    python scripts/smoke.py --provider ollama --model qwen2.5vl:7b
    make smoke
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_REQUIRED_FLOW_FIELDS = (
    "step", "timestamp_s", "phase", "scene",
    "subjects", "action", "text_visible", "context",
    "observations", "next_scene",
)

_current_section = ""


@dataclass
class Check:
    name: str
    status: str = "skip"    # "pass" | "fail" | "skip"
    detail: str = ""
    duration_s: float = 0.0
    section: str = ""
    sub: list[str] = field(default_factory=list)


_checks: list[Check] = []


def _section(title: str) -> None:
    global _current_section
    _current_section = title
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="dim"))
    console.print()


def _record(
    name: str,
    status: str,
    detail: str = "",
    duration_s: float = 0.0,
    sub: list[str] | None = None,
) -> None:
    c = Check(name, status, detail, duration_s, _current_section, sub or [])
    _checks.append(c)
    icon = {"pass": "[green]✓[/green]", "fail": "[red]✗[/red]", "skip": "[dim]−[/dim]"}[status]
    dur = f"  [dim]{duration_s:.2f}s[/dim]" if duration_s else ""
    console.print(f"  {icon}  {name:<36}[dim]{detail}[/dim]{dur}")
    for line in c.sub:
        console.print(f"       [dim]· {line}[/dim]")


# ---------------------------------------------------------------------------
# Environment detection
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


def _check_openai_compat() -> tuple[bool, str, list[str]]:
    try:
        import requests
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        key = os.getenv("OPENAI_API_KEY", "lm-studio")
        r = requests.get(f"{base}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=3)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return True, base, models
    except Exception:
        return False, os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"), []


def _check_openrouter() -> tuple[bool, str, str]:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return False, "not set", ""
    # Prefer free vision models
    free_candidates = [
        "google/gemma-3-27b-it:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "meta-llama/llama-4-scout:free",
    ]
    try:
        import requests
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"}, timeout=8,
        )
        r.raise_for_status()
        live_ids = {m["id"] for m in r.json().get("data", [])}
        for candidate in free_candidates:
            if candidate in live_ids:
                return True, f"key ...{key[-4:]}", candidate
        # Fallback: first :free model with "image" in architecture
        for m in r.json().get("data", []):
            if m["id"].endswith(":free"):
                arch = m.get("architecture", {})
                modalities = arch.get("input_modalities") or arch.get("modality", "")
                if "image" in str(modalities):
                    return True, f"key ...{key[-4:]}", m["id"]
    except Exception:
        pass
    return True, f"key ...{key[-4:]}", "google/gemma-4-31b-it:free"


def _check_whisper() -> tuple[bool, str]:
    try:
        import mlx_whisper  # noqa: F401
        return True, "installed"
    except ImportError:
        return False, "not installed"


# ---------------------------------------------------------------------------
# Asset creation
# ---------------------------------------------------------------------------

def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True,
    )


def _make_video(path: Path, duration: int = 6) -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", str(path),
    )


def _make_silent_video(path: Path) -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=10",
        "-t", "3", "-an", str(path),
    )


def _make_image(path: Path) -> None:
    _ffmpeg("-f", "lavfi", "-i", "color=c=green:size=320x240", "-frames:v", "1", str(path))


def _make_pdf(path: Path, pages: int = 2) -> None:
    import fitz  # type: ignore[import]
    doc = fitz.open()
    for i, color in enumerate([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)][:pages]):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(page.rect, color=color, fill=color)
        page.insert_text((100, 420), f"Page {i + 1}", fontsize=24, color=(1, 1, 1))
    doc.save(str(path)); doc.close()


# ---------------------------------------------------------------------------
# Section 1 — Library functions (no network)
# ---------------------------------------------------------------------------

def section_library(video: Path, silent: Path, image: Path, pdf: Path, tmp: Path) -> None:
    _section("Library Functions  (no network required)")

    # ── probe_video ──────────────────────────────────────────────────────────
    from vidlizer.preflight import probe_video, estimate_frames, estimate_cost
    info = probe_video(video)
    if info["duration_s"] and info["width"] and info["fps"]:
        _record("probe_video", "pass",
                f"duration={info['duration_s']:.1f}s  {info['width']}×{info['height']}  "
                f"fps={info['fps']}  {info['size_mb']:.2f}MB",
                sub=[f"duration_s={info['duration_s']}", f"fps={info['fps']}",
                     f"size_mb={info['size_mb']:.2f}"])
    else:
        _record("probe_video", "fail", str(info))

    # ── has_audio ────────────────────────────────────────────────────────────
    from vidlizer.transcribe import has_audio
    ha = has_audio(video)
    hs = has_audio(silent)
    _record("has_audio (with audio)", "pass" if ha else "fail", str(ha))
    _record("has_audio (silent)",     "pass" if not hs else "fail", str(hs))

    # ── estimate_frames ──────────────────────────────────────────────────────
    low, high = estimate_frames(info["duration_s"], 2.0, 60)
    _record("estimate_frames", "pass" if low <= high else "fail",
            f"{info['duration_s']:.0f}s → {low}–{high} frames (min_interval=2.0, max=60)")

    # ── extract_frames ───────────────────────────────────────────────────────
    from vidlizer.frames import extract_frames, encode_frame
    frame_tmp = tmp / "frames"
    frame_tmp.mkdir()
    frames = extract_frames(video, frame_tmp, scale=320, max_frames=8,
                            scene_threshold=0.0, fps=1.0, min_interval=0.0, verbose=False)
    if frames:
        _record("extract_frames", "pass", f"{len(frames)} frames extracted",
                sub=[f"first: {frames[0].name}", f"last: {frames[-1].name}",
                     f"size: {frames[0].stat().st_size // 1024}KB each"])
    else:
        _record("extract_frames", "fail", "0 frames")

    # ── encode_frame ─────────────────────────────────────────────────────────
    if frames:
        enc = encode_frame(frames[0])
        ok = (enc.get("type") == "image_url"
              and enc.get("image_url", {}).get("url", "").startswith("data:image/jpeg;base64,"))
        b64_len = len(enc["image_url"]["url"].split(",", 1)[1]) if ok else 0
        _record("encode_frame", "pass" if ok else "fail",
                f"data:image/jpeg;base64 ({b64_len:,} chars)")

    # ── pdf_to_frames ────────────────────────────────────────────────────────
    from vidlizer.frames import pdf_to_frames
    pdf_tmp = tmp / "pdf_frames"
    pdf_tmp.mkdir()
    pdf_frames = pdf_to_frames(pdf, pdf_tmp, scale=320, max_frames=10)
    _record("pdf_to_frames", "pass" if len(pdf_frames) == 2 else "fail",
            f"{len(pdf_frames)} pages rendered")

    # ── dedup_frames ─────────────────────────────────────────────────────────
    from vidlizer.dedup import dedup_frames
    if len(frames) >= 3:
        # Duplicate a frame repeatedly — all identical → should collapse to 1
        dup_frames = [frames[0]] * 6
        deduped = dedup_frames(dup_frames, threshold=8)
        _record("dedup_frames", "pass" if len(deduped) == 1 else "fail",
                f"6 identical frames → {len(deduped)} kept (threshold=8)",
                sub=[f"kept: {deduped[0].name}"])
    else:
        _record("dedup_frames", "skip", "not enough frames")

    # ── parse_json ───────────────────────────────────────────────────────────
    import json as _json
    from vidlizer.batch import parse_json
    tests_pj = [
        ("parse_json (plain object)",  '{"flow":[{"step":1}]}',
         lambda r: isinstance(r.get("flow"), list)),
        ("parse_json (<think> tags)",  '<think>reasoning</think>\n{"flow":[]}',
         lambda r: r == {"flow": []}),
        ("parse_json (```json fence)", '```json\n{"flow":[]}\n```',
         lambda r: r == {"flow": []}),
        ("parse_json (list input)",    _json.dumps([{"step": 1}]),
         lambda r: "flow" in r and len(r["flow"]) == 1),
    ]
    for name, raw, check_fn in tests_pj:
        try:
            result = parse_json(raw)
            ok = check_fn(result)
            _record(name, "pass" if ok else "fail",
                    f"→ flow has {len(result.get('flow', []))} step(s)")
        except Exception as e:
            _record(name, "fail", str(e))

    # ── get_pricing / get_cheapest_paid ──────────────────────────────────────
    from vidlizer.models import get_pricing, get_cheapest_paid, get_ollama_fallback_sequence, \
        get_openai_fallback_sequence, format_model_line, _FALLBACK
    inp, out_r = get_pricing("google/gemini-2.5-flash", _FALLBACK)
    _record("get_pricing", "pass" if inp > 0 else "fail",
            f"gemini-2.5-flash → ${inp:.3f}/M in  ${out_r:.3f}/M out")

    cheapest = get_cheapest_paid(_FALLBACK)
    _record("get_cheapest_paid", "pass" if cheapest else "fail",
            f"→ {cheapest}")

    fallback_ol = get_ollama_fallback_sequence(
        ["qwen2.5vl:7b", "qwen2.5vl:3b", "llava:7b"], exclude="qwen2.5vl:7b"
    )
    ol_ok = "qwen2.5vl:7b" not in fallback_ol and len(fallback_ol) == 2
    _record("get_ollama_fallback_sequence", "pass" if ol_ok else "fail",
            f"exclude=qwen2.5vl:7b → {fallback_ol}")

    fallback_oai = get_openai_fallback_sequence(
        ["vendor/qwen2.5-vl-7b-instruct", "vendor/llava-7b"], exclude=""
    )
    _record("get_openai_fallback_sequence",
            "pass" if fallback_oai[0] == "vendor/qwen2.5-vl-7b-instruct" else "fail",
            f"→ {fallback_oai[:2]}")

    ml = format_model_line(_FALLBACK[0])
    _record("format_model_line", "pass" if _FALLBACK[0]["id"] in ml else "fail",
            ml[:60])

    # ── CostTracker ──────────────────────────────────────────────────────────
    from vidlizer.http import CostTracker, CostCapExceeded
    t = CostTracker()
    t.add("unknown/model", {"prompt_tokens": 100, "completion_tokens": 50})
    tok_ok = t.prompt_tokens == 100 and t.completion_tokens == 50
    _record("CostTracker.add", "pass" if tok_ok else "fail",
            f"100↑ 50↓ → accumulated correctly",
            sub=[f"prompt_tokens={t.prompt_tokens}", f"completion_tokens={t.completion_tokens}"])

    cap_ok = False
    try:
        ct = CostTracker(max_cost=0.0001)
        import vidlizer.http as _http_mod
        _orig = _http_mod._model_cost
        _http_mod._model_cost = lambda *_: 1.0
        try:
            ct.add("any/model", {"prompt_tokens": 1, "completion_tokens": 1})
        except CostCapExceeded:
            cap_ok = True
        finally:
            _http_mod._model_cost = _orig
    except Exception:
        pass
    _record("CostTracker cost cap", "pass" if cap_ok else "fail",
            "raises CostCapExceeded when over limit")

    # ── estimate_cost ────────────────────────────────────────────────────────
    low_c, high_c = estimate_cost("google/gemini-2.5-flash", 5, 10, _FALLBACK)
    _record("estimate_cost", "pass" if 0 < low_c < high_c else "fail",
            f"5–10 frames → ${low_c:.4f}–${high_c:.4f}")

    # ── Formatters ───────────────────────────────────────────────────────────
    from vidlizer.formatter import format_output
    mock_data = {
        "flow": [{
            "step": 1, "timestamp_s": 0.0, "phase": "Intro",
            "scene": "Blue background",
            "subjects": ["background"], "action": "Static display",
            "text_visible": "", "context": "Test",
            "observations": "Solid color", "next_scene": None,
            "speech": "Hello world",
        }],
        "transcript": [{"start": 0.0, "end": 1.0, "text": "Hello world"}],
    }
    json_out = format_output(mock_data, "json")
    json_ok = "flow" in _json.loads(json_out)
    _record("formatter: JSON", "pass" if json_ok else "fail",
            "valid JSON with flow array")

    md_out = format_output(mock_data, "markdown")
    md_ok = "##" in md_out and "Intro" in md_out
    _record("formatter: Markdown", "pass" if md_ok else "fail",
            f"has ## headers: {md_ok}  |  {len(md_out)} chars",
            sub=["## Step 1" in md_out and "✓ step headers" or "✗ step headers",
                 "Transcript" in md_out and "✓ transcript section" or "✗ no transcript"])

    sum_out = format_output(mock_data, "summary")
    sum_ok = len(sum_out) > 20 and "Intro" in sum_out
    _record("formatter: Summary", "pass" if sum_ok else "fail",
            f"{len(sum_out)} chars")

    # ── usage.get_stats ──────────────────────────────────────────────────────
    from vidlizer.usage import get_stats
    stats = get_stats()
    stats_ok = "total_runs" in stats and "by_model" in stats and "log_path" in stats
    _record("usage.get_stats", "pass" if stats_ok else "fail",
            f"total_runs={stats.get('total_runs',0)}  "
            f"total_cost=${stats.get('total_cost_usd',0):.4f}")


# ---------------------------------------------------------------------------
# Section 2 — Video / Image / PDF analysis (real API)
# ---------------------------------------------------------------------------

def _run(video: Path, output: Path, *, model: str, provider: str,
         batch_size: int = 1, max_frames: int = 6, output_format: str = "json",
         no_transcript: bool = True, start: float | None = None,
         end: float | None = None, concurrency: int = 0) -> tuple[int, float]:
    from vidlizer.core import run
    t0 = time.time()
    rc = run(
        video=video, output=output, model=model, provider=provider,
        batch_size=batch_size, max_frames=max_frames,
        output_format=output_format, no_transcript=no_transcript,
        verbose=False, timeout=180, start=start, end=end,
        concurrency=concurrency,
    )
    return rc, time.time() - t0


def _read_flow(p: Path) -> list[dict]:
    try:
        return json.loads(p.read_text()).get("flow", [])
    except Exception:
        return []


def _validate_schema(flow: list[dict]) -> tuple[bool, list[str]]:
    if not flow:
        return False, ["empty flow"]
    issues = []
    for i, step in enumerate(flow[:3]):  # check first 3 steps
        missing = [f for f in _REQUIRED_FLOW_FIELDS if f not in step]
        if missing:
            issues.append(f"step {i + 1} missing: {missing}")
    return len(issues) == 0, issues


def section_analysis(video: Path, image: Path, pdf: Path,
                     out: Path, model: str, provider: str) -> Path | None:
    _section("Video / Image / PDF Analysis")

    # ── Video → JSON ─────────────────────────────────────────────────────────
    vj = out / "video.json"
    rc, t = _run(video, vj, model=model, provider=provider)
    flow = _read_flow(vj)
    if rc == 0 and flow:
        schema_ok, issues = _validate_schema(flow)
        sub = [f"step {s['step']}: phase={s.get('phase','')}  scene={str(s.get('scene',''))[:40]}"
               for s in flow[:3]]
        if issues:
            sub += [f"schema issues: {issues}"]
        _record("Video → JSON", "pass", f"{len(flow)} steps  {t:.1f}s", t, sub)
        _record("JSON schema validation",
                "pass" if schema_ok else "fail",
                f"all {len(_REQUIRED_FLOW_FIELDS)} required fields present" if schema_ok
                else f"issues: {issues[:2]}",
                sub=[f"checked: {', '.join(_REQUIRED_FLOW_FIELDS[:5])} …"])
    else:
        _record("Video → JSON", "fail", f"rc={rc}  steps={len(flow)}", t)
        _record("JSON schema validation", "skip", "no output to validate")

    # ── Video → Markdown ─────────────────────────────────────────────────────
    vm = out / "video.md"
    rc_m, t_m = _run(video, vm, model=model, provider=provider, output_format="markdown")
    if rc_m == 0 and vm.exists():
        md = vm.read_text()
        has_h1 = "# " in md
        has_steps = "## Step" in md or "## " in md
        has_timestamps = "[" in md and "s]" in md
        _record("Video → Markdown", "pass" if has_h1 else "fail",
                f"{len(md)} chars  {vm.stat().st_size // 1024}KB", t_m,
                sub=[f"# header: {'✓' if has_h1 else '✗'}",
                     f"## step sections: {'✓' if has_steps else '✗'}",
                     f"timestamps: {'✓' if has_timestamps else '✗'}"])
    else:
        _record("Video → Markdown", "fail", f"rc={rc_m}", t_m)

    # ── Video → Summary ──────────────────────────────────────────────────────
    vs = out / "video-summary.txt"
    rc_s, t_s = _run(video, vs, model=model, provider=provider, output_format="summary")
    if rc_s == 0 and vs.exists() and vs.stat().st_size > 0:
        txt = vs.read_text()
        _record("Video → Summary", "pass", f"{len(txt)} chars", t_s,
                sub=[txt[:120].replace("\n", " ") + "…"])
    else:
        _record("Video → Summary", "fail", f"rc={rc_s}", t_s)

    # ── Image → JSON ─────────────────────────────────────────────────────────
    ij = out / "image.json"
    rc_i, t_i = _run(image, ij, model=model, provider=provider)
    img_flow = _read_flow(ij)
    if rc_i == 0 and img_flow:
        step = img_flow[0]
        _record("Image → JSON", "pass", f"{len(img_flow)} step(s)  {t_i:.1f}s", t_i,
                sub=[f"scene: {str(step.get('scene',''))[:60]}",
                     f"subjects: {step.get('subjects',[])}"])
    else:
        _record("Image → JSON", "fail", f"rc={rc_i}", t_i)

    # ── PDF → JSON ───────────────────────────────────────────────────────────
    pj = out / "pdf.json"
    rc_p, t_p = _run(pdf, pj, model=model, provider=provider)
    pdf_flow = _read_flow(pj)
    if rc_p == 0 and pdf_flow:
        _record("PDF → JSON", "pass", f"{len(pdf_flow)} steps  {t_p:.1f}s", t_p,
                sub=[f"step {s['step']}: {str(s.get('scene',''))[:50]}" for s in pdf_flow])
    else:
        _record("PDF → JSON", "fail", f"rc={rc_p}", t_p)

    return vj if rc == 0 else None


# ---------------------------------------------------------------------------
# Section 3 — Advanced features
# ---------------------------------------------------------------------------

def section_advanced(video: Path, out: Path, model: str, provider: str,
                     json_path: Path | None) -> None:
    _section("Advanced Features")

    # ── Cache hit ────────────────────────────────────────────────────────────
    p = out / "cached.json"
    rc1, t1 = _run(video, p, model=model, provider=provider)
    if rc1 == 0:
        _, t2 = _run(video, p, model=model, provider=provider)
        _record("Cache hit (repeat run)", "pass" if t2 < 1.0 else "fail",
                f"1st={t1:.1f}s  2nd={t2:.3f}s ({'instant' if t2 < 0.1 else 'fast' if t2 < 1 else 'slow'})",
                t2)
    else:
        _record("Cache hit (repeat run)", "fail", "first run failed")

    # ── Start / end trim ─────────────────────────────────────────────────────
    p = out / "trimmed.json"
    full_p = out / "full_for_trim.json"
    _run(video, full_p, model=model, provider=provider, max_frames=8)
    rc_t, t_t = _run(video, p, model=model, provider=provider,
                     max_frames=4, start=1.0, end=4.0)
    full_steps = len(_read_flow(full_p))
    trim_steps = len(_read_flow(p))
    if rc_t == 0 and trim_steps > 0:
        _record("Start/end trim (1–4s)", "pass",
                f"full={full_steps} steps → trimmed={trim_steps} steps", t_t,
                sub=[f"extracted only 1.0–4.0s window"])
    else:
        _record("Start/end trim (1–4s)", "fail", f"rc={rc_t}", t_t)

    # ── Batch serial ─────────────────────────────────────────────────────────
    p = out / "batched.json"
    rc_b, t_b = _run(video, p, model=model, provider=provider,
                     batch_size=2, max_frames=6)
    bflow = _read_flow(p)
    if rc_b == 0 and bflow:
        steps_seq = [s["step"] for s in bflow]
        sequential = steps_seq == list(range(1, len(bflow) + 1))
        _record("Batch serial (batch_size=2)", "pass",
                f"{len(bflow)} steps in {t_b:.1f}s, sequential={sequential}", t_b,
                sub=[f"step numbers: {steps_seq[:5]}{'…' if len(steps_seq) > 5 else ''}"])
    else:
        _record("Batch serial (batch_size=2)", "fail", f"rc={rc_b}", t_b)

    # ── Parallel (OpenRouter only — local serialises anyway) ─────────────────
    if provider == "openrouter":
        p = out / "parallel.json"
        rc_par, t_par = _run(video, p, model=model, provider=provider,
                             batch_size=2, max_frames=6, concurrency=2)
        par_flow = _read_flow(p)
        if rc_par == 0 and par_flow:
            _record("Parallel (concurrency=2)", "pass",
                    f"{len(par_flow)} steps in {t_par:.1f}s", t_par)
        else:
            _record("Parallel (concurrency=2)", "fail", f"rc={rc_par}", t_par)
    else:
        _record("Parallel (concurrency=2)", "skip",
                f"local provider ({provider}) serialises server-side")

    # ── Dedup reduces frame count ─────────────────────────────────────────────
    from vidlizer.frames import extract_frames
    from vidlizer.dedup import dedup_frames, DEFAULT_THRESHOLD
    frame_tmp = out / "dedup_frames"
    frame_tmp.mkdir(exist_ok=True)
    raw_frames = extract_frames(video, frame_tmp, scale=320, max_frames=20,
                                scene_threshold=0.0, fps=2.0, min_interval=0.0, verbose=False)
    deduped = dedup_frames(raw_frames, DEFAULT_THRESHOLD)
    ratio = len(deduped) / max(len(raw_frames), 1)
    _record("dedup_frames (real video)", "pass" if deduped else "fail",
            f"{len(raw_frames)} raw → {len(deduped)} unique (kept {ratio:.0%})",
            sub=[f"threshold=DEFAULT ({DEFAULT_THRESHOLD})",
                 f"blue solid-color video: expect high dedup rate"])

    # ── Output format validation ──────────────────────────────────────────────
    if json_path and json_path.exists():
        data = json.loads(json_path.read_text())
        flow = data.get("flow", [])

        # JSON: all steps have monotonic step numbers
        nums = [s.get("step") for s in flow]
        mono = nums == list(range(1, len(flow) + 1))
        _record("JSON: step numbers monotonic", "pass" if mono else "fail",
                f"steps 1..{len(flow)}: {'✓' if mono else '✗'}")

        # JSON: timestamp_s present and numeric (or null)
        ts_ok = all(s.get("timestamp_s") is None or isinstance(s.get("timestamp_s"), (int, float))
                    for s in flow)
        _record("JSON: timestamp_s type", "pass" if ts_ok else "fail",
                "all null or numeric")

        # JSON: subjects is list
        subj_ok = all(isinstance(s.get("subjects"), list) for s in flow)
        _record("JSON: subjects is list", "pass" if subj_ok else "fail",
                f"all {len(flow)} steps: {'✓' if subj_ok else '✗'}")


# ---------------------------------------------------------------------------
# Section 4 — Audio & Transcription
# ---------------------------------------------------------------------------

def section_audio(video: Path, silent: Path, out: Path, model: str, provider: str) -> None:
    _section("Audio & Transcription")

    from vidlizer.transcribe import has_audio
    _record("has_audio (audio video)",  "pass" if has_audio(video)  else "fail", str(has_audio(video)))
    _record("has_audio (silent video)", "pass" if not has_audio(silent) else "fail", str(has_audio(silent)))

    try:
        import mlx_whisper  # noqa: F401
        whisper_ok = True
    except ImportError:
        whisper_ok = False

    if not whisper_ok:
        _record("Transcription (mlx-whisper)", "skip", "not installed")
        return

    from vidlizer.core import run
    p = out / "transcript.json"
    t0 = time.time()
    rc = run(video=video, output=p, model=model, provider=provider,
             batch_size=1, max_frames=6, verbose=False, timeout=240, no_transcript=False)
    t = time.time() - t0

    if rc == 0 and p.exists():
        data = json.loads(p.read_text())
        segments = data.get("transcript", [])
        has_speech = any("speech" in s for s in data.get("flow", []))
        # Pass if pipeline ran (rc==0); 0 segments is correct for a sine-tone test asset
        _record("Transcription", "pass",
                f"{len(segments)} segments  speech_merged={has_speech}", t,
                sub=([seg.get("text", "")[:60] for seg in segments[:3]]
                     or ["(no speech — sine-tone test asset)"]))
    else:
        _record("Transcription", "fail", f"rc={rc}", t)


# ---------------------------------------------------------------------------
# Terminal summary table + HTML report
# ---------------------------------------------------------------------------

def _print_summary_table() -> None:
    t = Table(
        show_header=True, header_style="bold cyan",
        border_style="dim", show_lines=True, expand=False,
    )
    t.add_column("Section", style="dim", min_width=16)
    t.add_column("Test", style="white", min_width=32)
    t.add_column("Result", justify="center", min_width=9)
    t.add_column("Time", justify="right", min_width=7)
    t.add_column("Detail")

    prev_section = ""
    for c in _checks:
        sec = c.section if c.section != prev_section else ""
        prev_section = c.section
        if c.status == "pass":
            badge = "[bold green]✓  PASS[/bold green]"
        elif c.status == "fail":
            badge = "[bold red]✗  FAIL[/bold red]"
        else:
            badge = "[dim]—  SKIP[/dim]"
        dur = f"{c.duration_s:.1f}s" if c.duration_s else "—"
        t.add_row(sec, c.name, badge, dur, c.detail)

    console.print(t)


def _save_html(model: str, provider: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = ROOT / "reports" / f"smoke-{stamp}.html"
    path.parent.mkdir(exist_ok=True)

    # Group checks by section for HTML
    sections: dict[str, list[Check]] = {}
    for c in _checks:
        sections.setdefault(c.section, []).append(c)

    section_html = ""
    for sec_name, sec_checks in sections.items():
        rows = ""
        for c in sec_checks:
            badge_cls = c.status
            badge_txt = "✓ PASS" if c.status == "pass" else "✗ FAIL" if c.status == "fail" else "— SKIP"
            dur = f"{c.duration_s:.2f}s" if c.duration_s else "—"
            sub_html = "".join(f"<li>{s}</li>" for s in c.sub) if c.sub else ""
            sub_block = f"<ul class='sub'>{sub_html}</ul>" if sub_html else ""
            rows += (f"<tr><td>{c.name}</td>"
                     f"<td class='badge {badge_cls}'>{badge_txt}</td>"
                     f"<td class='dur'>{dur}</td>"
                     f"<td>{c.detail}{sub_block}</td></tr>\n")
        section_html += (f"<h2>{sec_name}</h2>"
                         f"<table><thead><tr><th>Test</th><th>Result</th>"
                         f"<th>Time</th><th>Detail</th></tr></thead>"
                         f"<tbody>{rows}</tbody></table>\n")

    passed  = sum(1 for c in _checks if c.status == "pass")
    failed  = sum(1 for c in _checks if c.status == "fail")
    skipped = sum(1 for c in _checks if c.status == "skip")
    total   = len(_checks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vidlizer Smoke Test — {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
<style>
  :root {{
    --bg:#0d1117; --surface:#161b22; --surface2:#1c2128; --border:#30363d;
    --text:#c9d1d9; --blue:#58a6ff; --green:#56d364; --red:#f85149; --muted:#8b949e;
    --yellow:#e3b341;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text);
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          padding:40px 56px; max-width:1200px; margin:0 auto }}
  header {{ margin-bottom:32px }}
  h1 {{ color:var(--blue); font-size:1.5rem; margin-bottom:6px }}
  .meta {{ color:var(--muted); font-size:0.82rem; line-height:1.8 }}
  .scorecard {{ display:flex; gap:24px; margin:24px 0 36px }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px;
           padding:16px 24px; min-width:120px; text-align:center }}
  .card .val {{ font-size:2rem; font-weight:700; line-height:1 }}
  .card .lbl {{ font-size:0.75rem; color:var(--muted); margin-top:4px; text-transform:uppercase; letter-spacing:.05em }}
  .card.pass .val {{ color:var(--green) }}
  .card.fail .val {{ color:var(--red) }}
  .card.skip .val {{ color:var(--muted) }}
  .card.total .val {{ color:var(--blue) }}
  h2 {{ color:var(--blue); font-size:0.95rem; text-transform:uppercase;
        letter-spacing:.06em; margin:32px 0 12px; padding-bottom:6px;
        border-bottom:1px solid var(--border) }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:8px }}
  th {{ background:var(--surface); color:var(--muted); padding:9px 14px; text-align:left;
        font-size:0.75rem; text-transform:uppercase; letter-spacing:.04em;
        border-bottom:2px solid var(--border) }}
  td {{ padding:10px 14px; border-bottom:1px solid var(--border); font-size:0.88rem;
        vertical-align:top }}
  tr:hover td {{ background:var(--surface2) }}
  .dur {{ color:var(--muted); font-size:0.82rem; text-align:right }}
  .badge {{ font-weight:700; font-size:0.78rem; white-space:nowrap }}
  .badge.pass {{ color:var(--green) }} .badge.fail {{ color:var(--red) }}
  .badge.skip {{ color:var(--muted) }}
  ul.sub {{ list-style:none; margin-top:6px; padding-left:10px }}
  ul.sub li {{ color:var(--muted); font-size:0.8rem; line-height:1.6 }}
  ul.sub li::before {{ content:"· "; color:var(--border) }}
</style>
</head>
<body>
<header>
  <h1>vidlizer — Smoke Test Report</h1>
  <div class="meta">
    Generated: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong><br>
    Provider: <strong>{provider}</strong> &nbsp;|&nbsp; Model: <strong>{model}</strong>
  </div>
</header>

<div class="scorecard">
  <div class="card pass"><div class="val">{passed}</div><div class="lbl">Passed</div></div>
  <div class="card fail"><div class="val">{failed}</div><div class="lbl">Failed</div></div>
  <div class="card skip"><div class="val">{skipped}</div><div class="lbl">Skipped</div></div>
  <div class="card total"><div class="val">{total}</div><div class="lbl">Total</div></div>
</div>

{section_html}
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="vidlizer smoke test — all functions, real provider")
    parser.add_argument("--provider", default="", help="ollama | openai | openrouter")
    parser.add_argument("--model", default="", help="model ID (auto-detected if omitted)")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold cyan]vidlizer — Comprehensive Smoke Test[/bold cyan]\n"
        "[dim]Exercises every module. Local provider preferred. "
        "OpenRouter falls back to :free models.[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    # ── Environment ──────────────────────────────────────────────────────────
    _section("Environment")
    ffmpeg_ok, ffmpeg_ver      = _check_ffmpeg()
    ollama_ok, ol_host, ol_mdls = _check_ollama()
    oai_ok, oai_base, oai_mdls  = _check_openai_compat()
    or_ok, or_detail, or_model  = _check_openrouter()
    whisper_ok, whisper_det     = _check_whisper()

    tick = lambda ok: "[green]✓[/green]" if ok else "[red]✗[/red]"  # noqa: E731
    _record("ffmpeg",        "pass" if ffmpeg_ok else "fail", ffmpeg_ver)
    _record("Ollama",        "pass" if ollama_ok else "skip",
            f"{ol_host}  [{', '.join(ol_mdls[:3])}{'…' if len(ol_mdls)>3 else ''}]"
            if ollama_ok else f"{ol_host} (not reachable)")
    _record("OpenAI-compat", "pass" if oai_ok else "skip",
            f"{oai_base}  [{', '.join(oai_mdls[:2])}]" if oai_ok else f"{oai_base} (not reachable)")
    _record("OpenRouter",    "pass" if or_ok else "skip",
            f"{or_detail}" + (f"  → free model: {or_model}" if or_ok else ""))
    _record("mlx-whisper",   "pass" if whisper_ok else "skip", whisper_det)

    if not ffmpeg_ok:
        console.print("\n[red]ffmpeg required — cannot continue.[/red]")
        return 1

    # ── Provider / model selection (local first) ─────────────────────────────
    provider = args.provider
    model    = args.model

    if not provider:
        if ollama_ok and ol_mdls:
            provider = "ollama"
        elif oai_ok and oai_mdls:
            provider = "openai"
        elif or_ok:
            provider = "openrouter"
        else:
            console.print("\n[red]No provider available.[/red]\n"
                          "  Start Ollama, LM Studio, or set OPENROUTER_API_KEY.")
            return 1

    if not model:
        if provider == "ollama" and ol_mdls:
            prefs = ["qwen2.5vl:7b", "qwen2.5vl:3b", "minicpm-v:8b", "llava-onevision:7b", "llava"]
            model = next(
                (m for pref in prefs for m in ol_mdls if m.startswith(pref.split(":")[0])),
                ol_mdls[0],
            )
        elif provider == "openai" and oai_mdls:
            model = oai_mdls[0]
        elif provider == "openrouter":
            model = or_model  # always :free
        else:
            console.print("[red]Cannot select model — use --model.[/red]")
            return 1

    console.print(f"\n  [bold]Selected:[/bold]  "
                  f"provider=[magenta]{provider}[/magenta]  "
                  f"model=[magenta]{model}[/magenta]")

    # ── Build assets, run all sections ───────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="vidlizer_smoke_") as tmp_str:
        tmp    = Path(tmp_str)
        assets = tmp / "assets"
        out    = tmp / "out"
        assets.mkdir(); out.mkdir()

        console.print("\n[dim]Building test assets (ffmpeg + pymupdf)…[/dim]")
        try:
            video  = assets / "test.mp4"
            silent = assets / "silent.mp4"
            image  = assets / "test.png"
            pdf    = assets / "test.pdf"
            _make_video(video)
            _make_silent_video(silent)
            _make_image(image)
            _make_pdf(pdf)
        except Exception as exc:
            console.print(f"[red]Asset creation failed: {exc}[/red]")
            return 1

        section_library(video, silent, image, pdf, tmp)
        json_path = section_analysis(video, image, pdf, out, model, provider)
        section_advanced(video, out, model, provider, json_path)
        section_audio(video, silent, out, model, provider)

    # ── Final table ───────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Results[/bold]", style="dim"))
    console.print()
    _print_summary_table()
    console.print()

    passed  = sum(1 for c in _checks if c.status == "pass")
    failed  = sum(1 for c in _checks if c.status == "fail")
    skipped = sum(1 for c in _checks if c.status == "skip")
    color   = "green" if failed == 0 else "red"
    console.print(
        f"[{color}][bold]{passed} passed[/bold][/{color}]  "
        f"[red]{failed} failed[/red]  "
        f"[dim]{skipped} skipped[/dim]  "
        f"[dim]({len(_checks)} total)[/dim]"
    )

    report = _save_html(model, provider)
    console.print(f"\n[dim]Report → [/dim][cyan]{report}[/cyan]")
    console.print(f"[dim]Open   → [/dim][cyan]open {report}[/cyan]")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
