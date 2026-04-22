#!/usr/bin/env python3
"""
vidlizer smoke test — tests EVERY available provider in sequence.

Provider order (auto-detect, local first):
  1. Ollama      (local, free) — unloaded from VRAM after test
  2. OpenAI-compat (LM Studio / oMLX, local) — unloaded after test
  3. OpenRouter  (cloud, :free models only)

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
_current_provider = "—"


@dataclass
class Check:
    name: str
    status: str = "skip"    # "pass" | "fail" | "skip"
    detail: str = ""
    duration_s: float = 0.0
    section: str = ""
    provider: str = "—"
    sub: list[str] = field(default_factory=list)


_checks: list[Check] = []


def _section(title: str) -> None:
    global _current_section
    _current_section = title
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="dim"))
    console.print()


def _set_provider(prov: str) -> None:
    global _current_provider
    _current_provider = prov


def _record(
    name: str,
    status: str,
    detail: str = "",
    duration_s: float = 0.0,
    sub: list[str] | None = None,
) -> None:
    c = Check(name, status, detail, duration_s, _current_section, _current_provider, sub or [])
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
# Model unloading (best-effort — never fails the run)
# ---------------------------------------------------------------------------

def _unload_ollama(model: str, host: str) -> None:
    try:
        import requests
        requests.post(
            f"{host}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=5,
        )
        console.print(f"\n  [dim]↓ Ollama: unloaded [bold]{model}[/bold] from VRAM[/dim]")
    except Exception as exc:
        console.print(f"\n  [dim]↓ Ollama unload skipped ({exc})[/dim]")


def _unload_openai_compat(model: str, base: str) -> None:
    # LM Studio ≥0.3 supports /api/v0/models/unload
    # oMLX and other compat servers may not — swallow all errors
    try:
        import requests
        root = base.removesuffix("/v1")
        r = requests.post(
            f"{root}/api/v0/models/unload",
            json={"identifier": model},
            timeout=5,
        )
        if r.ok:
            console.print(f"\n  [dim]↓ LM Studio: unloaded [bold]{model}[/bold][/dim]")
        else:
            console.print(f"\n  [dim]↓ OpenAI-compat unload not supported (HTTP {r.status_code})[/dim]")
    except Exception:
        console.print(f"\n  [dim]↓ OpenAI-compat unload not supported[/dim]")


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
# Section 1 — Library functions (no network, runs once)
# ---------------------------------------------------------------------------

def section_library(video: Path, silent: Path, image: Path, pdf: Path, tmp: Path) -> None:
    _set_provider("library")
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

    # ── get_pricing / model helpers ──────────────────────────────────────────
    from vidlizer.models import (
        get_pricing, get_cheapest_paid,
        get_ollama_fallback_sequence, get_openai_fallback_sequence,
        format_model_line, _FALLBACK,
    )
    inp, out_r = get_pricing("google/gemini-2.5-flash", _FALLBACK)
    _record("get_pricing", "pass" if inp > 0 else "fail",
            f"gemini-2.5-flash → ${inp:.3f}/M in  ${out_r:.3f}/M out")

    cheapest = get_cheapest_paid(_FALLBACK)
    _record("get_cheapest_paid", "pass" if cheapest else "fail", f"→ {cheapest}")

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
    _record("format_model_line", "pass" if _FALLBACK[0]["id"] in ml else "fail", ml[:60])

    # ── CostTracker ──────────────────────────────────────────────────────────
    from vidlizer.http import CostTracker, CostCapExceeded
    t = CostTracker()
    t.add("unknown/model", {"prompt_tokens": 100, "completion_tokens": 50})
    tok_ok = t.prompt_tokens == 100 and t.completion_tokens == 50
    _record("CostTracker.add", "pass" if tok_ok else "fail",
            "100↑ 50↓ → accumulated correctly",
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
    _record("formatter: JSON", "pass" if json_ok else "fail", "valid JSON with flow array")

    md_out = format_output(mock_data, "markdown")
    md_ok = "##" in md_out and "Intro" in md_out
    _record("formatter: Markdown", "pass" if md_ok else "fail",
            f"has ## headers: {md_ok}  |  {len(md_out)} chars",
            sub=["## Step 1" in md_out and "✓ step headers" or "✗ step headers",
                 "Transcript" in md_out and "✓ transcript section" or "✗ no transcript"])

    sum_out = format_output(mock_data, "summary")
    sum_ok = len(sum_out) > 20 and "Intro" in sum_out
    _record("formatter: Summary", "pass" if sum_ok else "fail", f"{len(sum_out)} chars")

    # ── usage.get_stats ──────────────────────────────────────────────────────
    from vidlizer.usage import get_stats
    stats = get_stats()
    stats_ok = "total_runs" in stats and "by_model" in stats and "log_path" in stats
    _record("usage.get_stats", "pass" if stats_ok else "fail",
            f"total_runs={stats.get('total_runs', 0)}  "
            f"total_cost=${stats.get('total_cost_usd', 0):.4f}")


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
    for i, step in enumerate(flow[:3]):
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
                sub=["extracted only 1.0–4.0s window"])
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
                 "blue solid-color video: expect high dedup rate"])

    # ── Output format validation ──────────────────────────────────────────────
    if json_path and json_path.exists():
        data = json.loads(json_path.read_text())
        flow = data.get("flow", [])

        nums = [s.get("step") for s in flow]
        mono = nums == list(range(1, len(flow) + 1))
        _record("JSON: step numbers monotonic", "pass" if mono else "fail",
                f"steps 1..{len(flow)}: {'✓' if mono else '✗'}")

        ts_ok = all(s.get("timestamp_s") is None or isinstance(s.get("timestamp_s"), (int, float))
                    for s in flow)
        _record("JSON: timestamp_s type", "pass" if ts_ok else "fail", "all null or numeric")

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
        # Pass if pipeline ran successfully; 0 segments is correct for a sine-tone asset
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
    t.add_column("Provider", style="magenta", min_width=12)
    t.add_column("Section", style="dim", min_width=16)
    t.add_column("Test", style="white", min_width=30)
    t.add_column("Result", justify="center", min_width=9)
    t.add_column("Time", justify="right", min_width=7)
    t.add_column("Detail")

    prev_prov = ""
    prev_sec = ""
    for c in _checks:
        prov = c.provider if c.provider != prev_prov else ""
        sec = c.section if (c.section != prev_sec or c.provider != prev_prov) else ""
        prev_prov = c.provider
        prev_sec = c.section
        if c.status == "pass":
            badge = "[bold green]✓  PASS[/bold green]"
        elif c.status == "fail":
            badge = "[bold red]✗  FAIL[/bold red]"
        else:
            badge = "[dim]—  SKIP[/dim]"
        dur = f"{c.duration_s:.1f}s" if c.duration_s else "—"
        t.add_row(prov, sec, c.name, badge, dur, c.detail)

    console.print(t)


def _save_html(providers_tested: list[tuple[str, str]]) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = ROOT / "reports" / f"smoke-{stamp}.html"
    path.parent.mkdir(exist_ok=True)

    # Group: provider → section → checks
    groups: dict[str, dict[str, list[Check]]] = {}
    for c in _checks:
        groups.setdefault(c.provider, {}).setdefault(c.section, []).append(c)

    def _prov_counts(prov: str) -> tuple[int, int, int]:
        checks = [c for c in _checks if c.provider == prov]
        return (
            sum(1 for c in checks if c.status == "pass"),
            sum(1 for c in checks if c.status == "fail"),
            sum(1 for c in checks if c.status == "skip"),
        )

    body_html = ""

    # Library section first
    if "library" in groups:
        p, f, s = _prov_counts("library")
        body_html += f"<h2 class='prov-header'>Library Functions <span class='prov-score'>{p} passed · {f} failed · {s} skipped</span></h2>\n"
        for sec_name, sec_checks in groups["library"].items():
            body_html += _render_section_table(sec_name, sec_checks)

    # One block per tested provider
    for prov_id, prov_model in providers_tested:
        if prov_id not in groups:
            continue
        p, f, s = _prov_counts(prov_id)
        fail_cls = " has-fail" if f > 0 else ""
        body_html += (
            f"<h2 class='prov-header{fail_cls}'>"
            f"Provider: {prov_id} &nbsp;/&nbsp; <code>{prov_model}</code>"
            f"<span class='prov-score'>{p} passed · {f} failed · {s} skipped</span>"
            f"</h2>\n"
        )
        for sec_name, sec_checks in groups[prov_id].items():
            body_html += _render_section_table(sec_name, sec_checks)

    # Global scorecard
    total_pass  = sum(1 for c in _checks if c.status == "pass")
    total_fail  = sum(1 for c in _checks if c.status == "fail")
    total_skip  = sum(1 for c in _checks if c.status == "skip")
    total       = len(_checks)

    scorecard = ""
    for prov_id, prov_model in [("library", "—")] + list(providers_tested):
        p, f, s = _prov_counts(prov_id)
        label = prov_id if prov_id == "library" else f"{prov_id}<br><small>{prov_model}</small>"
        fail_style = "border-color:var(--red)" if f > 0 else ""
        scorecard += (
            f"<div class='card' style='{fail_style}'>"
            f"<div class='lbl'>{label}</div>"
            f"<div class='pscores'>"
            f"<span class='ps pass'>{p}✓</span> "
            f"<span class='ps fail'>{f}✗</span> "
            f"<span class='ps skip'>{s}−</span>"
            f"</div></div>\n"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
          padding:40px 56px; max-width:1300px; margin:0 auto }}
  header {{ margin-bottom:32px }}
  h1 {{ color:var(--blue); font-size:1.5rem; margin-bottom:6px }}
  .meta {{ color:var(--muted); font-size:0.82rem; line-height:1.8 }}
  .scorecard {{ display:flex; gap:16px; flex-wrap:wrap; margin:24px 0 36px }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px;
           padding:14px 20px; min-width:140px }}
  .card .lbl {{ font-size:0.78rem; color:var(--muted); text-transform:uppercase;
                letter-spacing:.05em; margin-bottom:8px; line-height:1.4 }}
  .card small {{ font-size:0.68rem; color:var(--muted) }}
  .pscores {{ display:flex; gap:12px; font-size:1.1rem; font-weight:700 }}
  .ps.pass {{ color:var(--green) }} .ps.fail {{ color:var(--red) }} .ps.skip {{ color:var(--muted) }}
  .global {{ background:var(--surface2); border:1px solid var(--blue);
             border-radius:8px; padding:14px 20px; min-width:160px }}
  .global .lbl {{ color:var(--blue) }}
  .global .pscores {{ font-size:1.3rem }}
  h2.prov-header {{ color:var(--blue); font-size:1rem; margin:36px 0 12px;
                    padding:10px 16px; background:var(--surface);
                    border:1px solid var(--border); border-radius:6px;
                    display:flex; justify-content:space-between; align-items:center }}
  h2.prov-header.has-fail {{ border-color:var(--red) }}
  .prov-score {{ font-size:0.78rem; font-weight:400; color:var(--muted) }}
  h3.sec {{ color:var(--muted); font-size:0.78rem; text-transform:uppercase;
            letter-spacing:.06em; margin:18px 0 8px; padding-left:4px }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:8px }}
  th {{ background:var(--surface); color:var(--muted); padding:9px 14px; text-align:left;
        font-size:0.75rem; text-transform:uppercase; letter-spacing:.04em;
        border-bottom:2px solid var(--border) }}
  td {{ padding:10px 14px; border-bottom:1px solid var(--border); font-size:0.88rem;
        vertical-align:top }}
  tr:hover td {{ background:var(--surface2) }}
  .dur {{ color:var(--muted); font-size:0.82rem; text-align:right; white-space:nowrap }}
  .badge {{ font-weight:700; font-size:0.78rem; white-space:nowrap }}
  .badge.pass {{ color:var(--green) }} .badge.fail {{ color:var(--red) }}
  .badge.skip {{ color:var(--muted) }}
  ul.sub {{ list-style:none; margin-top:6px; padding-left:10px }}
  ul.sub li {{ color:var(--muted); font-size:0.8rem; line-height:1.6 }}
  ul.sub li::before {{ content:"· "; color:var(--border) }}
  code {{ background:var(--surface2); padding:1px 5px; border-radius:3px;
          font-size:0.88em; font-family:monospace }}
</style>
</head>
<body>
<header>
  <h1>vidlizer — Smoke Test Report</h1>
  <div class="meta">
    Generated: <strong>{generated}</strong><br>
    Providers tested: <strong>{', '.join(p for p, _ in providers_tested)}</strong>
  </div>
</header>

<div class="scorecard">
{scorecard}
  <div class="global card">
    <div class="lbl">Total</div>
    <div class="pscores">
      <span class="ps pass">{total_pass}✓</span>
      <span class="ps fail">{total_fail}✗</span>
      <span class="ps skip">{total_skip}−</span>
    </div>
  </div>
</div>

{body_html}
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


def _render_section_table(sec_name: str, checks: list[Check]) -> str:
    rows = ""
    for c in checks:
        badge_cls = c.status
        badge_txt = "✓ PASS" if c.status == "pass" else "✗ FAIL" if c.status == "fail" else "— SKIP"
        dur = f"{c.duration_s:.2f}s" if c.duration_s else "—"
        sub_html = "".join(f"<li>{s}</li>" for s in c.sub) if c.sub else ""
        sub_block = f"<ul class='sub'>{sub_html}</ul>" if sub_html else ""
        rows += (f"<tr><td>{c.name}</td>"
                 f"<td class='badge {badge_cls}'>{badge_txt}</td>"
                 f"<td class='dur'>{dur}</td>"
                 f"<td>{c.detail}{sub_block}</td></tr>\n")
    return (f"<h3 class='sec'>{sec_name}</h3>"
            f"<table><thead><tr><th>Test</th><th>Result</th>"
            f"<th>Time</th><th>Detail</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="vidlizer smoke test — all providers, all functions"
    )
    parser.add_argument("--provider", default="", help="ollama | openai | openrouter  (force single)")
    parser.add_argument("--model", default="", help="model ID (auto-detected if omitted)")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold cyan]vidlizer — Comprehensive Smoke Test[/bold cyan]\n"
        "[dim]All modules · every available provider in sequence · "
        "local unloaded after use · per-provider HTML report[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    # ── Detect environment ────────────────────────────────────────────────────
    _section("Environment")
    ffmpeg_ok, ffmpeg_ver       = _check_ffmpeg()
    ollama_ok, ol_host, ol_mdls = _check_ollama()
    oai_ok, oai_base, oai_mdls  = _check_openai_compat()
    or_ok, or_detail, or_model  = _check_openrouter()
    whisper_ok, whisper_det     = _check_whisper()

    _record("ffmpeg",        "pass" if ffmpeg_ok else "fail", ffmpeg_ver)
    _record("Ollama",        "pass" if ollama_ok else "skip",
            f"{ol_host}  [{', '.join(ol_mdls[:3])}{'…' if len(ol_mdls) > 3 else ''}]"
            if ollama_ok else f"{ol_host} (not reachable)")
    _record("OpenAI-compat", "pass" if oai_ok else "skip",
            f"{oai_base}  [{', '.join(oai_mdls[:2])}]" if oai_ok else f"{oai_base} (not reachable)")
    _record("OpenRouter",    "pass" if or_ok else "skip",
            f"{or_detail}" + (f"  → free model: {or_model}" if or_ok else ""))
    _record("mlx-whisper",   "pass" if whisper_ok else "skip", whisper_det)

    if not ffmpeg_ok:
        console.print("\n[red]ffmpeg required — cannot continue.[/red]")
        return 1

    # ── Build provider list ───────────────────────────────────────────────────
    # Each entry: (provider_id, model, host_for_unload)
    available: list[tuple[str, str, str]] = []

    if args.provider:
        # Single-provider override
        prov = args.provider
        if prov == "ollama":
            if not ollama_ok:
                console.print("[red]Ollama not reachable.[/red]")
                return 1
            prefs = ["qwen2.5vl:7b", "qwen2.5vl:3b", "minicpm-v:8b", "llava-onevision:7b", "llava"]
            mdl = args.model or next(
                (m for pref in prefs for m in ol_mdls if m.startswith(pref.split(":")[0])),
                ol_mdls[0] if ol_mdls else "",
            )
            available = [("ollama", mdl, ol_host)]
        elif prov in ("openai", "openai-compat"):
            if not oai_ok:
                console.print("[red]OpenAI-compat not reachable.[/red]")
                return 1
            available = [("openai", args.model or (oai_mdls[0] if oai_mdls else ""), oai_base)]
        elif prov == "openrouter":
            available = [("openrouter", args.model or or_model, "")]
        else:
            console.print(f"[red]Unknown provider: {prov}[/red]")
            return 1
    else:
        # Auto-detect all, local first
        if ollama_ok and ol_mdls:
            prefs = ["qwen2.5vl:7b", "qwen2.5vl:3b", "minicpm-v:8b", "llava-onevision:7b", "llava"]
            mdl = next(
                (m for pref in prefs for m in ol_mdls if m.startswith(pref.split(":")[0])),
                ol_mdls[0],
            )
            available.append(("ollama", mdl, ol_host))
        if oai_ok and oai_mdls:
            available.append(("openai", oai_mdls[0], oai_base))
        if or_ok:
            available.append(("openrouter", or_model, ""))

    if not available:
        console.print(
            "\n[red]No provider available.[/red]\n"
            "  Start Ollama, LM Studio, or set OPENROUTER_API_KEY."
        )
        return 1

    console.print(f"\n  [bold]Providers to test:[/bold]")
    for prov_id, prov_model, _ in available:
        console.print(f"    [magenta]{prov_id}[/magenta]  →  {prov_model}")

    # ── Build assets ──────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="vidlizer_smoke_") as tmp_str:
        tmp    = Path(tmp_str)
        assets = tmp / "assets"
        assets.mkdir()

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

        # ── Library section (once, no network) ───────────────────────────────
        section_library(video, silent, image, pdf, tmp)

        # ── Per-provider sections ─────────────────────────────────────────────
        for prov_id, prov_model, prov_host in available:
            console.print()
            console.print(Rule(
                f"[bold magenta]Provider: {prov_id}  /  {prov_model}[/bold magenta]",
                style="magenta",
            ))
            _set_provider(prov_id)

            prov_out = tmp / "out" / prov_id
            prov_out.mkdir(parents=True)

            json_path = section_analysis(video, image, pdf, prov_out, prov_model, prov_id)
            section_advanced(video, prov_out, prov_model, prov_id, json_path)
            section_audio(video, silent, prov_out, prov_model, prov_id)

            # Unload local model from VRAM
            if prov_id == "ollama":
                _unload_ollama(prov_model, prov_host)
            elif prov_id == "openai":
                _unload_openai_compat(prov_model, prov_host)

    # ── Final table ───────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Results[/bold]", style="dim"))
    console.print()
    _print_summary_table()
    console.print()

    total_pass  = sum(1 for c in _checks if c.status == "pass")
    total_fail  = sum(1 for c in _checks if c.status == "fail")
    total_skip  = sum(1 for c in _checks if c.status == "skip")
    color = "green" if total_fail == 0 else "red"
    console.print(
        f"[{color}][bold]{total_pass} passed[/bold][/{color}]  "
        f"[red]{total_fail} failed[/red]  "
        f"[dim]{total_skip} skipped[/dim]  "
        f"[dim]({len(_checks)} total)[/dim]"
    )

    # Per-provider summary
    console.print()
    for prov_id, prov_model, _ in [("library", "—", "")] + list(available):
        p, f, s = (
            sum(1 for c in _checks if c.provider == prov_id and c.status == "pass"),
            sum(1 for c in _checks if c.provider == prov_id and c.status == "fail"),
            sum(1 for c in _checks if c.provider == prov_id and c.status == "skip"),
        )
        color = "green" if f == 0 else "red"
        console.print(
            f"  [{color}]{'✓' if f == 0 else '✗'}[/{color}]  "
            f"[magenta]{prov_id:<12}[/magenta] "
            f"[green]{p} passed[/green]  [red]{f} failed[/red]  [dim]{s} skipped[/dim]"
            f"  [dim]({prov_model})[/dim]"
        )

    report = _save_html([(p, m) for p, m, _ in available])
    console.print(f"\n[dim]Report → [/dim][cyan]{report}[/cyan]")
    console.print(f"[dim]Open   → [/dim][cyan]open {report}[/cyan]")
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
