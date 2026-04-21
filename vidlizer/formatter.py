"""Output formatters: json, summary, markdown."""
from __future__ import annotations

import json


def to_json(data: dict, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)


def to_summary(data: dict) -> str:
    """Flatten flow into a plain-text narrative grouped by phase."""
    flow = data.get("flow", [])
    if not flow:
        return "No steps found."

    seen: list[str] = []
    for step in flow:
        p = step.get("phase") or "General"
        if p not in seen:
            seen.append(p)

    paragraphs: list[str] = []
    for phase in seen:
        phase_steps = [s for s in flow if (s.get("phase") or "General") == phase]
        actions = [s["action"] for s in phase_steps if s.get("action")]
        if actions:
            paragraphs.append(f"{phase}: {' '.join(actions)}")

    if transcript := data.get("transcript"):
        full = " ".join(s.get("text", "").strip() for s in transcript if s.get("text"))
        if full:
            paragraphs.append(f"Transcript: {full}")

    return "\n\n".join(paragraphs)


def to_markdown(data: dict) -> str:
    """Convert flow to a readable Markdown document."""
    flow = data.get("flow", [])
    parts: list[str] = ["# Video Analysis\n"]

    for step in flow:
        ts = step.get("timestamp_s")
        ts_str = f" · {ts:.1f}s" if ts is not None else ""
        phase = step.get("phase", "")
        num = step.get("step", "?")

        heading = f"## Step {num}{ts_str}"
        if phase:
            heading += f" — {phase}"
        parts.append(heading + "\n")

        if scene := step.get("scene"):
            parts.append(f"**Scene**: {scene}\n")
        if action := step.get("action"):
            parts.append(f"**Action**: {action}\n")
        if subjects := step.get("subjects"):
            if isinstance(subjects, list) and subjects:
                parts.append(f"**Subjects**: {', '.join(str(s) for s in subjects)}\n")
        if text := step.get("text_visible"):
            parts.append(f"**Text on screen**: `{text}`\n")
        if obs := step.get("observations"):
            parts.append(f"**Observations**: {obs}\n")
        if speech := step.get("speech"):
            parts.append(f"> *{speech}*\n")

    if transcript := data.get("transcript"):
        parts.append("\n---\n\n## Full Transcript\n")
        for seg in transcript:
            t = seg.get("start", 0)
            text = seg.get("text", "").strip()
            if text:
                parts.append(f"[{t:.1f}s] {text}  ")

    return "\n".join(parts)


def format_output(data: dict, fmt: str) -> str:
    if fmt == "markdown":
        return to_markdown(data)
    if fmt == "summary":
        return to_summary(data)
    return to_json(data)
