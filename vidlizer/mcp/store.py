"""Disk-backed analysis registry for the MCP server."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_STORE_DIR = Path.home() / ".cache" / "vidlizer" / "analyses"


def _dir() -> Path:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR


def make_id(source: str, params: dict) -> str:
    """Stable 16-char hex id derived from source + params."""
    key = json.dumps({"source": source, **params}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def exists(aid: str) -> bool:
    return (_dir() / f"{aid}.json").exists()


def save(aid: str, source: str, params: dict, data: dict) -> None:
    flow = data.get("flow", [])
    phases = sorted({s.get("phase", "") for s in flow if s.get("phase")})
    record = {
        "id": aid,
        "source": source,
        "params": params,
        "created_at": time.time(),
        "step_count": len(flow),
        "phases": phases,
        "has_transcript": bool(data.get("transcript")),
        "duration_s": _last_timestamp(flow),
        "data": data,
    }
    (_dir() / f"{aid}.json").write_text(json.dumps(record, indent=2))


def load(aid: str) -> dict | None:
    p = _dir() / f"{aid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_all() -> list[dict]:
    """Return lightweight meta list (no data payloads)."""
    results = []
    for f in sorted(_dir().glob("*.json")):
        try:
            rec = json.loads(f.read_text())
            results.append({
                "id": rec["id"],
                "source": rec.get("source", ""),
                "step_count": rec.get("step_count", 0),
                "phases": rec.get("phases", []),
                "has_transcript": rec.get("has_transcript", False),
                "duration_s": rec.get("duration_s"),
                "created_at": rec.get("created_at", 0),
            })
        except Exception:
            continue
    return sorted(results, key=lambda x: x["created_at"], reverse=True)


def delete(aid: str) -> bool:
    p = _dir() / f"{aid}.json"
    if p.exists():
        p.unlink()
        return True
    return False


def _last_timestamp(flow: list[dict]) -> float | None:
    for step in reversed(flow):
        ts = step.get("timestamp_s")
        if ts is not None:
            try:
                return float(ts)
            except (TypeError, ValueError):
                pass
    return None
