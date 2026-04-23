"""Persistent per-run usage tracking (tokens, cost, model)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_USAGE_PATH = Path.home() / ".cache" / "vidlizer" / "usage.jsonl"


def record_run(
    *,
    model: str,
    provider: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    source: str,
    steps: int,
) -> None:
    """Append one usage record. Silent on failure (never crash the main run)."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return  # never pollute usage stats with test runs
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6),
        "source": os.path.basename(source),
        "steps": steps,
    }
    try:
        _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _USAGE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def get_stats() -> dict:
    """Read all usage records and return aggregated statistics."""
    records: list[dict] = []
    if _USAGE_PATH.exists():
        for line in _USAGE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_model: dict[str, dict] = defaultdict(lambda: {
        "runs": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
        "steps": 0, "provider": "",
    })

    for r in records:
        m = r.get("model", "unknown")
        by_model[m]["runs"] += 1
        by_model[m]["tokens_in"] += r.get("tokens_in", 0)
        by_model[m]["tokens_out"] += r.get("tokens_out", 0)
        by_model[m]["cost_usd"] = round(
            by_model[m]["cost_usd"] + r.get("cost_usd", 0.0), 6
        )
        by_model[m]["steps"] += r.get("steps", 0)
        by_model[m]["provider"] = r.get("provider", "")

    model_rows = sorted(
        [{"model": k, **v} for k, v in by_model.items()],
        key=lambda x: x["runs"],
        reverse=True,
    )

    total_cost = sum(r.get("cost_usd", 0.0) for r in records)
    return {
        "total_runs": len(records),
        "total_cost_usd": round(total_cost, 6),
        "total_tokens_in": sum(r.get("tokens_in", 0) for r in records),
        "total_tokens_out": sum(r.get("tokens_out", 0) for r in records),
        "total_steps": sum(r.get("steps", 0) for r in records),
        "by_model": model_rows,
        "log_path": str(_USAGE_PATH),
    }


def clear_stats() -> int:
    """Delete the usage log. Returns number of records deleted."""
    if not _USAGE_PATH.exists():
        return 0
    lines = [ln for ln in _USAGE_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    count = len(lines)
    _USAGE_PATH.unlink()
    return count
