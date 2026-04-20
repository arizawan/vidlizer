#!/usr/bin/env python3
"""Fetch and cache OpenRouter vision-capable models with live pricing."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

_CACHE_PATH = Path.home() / ".cache" / "vidlizer" / "models.json"
_CACHE_TTL = 3600  # seconds

# Fallback list if API is unreachable or key is missing
_FALLBACK: list[dict] = [
    {"id": "google/gemini-2.5-flash",             "name": "Gemini 2.5 Flash",          "input_usd_per_1m": 0.15,  "output_usd_per_1m": 0.60,  "free": False, "rate_limited": False, "per_req_limit_tokens": None, "context_length": 1048576},
    {"id": "google/gemini-2.5-flash-lite",        "name": "Gemini 2.5 Flash Lite",     "input_usd_per_1m": 0.075, "output_usd_per_1m": 0.30,  "free": False, "rate_limited": False, "per_req_limit_tokens": None, "context_length": 1048576},
    {"id": "google/gemini-2.5-pro",               "name": "Gemini 2.5 Pro",            "input_usd_per_1m": 1.25,  "output_usd_per_1m": 10.0,  "free": False, "rate_limited": False, "per_req_limit_tokens": None, "context_length": 1048576},
    {"id": "nvidia/nemotron-nano-12b-v2-vl:free", "name": "Nemotron Nano 12B",         "input_usd_per_1m": 0.0,   "output_usd_per_1m": 0.0,   "free": True,  "rate_limited": True,  "per_req_limit_tokens": 8192, "context_length": 131072},
    {"id": "google/gemma-4-31b-it:free",          "name": "Gemma 4 31B",               "input_usd_per_1m": 0.0,   "output_usd_per_1m": 0.0,   "free": True,  "rate_limited": True,  "per_req_limit_tokens": None, "context_length": 131072},
    {"id": "openai/gpt-4o",                       "name": "GPT-4o",                    "input_usd_per_1m": 2.50,  "output_usd_per_1m": 10.0,  "free": False, "rate_limited": False, "per_req_limit_tokens": None, "context_length": 128000},
    {"id": "openai/gpt-4o-mini",                  "name": "GPT-4o Mini",               "input_usd_per_1m": 0.15,  "output_usd_per_1m": 0.60,  "free": False, "rate_limited": False, "per_req_limit_tokens": None, "context_length": 128000},
]


def _parse_per_req_limit(raw: dict | None) -> int | None:
    """Return the smaller of prompt/completion token limits, or None."""
    if not raw:
        return None
    try:
        vals = [int(v) for v in raw.values() if v]
        return min(vals) if vals else None
    except (TypeError, ValueError):
        return None


def _is_vision(model: dict) -> bool:
    arch = model.get("architecture", {})
    modalities = arch.get("input_modalities") or arch.get("modality", "")
    if isinstance(modalities, list):
        return "image" in modalities
    return "image" in str(modalities)


def _parse_price(raw: str | float | None) -> float:
    """OpenRouter returns prices as strings like '0.00000015' (per token) or floats."""
    if raw is None:
        return 0.0
    try:
        per_token = float(raw)
        return per_token * 1_000_000  # convert to per-million
    except (TypeError, ValueError):
        return 0.0


def fetch_models(api_key: str | None = None, force_refresh: bool = False) -> list[dict]:
    """Return list of vision models with pricing. Uses 1h disk cache."""
    if not force_refresh and _CACHE_PATH.exists():
        age = time.time() - _CACHE_PATH.stat().st_mtime
        if age < _CACHE_TTL:
            try:
                cached = json.loads(_CACHE_PATH.read_text())
                if isinstance(cached, list) and cached:
                    return cached
            except (json.JSONDecodeError, OSError):
                pass

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return _FALLBACK

    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        r.raise_for_status()
        raw_models: list[dict] = r.json().get("data", [])
    except Exception:
        return _FALLBACK

    vision = []
    for m in raw_models:
        if not _is_vision(m):
            continue
        pricing = m.get("pricing", {})
        model_id: str = m.get("id", "")
        is_free = model_id.endswith(":free") or (
            _parse_price(pricing.get("prompt")) == 0.0
            and _parse_price(pricing.get("completion")) == 0.0
        )
        per_req = m.get("per_request_limits")
        vision.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "input_usd_per_1m": _parse_price(pricing.get("prompt")),
            "output_usd_per_1m": _parse_price(pricing.get("completion")),
            "free": is_free,
            "rate_limited": is_free,  # all free-tier models carry OpenRouter RPM/RPD limits
            "per_req_limit_tokens": _parse_per_req_limit(per_req),
            "context_length": m.get("context_length", 0),
        })

    if not vision:
        return _FALLBACK

    # Sort: free first, then by input price ascending
    vision.sort(key=lambda m: (not m["free"], m["input_usd_per_1m"]))

    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(vision, indent=2))
    except OSError:
        pass

    return vision


def get_pricing(model_id: str, models: list[dict] | None = None) -> tuple[float, float]:
    """Return (input_usd_per_1m, output_usd_per_1m) for a model id."""
    if models is None:
        models = _FALLBACK
    for m in models:
        if m["id"] == model_id:
            return m["input_usd_per_1m"], m["output_usd_per_1m"]
    # Prefix match fallback
    for m in models:
        if model_id.startswith(m["id"].split(":")[0]):
            return m["input_usd_per_1m"], m["output_usd_per_1m"]
    return 0.0, 0.0


def get_cheapest_paid(models: list[dict] | None = None) -> str:
    """Return the ID of the cheapest non-free vision model."""
    pool = models or _FALLBACK
    paid = [m for m in pool if not m.get("free")]
    if not paid:
        return "google/gemini-2.5-flash"
    return min(paid, key=lambda m: m["input_usd_per_1m"])["id"]


def format_model_line(m: dict) -> str:
    """One-line label for model picker: id  [price  ctx  RL-flag]."""
    parts: list[str] = []

    if m["free"]:
        parts.append("free")
    else:
        inp = m["input_usd_per_1m"]
        out = m["output_usd_per_1m"]
        parts.append(f"${inp:.3f}/${out:.3f} per M")

    ctx = m.get("context_length") or 0
    if ctx >= 1_000_000:
        parts.append(f"{ctx // 1_000_000}M ctx")
    elif ctx >= 1_000:
        parts.append(f"{ctx // 1_000}K ctx")

    if m.get("rate_limited"):
        limit = m.get("per_req_limit_tokens")
        rl = f"⚡ RL ({limit // 1000}K/req)" if limit else "⚡ rate-limited"
        parts.append(rl)

    return f"{m['id']}  [{', '.join(parts)}]"


def format_price_label(m: dict) -> str:
    """Short price-only label (kept for backwards compat)."""
    if m["free"]:
        return "free"
    inp = m["input_usd_per_1m"]
    out = m["output_usd_per_1m"]
    if inp < 0.10:
        return f"~${inp:.3f}/M in"
    return f"${inp:.2f}/${out:.2f} per M in/out"
