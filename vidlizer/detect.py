"""Provider and dependency detection helpers — read-only, no side effects."""
from __future__ import annotations

import os
import shutil
import subprocess

import requests

VISION_HINTS: frozenset[str] = frozenset([
    "vl", "vision", "llava", "minicpm-v", "visual", "clip",
    "pixtral", "idefics", "cogvlm", "internvl", "phi-3-v",
    "molmo", "qwen2.5vl", "qwen3vl", "gemma-3",
])

OL_PREFS: list[str] = ["qwen2.5vl:7b", "qwen2.5vl:3b", "minicpm-v:8b", "llava-onevision:7b", "llava"]

OLLAMA_MINIMAL   = "qwen2.5vl:3b"
LMSTUDIO_MINIMAL = "mlx-community/Qwen2.5-VL-3B-Instruct-8bit"
OMLX_MINIMAL     = "mlx-community/Qwen2.5-VL-3B-Instruct-8bit"

_OR_FREE_CANDIDATES = [
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "meta-llama/llama-4-scout:free",
]


def is_vision_model(mid: str) -> bool:
    m = mid.lower()
    return any(h in m for h in VISION_HINTS)


def pick_best_vision(models: list[str], prefs: list[str]) -> str | None:
    """Return preferred vision model, then any vision model, then None."""
    for pref in prefs:
        for m in models:
            if m.lower().startswith(pref.split(":")[0].lower()):
                return m
    return next((m for m in models if is_vision_model(m)), None)


def check_ffmpeg() -> tuple[bool, str]:
    """Return (ok, version_string)."""
    if not shutil.which("ffmpeg"):
        return False, "not found"
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        line = r.stdout.split("\n")[0]
        ver = line.split("version ")[1].split(" ")[0] if "version " in line else "?"
        return True, ver
    except Exception:
        return True, "?"


def check_ollama(host: str | None = None) -> tuple[bool, str, list[str]]:
    """Return (ok, host_url, model_list)."""
    try:
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, host, models
    except Exception:
        return False, host or os.getenv("OLLAMA_HOST", "http://localhost:11434"), []


def _probe_openai_compat(base: str, key: str = "local") -> tuple[bool, list[str]]:
    try:
        r = requests.get(f"{base.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=3)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return True, models
    except Exception:
        return False, []


def check_lmstudio(base: str | None = None) -> tuple[bool, str, list[str]]:
    """Return (ok, base_url, model_list)."""
    base = base or "http://localhost:1234/v1"
    ok, mdls = _probe_openai_compat(base, os.getenv("OPENAI_API_KEY", "lm-studio"))
    return ok, base, mdls


def check_omlx(base: str | None = None) -> tuple[bool, str, list[str]]:
    """Return (ok, base_url, model_list)."""
    base = base or "http://localhost:8000/v1"
    ok, mdls = _probe_openai_compat(base, "local")
    return ok, base, mdls


def check_openrouter(key: str | None = None) -> tuple[bool, str, str]:
    """Return (ok, detail, free_model_id)."""
    key = key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return False, "not set", ""
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"}, timeout=8,
        )
        r.raise_for_status()
        live_ids = {m["id"] for m in r.json().get("data", [])}
        for candidate in _OR_FREE_CANDIDATES:
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
    return True, f"key ...{key[-4:]}", "google/gemma-3-27b-it:free"


def check_whisper() -> tuple[bool, str]:
    """Return (ok, status_string)."""
    try:
        import mlx_whisper  # noqa: F401
        return True, "installed"
    except ImportError:
        return False, "not installed"
