#!/usr/bin/env python3
"""In-memory result cache with 10-minute TTL."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_STORE: dict[str, tuple[dict, float]] = {}
_TTL = 600  # seconds


def _key(path: Path | None, params: dict) -> str:
    data = {k: str(v) for k, v in sorted(params.items())}
    if path and path.exists():
        st = path.stat()
        data["_path"] = str(path.resolve())
        data["_mtime"] = str(st.st_mtime)
        data["_size"] = str(st.st_size)
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def get(path: Path | None, params: dict) -> dict | None:
    k = _key(path, params)
    entry = _STORE.get(k)
    if entry:
        val, ts = entry
        if time.time() - ts < _TTL:
            return val
        del _STORE[k]
    return None


def put(path: Path | None, params: dict, result: dict) -> None:
    _STORE[_key(path, params)] = (result, time.time())


def clear() -> None:
    _STORE.clear()
