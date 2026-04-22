"""HTTP layer: streaming POST to Ollama and OpenAI-compatible endpoints."""
from __future__ import annotations

import json
import os
import time

import requests
from rich.console import Console
from rich.live import Live
from rich.text import Text

from vidlizer.models import get_pricing

_console = Console(stderr=True, highlight=False)

_live_models: list[dict] = []


def _model_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = get_pricing(model, _live_models or None)
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class CostCapExceeded(RuntimeError):
    """Raised when accumulated cost crosses the configured cap."""


class ImageLimitError(RuntimeError):
    """Raised when the model rejects the request due to too many images."""


class CostTracker:
    def __init__(self, max_cost: float = 0.0) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.batches_done = 0
        self.batches_total = 0
        self.max_cost = max(0.0, max_cost)

    def add(self, model: str, usage: dict) -> float:
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        c = _model_cost(model, pt, ct)
        self.cost_usd += c
        self.batches_done += 1
        if self.max_cost > 0 and self.cost_usd > self.max_cost:
            raise CostCapExceeded(
                f"cost cap ${self.max_cost:.2f} exceeded "
                f"(spent ~${self.cost_usd:.4f}) — aborting"
            )
        return c

    def summary(self) -> str:
        tok = f"{self.prompt_tokens:,}↑  {self.completion_tokens:,}↓"
        cost = f"~${self.cost_usd:.4f}" if self.cost_usd > 0 else "free"
        return f"tokens: {tok}  |  cost: {cost}"


def _post_ollama(
    model: str,
    payload: dict,
    timeout: int,
    verbose: bool,
    tracker: CostTracker | None,
    label: str,
    n_frames: int,
    endpoint: str | None,
) -> dict:
    """Post to Ollama native /api/chat (newline-delimited JSON streaming)."""
    native_messages = []
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if "," in url:
                        url = url.split(",", 1)[1]
                    images.append(url)
            native_msg: dict = {"role": msg["role"], "content": "\n".join(text_parts)}
            if images:
                native_msg["images"] = images
        else:
            native_msg = {"role": msg["role"], "content": str(content)}
        native_messages.append(native_msg)

    native_payload = {
        "model": model,
        "messages": native_messages,
        "stream": True,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
    }

    _url = endpoint or "http://localhost:11434/api/chat"
    payload_bytes = json.dumps(native_payload).encode()
    kb = len(payload_bytes) // 1024

    frame_info = f"  [white]{n_frames}f[/white]" if n_frames else ""
    _console.print(
        f"[cyan]→[/cyan] [bold]{label}[/bold]{frame_info}  "
        f"[dim]{kb} KB[/dim]  [dim]→[/dim]  [magenta]{model}[/magenta]"
    )
    if verbose:
        _console.print(f"[dim][debug] Ollama native POST {_url} {kb} KB[/dim]", markup=False)

    r = requests.post(
        _url,
        headers={"Content-Type": "application/json"},
        data=payload_bytes,
        timeout=timeout,
        stream=True,
    )

    if not r.ok:
        raise RuntimeError(f"Ollama {r.status_code}: {r.text[:500]}")

    full_content = ""
    usage: dict = {}
    _start = time.time()
    _chars = 0

    def _ollama_live() -> Text:
        t = Text()
        t.append(f"   {label}", style="bold")
        if n_frames:
            t.append(f"  {n_frames}f", style="white")
        t.append(f"  {time.time() - _start:.1f}s", style="dim")
        if _chars:
            t.append(f"  {_chars:,} chars", style="dim")
        return t

    with Live(_ollama_live(), console=_console, refresh_per_second=4, transient=True) as live:
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if chunk.get("error"):
                raise RuntimeError(f"Ollama error: {chunk['error']}")

            piece = chunk.get("message", {}).get("content", "")
            if piece:
                full_content += piece
                _chars += len(piece)
                live.update(_ollama_live())

            if chunk.get("done"):
                usage = {
                    "prompt_tokens": chunk.get("prompt_eval_count", 0) or 0,
                    "completion_tokens": chunk.get("eval_count", 0) or 0,
                }

    if tracker:
        tracker.add(model, usage)

    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    _console.print(
        f"   [dim]tokens:[/dim] [cyan]{pt:,}[/cyan][dim]↑[/dim]  "
        f"[cyan]{ct:,}[/cyan][dim]↓[/dim]  [cyan]free[/cyan]"
    )
    if verbose:
        _console.print(f"[dim][debug] Ollama stream done, {len(full_content)} chars[/dim]", markup=False)

    return {"choices": [{"message": {"content": full_content}}], "usage": usage}


def post(
    api_key: str | None,
    model: str,
    payload: dict,
    timeout: int,
    verbose: bool,
    tracker: CostTracker | None = None,
    label: str = "",
    n_frames: int = 0,
    endpoint: str | None = None,
    req_headers: dict | None = None,
    is_ollama: bool = False,
    no_stream_opts: bool = False,
) -> dict:
    if is_ollama:
        return _post_ollama(model, payload, timeout, verbose, tracker, label, n_frames, endpoint)

    if no_stream_opts:
        stream_payload = {**payload, "stream": True}
    else:
        stream_payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    payload_bytes = json.dumps(stream_payload).encode()
    kb = len(payload_bytes) // 1024

    frame_info = f"  [white]{n_frames}f[/white]" if n_frames else ""
    _console.print(
        f"[cyan]→[/cyan] [bold]{label}[/bold]{frame_info}  "
        f"[dim]{kb} KB[/dim]  [dim]→[/dim]  [magenta]{model}[/magenta]"
    )
    if verbose:
        _console.print(f"[dim][debug] POST stream=True payload={kb} KB[/dim]", markup=False)

    _url = endpoint or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    _headers = req_headers or {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/arizawan/vidlizer",
        "X-Title": "vidlizer",
    }
    r = requests.post(
        _url,
        headers=_headers,
        data=payload_bytes,
        timeout=timeout,
        stream=True,
    )

    if r.status_code == 429:
        raise RuntimeError(f"rate_limited: {r.text[:500]}")
    if not r.ok:
        err_text = r.text[:1000]
        if "image" in err_text.lower() and ("most" in err_text.lower() or "limit" in err_text.lower()):
            raise ImageLimitError(err_text)
        raise RuntimeError(f"API {r.status_code}: {err_text}")

    full_content = ""
    usage: dict = {}
    _start = time.time()
    _chars = 0

    def _live_text() -> Text:
        t = Text()
        t.append(f"   {label}", style="bold")
        if n_frames:
            t.append(f"  {n_frames}f", style="white")
        t.append(f"  {time.time() - _start:.1f}s", style="dim")
        if _chars:
            t.append(f"  {_chars:,} chars", style="dim")
        return t

    with Live(_live_text(), console=_console, refresh_per_second=4, transient=True) as live:
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = chunk["usage"]

            if "error" in chunk:
                err = chunk["error"]
                err_str = json.dumps(err) if isinstance(err, dict) else str(err)
                if "image" in err_str.lower() and ("most" in err_str.lower() or "limit" in err_str.lower()):
                    raise ImageLimitError(err_str)
                raise RuntimeError(f"API error: {err_str}")

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    full_content += piece
                    _chars += len(piece)
                    live.update(_live_text())

    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    if tracker:
        batch_cost = tracker.add(model, usage)
    else:
        batch_cost = _model_cost(model, pt, ct)

    cost_str = f"[green]~${batch_cost:.4f}[/green]" if batch_cost > 0 else "[cyan]free[/cyan]"
    _console.print(
        f"   [dim]tokens:[/dim] [cyan]{pt:,}[/cyan][dim]↑[/dim]  "
        f"[cyan]{ct:,}[/cyan][dim]↓[/dim]  {cost_str}"
    )

    if verbose:
        _console.print(f"[dim][debug] stream complete, content={len(full_content)} chars[/dim]", markup=False)

    return {
        "choices": [{"message": {"content": full_content}}],
        "usage": usage,
    }
