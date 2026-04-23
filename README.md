<div align="center">

# vidlizer

**Point it at a video, image, or PDF. Get structured JSON — scene by scene.**

[![PyPI](https://img.shields.io/pypi/v/vidlizer.svg)](https://pypi.org/project/vidlizer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](#requirements)
[![CI](https://github.com/arizawan/vidlizer/actions/workflows/ci.yml/badge.svg)](https://github.com/arizawan/vidlizer/actions)
[![Tests](https://img.shields.io/badge/tests-248%20passing-brightgreen.svg)](#testing)

![demo](https://raw.githubusercontent.com/arizawan/vidlizer/main/assets/demo.gif)

</div>

---

vidlizer pulls frames out of any video, image, or PDF using ffmpeg, sends them to a vision LLM, and returns a `flow` array — one entry per scene. Each entry tells you what happened, who was on screen, what text was visible, and what changed. If the video has audio, it transcribes it with Apple MLX Whisper and merges the speech into each step.

Runs fully local via [Ollama](https://ollama.com) or any OpenAI-compatible server (LM Studio, vLLM, oMLX) — no API key, no data leaving your machine. Or connect [OpenRouter](https://openrouter.ai) for cloud models. `vidlizer setup` detects what you have installed and writes your config in under a minute.

```bash
vidlizer demo.mp4
vidlizer "https://youtube.com/watch?v=..."
vidlizer screenshot.png
vidlizer document.pdf
```

---

## ✨ Features

- **Any input** — local video, image (jpg/png/webp/…), PDF, or URL (YouTube, Loom, Vimeo, Twitter)
- **4 providers** — Ollama (fully offline), LM Studio (port 1234), oMLX (Apple Silicon, port 8000), OpenRouter (cloud) — auto-detected in that order
- **Cross-provider fallback** — primary model fails → automatically switches provider (e.g. oMLX → OpenRouter)
- **JSON repair** — malformed model output is re-sent to the model to fix before skipping; recovers from partial JSON
- **Free-model guard** — `:free` OpenRouter models auto-force `concurrency=1` to stay within rate limits
- **3 output formats** — `--format json` (default), `summary` (plain text by phase), `markdown` (step-per-section doc)
- **Usage tracking** — `--stats` shows per-model token + cost breakdown across all runs; `get_usage_stats()` MCP tool
- **Auto transcript** — detects audio, transcribes with Apple MLX Whisper (Neural Engine), merges speech into each flow step
- **Perceptual dedup** — removes near-duplicate frames before sending (saves tokens)
- **analyze_moment** — `--start`/`--end` flags to focus on a time range
- **In-memory cache** — repeat runs on the same file skip the API call
- **Cost guard** — aborts if spend exceeds `MAX_COST_USD` (default $1.00)
- **Live progress** — Rich streaming indicator shows elapsed time and token count per batch
- **MCP server** — use from Claude Code, Cursor, Claude Desktop; provider/model locked via env vars; result includes `model_used` + `provider_used`
- **Auto-install** — missing `ffmpeg` is brew-installed; `mlx-whisper` is pip-installed on first audio video
- **Mac-native** — file picker dialog, Apple MLX transcription, handles macOS Unicode filenames (e.g. "11:26 AM")

---

## 📦 Requirements

- macOS (Apple Silicon recommended for transcription speed)
- Python 3.10+
- **Ollama mode**: [Ollama](https://ollama.com) installed + a vision model pulled (5 GB+ RAM)
- **LM Studio mode**: [LM Studio](https://lmstudio.ai) 0.3.16+ with a vision model loaded
- **Cloud mode**: An [OpenRouter API key](https://openrouter.ai/keys)

`ffmpeg` is installed automatically via Homebrew on first run if missing.

---

## 🚀 Install

### Option 1 — pipx (recommended, isolated)

```bash
pipx install vidlizer
vidlizer setup    # interactive wizard: detects providers, writes .env
```

### Option 2 — pip / virtualenv

```bash
pip install vidlizer
vidlizer setup
```

### Option 3 — from source

```bash
git clone https://github.com/arizawan/vidlizer.git
cd vidlizer
python -m venv .venv && source .venv/bin/activate
pip install -e .
vidlizer setup    # or: cp env.sample .env
```

### First-run wizard

`vidlizer setup` detects all installed providers, lets you pick primary + fallback, and writes a `.env` for you. It also offers to pull a vision model for Ollama if none is installed.

```
$ vidlizer setup
  Detected providers:
    1.  Ollama        → qwen2.5vl:3b
    2.  OpenRouter    → google/gemma-3-27b-it:free

  Primary provider (1–2): 1
  Fallback (1–1, Enter to skip): 2

✓  .env written → /your/project/.env
```

### Health check

```bash
vidlizer doctor   # shows ffmpeg, .env, provider, model status
```

### Manual provider setup

**Ollama** (fully offline, no API key):

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5vl:3b   # ~3.2 GB, requires 5 GB+ RAM (recommended)
ollama pull qwen2.5vl:7b   # ~6.0 GB, requires 10 GB+ RAM (best quality)
```

**LM Studio** (GPU-accelerated local inference):

```bash
# In LM Studio: load a vision model (e.g. Qwen2.5-VL 7B), enable the local server
# Set PROVIDER=openai and OPENAI_BASE_URL=http://localhost:1234/v1 in .env
```

**OpenRouter** (cloud):

```bash
# Paste your OpenRouter key in .env:
# OPENROUTER_API_KEY=sk-or-v1-...
```

---

## ⚡ Quick start

```bash
# Ollama — fully local, no API key
vidlizer demo.mp4 --provider ollama --model qwen2.5vl:3b

# LM Studio (or any OpenAI-compat server)
vidlizer demo.mp4 --provider openai --model qwen/qwen2.5-vl-7b-instruct

# OpenRouter (cloud)
vidlizer demo.mp4 --provider openrouter --model google/gemini-2.5-flash

# Analyze a YouTube video
vidlizer "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Analyze a single image or PDF
vidlizer screenshot.png
vidlizer report.pdf

# Focus on a time range
vidlizer demo.mp4 --start 30 --end 90

# Output as Markdown or plain-text summary
vidlizer demo.mp4 --format markdown -o result.md
vidlizer demo.mp4 --format summary -o result.txt
```

Run with no arguments for an interactive file picker + provider/model selector.

---

## 📄 Output

Three formats via `--format`:

| Format | Flag | Default extension | Description |
|---|---|---|---|
| JSON | `--format json` | `.analysis.json` | Full structured flow array (default) |
| Markdown | `--format markdown` | `.analysis.md` | Step-per-section document with scene/action/speech |
| Summary | `--format summary` | `.analysis.txt` | Plain text grouped by phase |

Default output path is `<normalized-name>.analysis.json` (or matching extension), or pass `-o path`.

```json
{
  "flow": [
    {
      "step": 1,
      "timestamp_s": 0.0,
      "phase": "Introduction",
      "scene": "Title card with product logo on dark background.",
      "subjects": ["Logo", "Text overlay"],
      "action": "Static title card displayed.",
      "text_visible": "vidlizer — analyze any video",
      "context": "Opening of a product demo.",
      "observations": "Clean minimal design.",
      "next_scene": "Screen recording of the CLI.",
      "speech": "Welcome to the vidlizer demo."
    }
  ],
  "transcript": [
    { "start": 0.0, "end": 2.4, "text": "Welcome to the vidlizer demo." }
  ],
  "model_used": "gemma-4-E2B-it-MLX-4bit",
  "provider_used": "openai"
}
```

`model_used` and `provider_used` reflect the model that actually produced the result — including after fallback. Surfaced in MCP `analyze_video` response so agents know which provider ran.

### Flow step fields

| Field | Description |
|---|---|
| `step` | Sequential integer |
| `timestamp_s` | Approximate time in seconds (from frame label) |
| `phase` | Logical section — Introduction, Demo, Action, Conclusion… |
| `scene` | What is currently visible |
| `subjects` | Key people, objects, UI elements present |
| `action` | What is happening — interaction, movement, narration, event |
| `text_visible` | All readable text on screen |
| `context` | Persistent state — timer, score, topic, brand… |
| `observations` | Errors, anomalies, emotions, key facts |
| `next_scene` | Brief description of what follows |
| `speech` | Transcript text spoken during this step (audio videos only) |

---

## 🤖 Models

All providers send **one frame per request** (batch_size=1) for maximum compatibility with context-limited models. Thinking-mode output (`<think>` tags) is stripped automatically.

### Local — Ollama

Fully offline, no API key. Install with `ollama pull <name>`:

| Model | Disk | RAM | Notes |
|---|---|---|---|
| `qwen2.5vl:3b` ★ | 3.2 GB | ~5 GB | **Recommended** — 128K ctx, strong JSON, multi-image |
| `qwen2.5vl:7b` ★ | 6.0 GB | ~9 GB | Best Ollama quality — 128K ctx, needs 10+ GB RAM |
| `minicpm-v:8b` | 5.5 GB | ~8 GB | Strong OCR + visual reasoning, 32K ctx |
| `llava-onevision:7b` | 5.5 GB | ~8 GB | Strong multi-image + video frames, reliable JSON |

Fallback order (if configured model unavailable): `qwen2.5vl:7b` → `qwen2.5vl:3b` → `minicpm-v:8b` → `llava-onevision:7b` → `llava:13b` → `llava:7b`

Uses Ollama's native `/api/chat` with `format: json` for reliable structured output.

### Local — LM Studio / oMLX / vLLM / LocalAI (OpenAI-compatible)

GPU-accelerated inference via any OpenAI-compatible server. Set `PROVIDER=openai` and point `OPENAI_BASE_URL` at your server.

| Model | VRAM | Notes |
|---|---|---|
| `qwen/qwen2.5-vl-7b-instruct` ★ | ~8 GB | **Recommended** — 128K ctx, reliable JSON, multi-image |
| `qwen/qwen3-vl-8b` ★ | ~10 GB | Latest Qwen vision — thinking tags stripped automatically |
| `qwen/qwen2.5-vl-3b-instruct` | ~5 GB | Lightweight — fast, 5–6 GB VRAM |
| `google/gemma-4-e4b-it` | ~6 GB | Google MoE — LM Studio 0.3.16+ native support |
| `google/gemma-4-9b-it` | ~10 GB | Stronger Gemma 4 — 128K ctx, better instruction following |
| `zai-org/glm-4.6v-flash` | ~8 GB | ZhipuAI MoE — 128K ctx, strong JSON, low latency |
| `openbmb/minicpm-v-4.5` | ~8 GB | 8B Qwen3-based — strong OCR, multi-image, vLLM ready |

**oMLX** (Apple Silicon native, [omlx.ai](https://omlx.ai)) — MLX-format models from HuggingFace, **auto-detected on port 8000** (distinct from LM Studio). Model IDs are HuggingFace paths:

| oMLX Model | Unified RAM | Notes |
|---|---|---|
| `mlx-community/Qwen2.5-VL-7B-Instruct-8bit` ★ | ~8 GB | Best Apple Silicon pick — fast, strong JSON |
| `mlx-community/Qwen2.5-VL-3B-Instruct-8bit` | ~4 GB | Lightweight — 4–5 GB RAM |
| `mlx-community/Qwen3-VL-8B-8bit` | ~9 GB | Latest Qwen vision — thinking tags stripped |
| `mlx-community/MiniCPM-V-2_6-8bit` | ~8 GB | Strong OCR + reasoning |

Model IDs are as shown in LM Studio's model browser, oMLX's admin panel, or your vLLM config. LM Studio / oMLX serve one model at a time (no fallback needed). vLLM with multiple models loaded uses the same fallback sequence.

Fallback fragment order (vLLM / oMLX with multiple models): `qwen2.5-vl-7b` → `qwen2.5-vl-3b` → `qwen3-vl` → `gemma-4` → `glm-4` → `minicpm-v` → `llava-onevision` → `llava`

### Cloud — OpenRouter

Models fetched live with current pricing. Run `vidlizer --list-models` to see the live list.

| Model | Input / 1M tokens | Notes |
|---|---|---|
| `google/gemini-2.5-flash` ★ | $0.15 | **Recommended** — 1M ctx, fast, accurate |
| `google/gemini-2.5-flash-lite` | $0.075 | Cheaper — slightly less accurate |
| `google/gemini-2.5-pro` | $1.25 | Best quality, higher cost |
| `openai/gpt-4o` | $2.50 | OpenAI flagship |
| `openai/gpt-4o-mini` | $0.15 | OpenAI budget option |
| `nvidia/nemotron-nano-12b-v2-vl:free` | free ⚡ | Rate-limited (8K/req), 128K ctx |
| `google/gemma-4-31b-it:free` | free ⚡ | Rate-limited, 128K ctx |

Free models auto-fallback to the cheapest available paid model on failure.

---

## 🎙️ Transcription

For videos with an audio track, vidlizer automatically:

1. Extracts a mono 16kHz WAV with ffmpeg
2. Transcribes with **Apple MLX Whisper** (Neural Engine + GPU on M-series)
3. Merges each transcript segment into the nearest flow step as `speech`

On first use, `mlx-whisper` is pip-installed and the base model (~150 MB) is downloaded once.

To opt out: `--no-transcript`

---

## 🛠️ CLI reference

```
vidlizer [video] [options]

positional:
  video                 Path to file or URL (YouTube/Loom/Vimeo/Twitter)

options:
  -o, --output PATH     Output path (default: <name>.analysis.json/.md/.txt)
  --format FORMAT       Output format: json (default), summary, markdown
  --provider PROVIDER   ollama | openai | openrouter  (openai covers LM Studio + oMLX + vLLM)
  --model MODEL         Model slug — Ollama name, OpenAI-compat ID, or OpenRouter slug
  --max-frames N        Max frames to send (default 60, hard cap 200)
  --start SECONDS       Analyze from this timestamp
  --end SECONDS         Analyze up to this timestamp
  --scene THRESHOLD     Scene-change sensitivity 0–1 (default 0.1)
  --min-interval SECS   Minimum seconds between frames (default 2)
  --fps FPS             Extract at fixed FPS instead of scene-change
  --scale PX            Frame width in pixels (default 512)
  --batch-size N        Frames per API call (0 = auto, default 1 for all providers)
  --concurrency N       Parallel batch workers (default: 4 for OpenRouter, 1 for local)
  --dedup-threshold N   Perceptual dedup Hamming distance (default 8, 0 = off)
  --no-transcript       Skip audio transcription
  --max-cost USD        Abort if spend exceeds this (default 1.00)
  --timeout SECS        Per-request timeout (default 600)
  -v, --verbose         Debug output
  --stats               Show token + cost usage stats by model, then exit
  --clear-stats         Reset usage log, then exit
```

---

## 🔧 Environment variables

Copy `env.sample` to `.env`:

```bash
# ── Provider ────────────────────────────────────────────────────────────────
# ollama    — local Ollama server (no API key, no cost)
# openai    — any OpenAI-compatible server (LM Studio, oMLX, vLLM, LocalAI, real OpenAI)
# openrouter — cloud inference via OpenRouter
PROVIDER=ollama

# ── Ollama (local, no API key) ──────────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:3b

# ── OpenAI-compatible (LM Studio, oMLX, vLLM, LocalAI, real OpenAI) ─────────
#   LM Studio default port: 1234  |  oMLX default port: 8000  |  vLLM: 8000
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio        # "not-needed" for oMLX/vLLM, real key for OpenAI
OPENAI_MODEL=                   # exact model ID as shown in server (required)

# ── OpenRouter (cloud) ──────────────────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemini-2.5-flash

# ── Frame extraction ────────────────────────────────────────────────────────
SCENE_THRESHOLD=0.1     # lower = more frames
MIN_INTERVAL=2          # seconds between forced frames
MAX_FRAMES=60
FRAME_WIDTH=512
MAX_COST_USD=1.00
BATCH_SIZE=0            # 0 = auto (defaults to 1 per request for all providers)
REQUEST_TIMEOUT=600

# ── Fallback ────────────────────────────────────────────────────────────────
# Same-provider fallback: set FALLBACK_MODEL only.
#   Blank = auto-detect installed models (Ollama) or available models (OpenAI-compat).
# Cross-provider fallback: set FALLBACK_PROVIDER + FALLBACK_MODEL.
#   FALLBACK_BASE_URL and FALLBACK_API_KEY default to the primary provider's values
#   if not set — override only when the fallback uses a different server/key.
FALLBACK_PROVIDER=      # ollama | openai | openrouter  (blank = same as PROVIDER)
FALLBACK_MODEL=         # model ID for fallback  (blank = auto-detect same-provider)
FALLBACK_BASE_URL=      # base URL for fallback openai/ollama server (optional)
FALLBACK_API_KEY=       # API key for fallback provider (optional)
```

---

## 📊 Usage tracking

Every successful run appends a record to `~/.cache/vidlizer/usage.jsonl`. Test runs (`pytest`) are excluded.

```bash
vidlizer --stats          # print per-model breakdown
vidlizer --clear-stats    # reset log
```

Example output:

```
Usage statistics  (/Users/you/.cache/vidlizer/usage.jsonl)

  Total runs:       12
  Total tokens in:  58,240
  Total tokens out: 6,102
  Total cost:       ~$0.0312
  Total steps:      94

  Model                               Provider     Runs   Tokens in  Tokens out  Cost USD
  gemma-4-E2B-it-MLX-4bit             openai          9      45,800       4,900      free
  google/gemini-2.5-flash             openrouter       3      12,440       1,202  ~$0.0312
```

Also available as MCP tool: `get_usage_stats()` → same data as JSON. `clear_usage_stats()` resets the log.

---

## 🔌 MCP Server

Use vidlizer from any MCP-compatible agent — Claude Code, Cursor, Claude Desktop, Gemini CLI.

**Model and provider are set via env vars and cannot be overridden by the AI agent.** This prevents agents from switching to unexpected or expensive models mid-session.

### Install

```bash
pip install -e ".[mcp]"   # adds mcp package + vidlizer-mcp entry point
```

### Configure

Use the **absolute path** to the venv binary — no shell activation needed. Run `which vidlizer-mcp` inside the venv to get the path.

All configs below use JSON format (works in Claude Code, Cursor, Claude Desktop, Gemini CLI). For `claude mcp add` CLI form, replace `"env": { "KEY": "val" }` with `-e KEY=val` flags.

---

#### Single provider — OpenRouter (cloud)

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "OPENROUTER_MODEL": "google/gemini-2.5-flash"
      }
    }
  }
}
```

> Free models (`:free` suffix) automatically fall back to the cheapest paid model if rate-limited — no extra config needed.

---

#### Single provider — Ollama (local, no API key)

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "ollama",
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen2.5vl:7b"
      }
    }
  }
}
```

> `FALLBACK_MODEL` is optional — omitting it auto-detects installed models and tries them in preferred order.

---

#### Single provider — LM Studio (OpenAI-compatible, port 1234)

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "openai",
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
        "OPENAI_API_KEY": "lm-studio",
        "OPENAI_MODEL": "qwen/qwen2.5-vl-7b-instruct"
      }
    }
  }
}
```

---

#### Single provider — oMLX (Apple Silicon, port 8000)

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "openai",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "OPENAI_API_KEY": "not-needed",
        "OPENAI_MODEL": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
      }
    }
  }
}
```

---

#### Cross-provider fallback — oMLX → OpenRouter

Primary runs on local oMLX; if it fails, switches to OpenRouter cloud automatically.

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "openai",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "OPENAI_API_KEY": "not-needed",
        "OPENAI_MODEL": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
        "FALLBACK_PROVIDER": "openrouter",
        "FALLBACK_MODEL": "google/gemini-2.5-flash",
        "FALLBACK_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

---

#### Cross-provider fallback — oMLX → Ollama

Both local; oMLX fails (model missing / server down) → Ollama takes over.

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "openai",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "OPENAI_API_KEY": "not-needed",
        "OPENAI_MODEL": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
        "FALLBACK_PROVIDER": "ollama",
        "FALLBACK_MODEL": "qwen2.5vl:7b"
      }
    }
  }
}
```

> `FALLBACK_BASE_URL` defaults to `OLLAMA_HOST` (or `http://localhost:11434`) — only set it if Ollama is on a non-standard host.

---

#### Cross-provider fallback — Ollama → OpenRouter

Local-first; cloud kicks in if no model is installed or inference fails.

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "ollama",
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen2.5vl:7b",
        "FALLBACK_PROVIDER": "openrouter",
        "FALLBACK_MODEL": "google/gemini-2.5-flash",
        "FALLBACK_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

---

#### Same-provider fallback — pin explicit fallback model

When auto-detection isn't wanted; both models use the same provider.

```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "ollama",
        "OLLAMA_MODEL": "qwen2.5vl:7b",
        "FALLBACK_MODEL": "qwen2.5vl:3b"
      }
    }
  }
}
```

---

**Fallback env var reference:**

| Var | Purpose | Default |
|---|---|---|
| `FALLBACK_PROVIDER` | Provider for fallback (`ollama`/`openai`/`openrouter`) | same as `PROVIDER` |
| `FALLBACK_MODEL` | Model ID for fallback | auto-detect (same provider) |
| `FALLBACK_BASE_URL` | Base URL for fallback `openai`/`ollama` server | `OPENAI_BASE_URL` or `OLLAMA_HOST` |
| `FALLBACK_API_KEY` | API key for fallback provider | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` |

### Logs

All activity (frame extraction, API calls, errors) written to:
```bash
tail -f ~/.cache/vidlizer/mcp.log
```

### Tools

| Tool | Returns | Tokens out |
|---|---|---|
| `analyze_video(path, **opts)` | `analysis_id` + meta | ~100 |
| `list_analyses()` | all stored analyses (meta only) | ~50/entry |
| `get_summary(id, level)` | brief/medium/full text summary | ~200–2K |
| `get_step(id, step)` | single flow step | ~150 |
| `get_steps(id, start, end)` | step range | scaled |
| `get_phase(id, phase)` | all steps in named phase | scaled |
| `search_analysis(id, query)` | steps matching text (`query` or `keyword`) | only hits |
| `get_transcript(id, start_s, end_s)` | transcript slice | scaled |
| `get_full_analysis(id)` | full flow + transcript | full |
| `delete_analysis(id)` | confirmation | ~10 |
| `get_usage_stats()` | per-model token + cost breakdown | ~50/model |
| `clear_usage_stats()` | reset usage log | ~10 |

### Token-efficient workflow

`analyze_video` stores the full result on disk and returns only `analysis_id` + metadata (~100 tokens). The LLM pulls specific parts on demand — a 60-step video costs ~100 tokens to register but only loads what's needed per query.

```
agent: analyze_video("demo.mp4")
→ { "analysis_id": "abc123", "step_count": 42, "phases": ["Intro", "Demo", "Outro"] }

agent: get_summary("abc123", level="brief")
→ "Intro: Title card displayed | Demo: CLI recording | Outro: Results shown"

agent: get_phase("abc123", "Demo")
→ [ { step, timestamp_s, action, scene }, … ]

agent: search_analysis("abc123", "error")
→ [ { step: 17, matched_field: "observations", matched_value: "Stack trace visible" } ]
```

### MCP resources

| URI | Content |
|---|---|
| `vidlizer://analyses` | All analyses (meta JSON) |
| `vidlizer://analyses/{id}` | Full analysis JSON |
| `vidlizer://analyses/{id}/summary` | Medium text summary |

---

## 🧪 Testing

Fully automated test suite — **223 unit + integration tests, 3 e2e tests**.

```bash
make install-dev    # installs pytest, pytest-html, pytest-mock
make test           # runs unit + integration (no network) → reports/test-report.html
make test-e2e       # also runs YouTube download + full pipeline e2e
make smoke          # real-provider smoke test against all detected providers
```

Unit/integration report: `reports/test-report.html`. Tests cover:

- Frame extraction (ffmpeg), perceptual dedup, cache TTL
- Audio detection, transcript merge (no duplicates)
- PDF → frames, image encoding, URL detection
- Output formatter: json/summary/markdown correctness
- Full pipeline with mocked OpenRouter (fake HTTP server)
- HTTP layer: SSE parsing, 429 handling, cost cap, Ollama streaming
- Batch: JSON repair retry, free-model concurrency guard
- Models: pricing lookup, fallback sequences, format helpers
- Usage tracking: record/stats/clear lifecycle
- Real CLI subprocess invocations against real media
- Real YouTube download + full analysis (opt-in `-m e2e`)

### Smoke test

`make smoke` runs the full real-provider pipeline against every detected provider in order (Ollama → LM Studio → oMLX → OpenRouter). Before tests start, it interactively prompts for each local provider:

- Vision model found → `"Test ollama with qwen2.5vl:3b? [Y/n]"`
- No vision model (Ollama) → `"Download qwen2.5vl:3b (~2.3 GB)? [y/N]"`
- No vision model (LM Studio/oMLX) → instructions to load one + retry prompt

Each provider is tested in isolation with its own output directory (prevents cache contamination). Local models are unloaded from VRAM after each provider's tests. Results land in `reports/smoke-TIMESTAMP.html` with per-provider pass/fail scorecards.

```bash
make smoke                                  # auto-detect all providers
make smoke ARGS="--provider ollama"         # force single provider
make smoke ARGS="--provider openrouter"     # OpenRouter only (no local needed)
```

---

## 📝 License

MIT — see [LICENSE](LICENSE).
