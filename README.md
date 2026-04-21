<div align="center">

# vidlizer

**Feed it any video, image, or PDF — get back a structured JSON timeline of everything that happened.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](#requirements)
[![Tests](https://img.shields.io/badge/tests-103%20passing-brightgreen.svg)](#testing)

![demo](assets/demo.gif)

</div>

---

vidlizer extracts frames with ffmpeg, sends them to a vision model, and writes a `flow` array describing every scene, action, and visible text. For videos with audio it automatically transcribes speech with Apple MLX Whisper and merges it into each step.

Two provider modes: **local** via [Ollama](https://ollama.com) (no API key, no cost) or **cloud** via [OpenRouter](https://openrouter.ai).

```bash
vidlizer demo.mp4
vidlizer "https://youtube.com/watch?v=..."
vidlizer screenshot.png
vidlizer document.pdf
```

---

## ✨ Features

- **Any input** — local video, image (jpg/png/webp/…), PDF, or URL (YouTube, Loom, Vimeo, Twitter)
- **Local inference** — run fully offline via Ollama (`--provider ollama`), no API key needed
- **Cloud inference** — OpenRouter with 7 curated models, live pricing, free model auto-fallback
- **3 output formats** — `--format json` (default), `summary` (plain text by phase), `markdown` (step-per-section doc)
- **Auto transcript** — detects audio, transcribes with Apple MLX Whisper (Neural Engine), merges speech into each flow step
- **Perceptual dedup** — removes near-duplicate frames before sending (saves tokens)
- **analyze_moment** — `--start`/`--end` flags to focus on a time range
- **In-memory cache** — repeat runs on the same file skip the API call
- **Cost guard** — aborts if spend exceeds `MAX_COST_USD` (default $1.00)
- **Live progress** — Rich streaming indicator shows elapsed time and token count per batch
- **Auto-install** — missing `ffmpeg` is brew-installed; `mlx-whisper` is pip-installed on first audio video
- **Mac-native** — file picker dialog, Apple MLX transcription, osascript integration

---

## 📦 Requirements

- macOS (Apple Silicon recommended for transcription speed)
- Python 3.10+
- **Local mode**: [Ollama](https://ollama.com) installed + a vision model pulled (8 GB+ RAM)
- **Cloud mode**: An [OpenRouter API key](https://openrouter.ai/keys)

`ffmpeg` is installed automatically via Homebrew on first run if missing.

---

## 🚀 Install

```bash
git clone https://github.com/arizawan/vidlizer.git
cd vidlizer
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp env.sample .env        # configure provider + keys
```

**Local mode** (no API key needed):

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5vl:3b   # ~3.2 GB, requires 8 GB+ RAM
```

**Cloud mode**:

```bash
# Paste your OpenRouter key in .env:
# OPENROUTER_API_KEY=sk-or-v1-...
```

---

## ⚡ Quick start

```bash
# Local inference (Ollama, free, no API key)
vidlizer demo.mp4 --provider ollama --model qwen2.5vl:3b

# Cloud inference (OpenRouter)
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
  ]
}
```

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

### Local (Ollama) — no API key, no cost

Requires 8 GB+ RAM. Install a model with `ollama pull <name>`:

| Model | Disk | RAM | Notes |
|---|---|---|---|
| `qwen2.5vl:3b` | 3.2 GB | ~5 GB | **Recommended** — 125K ctx, strong JSON, multi-image |
| `qwen2.5vl:7b` | 6.0 GB | ~9 GB | Best local quality, needs 10+ GB RAM |
| `minicpm-v:8b` | 5.5 GB | ~8 GB | Strong OCR + visual reasoning, 32K ctx |

Uses Ollama's native `/api/chat` endpoint with `format: json` for reliable structured output. One frame per request (per-frame batching is automatic).

### Cloud (OpenRouter)

Models fetched live with current pricing:

| Model | Pricing | Notes |
|---|---|---|
| `google/gemini-2.5-flash` | ~$0.001/run | **Recommended** — fast, accurate |
| `google/gemini-2.5-flash-lite` | cheaper | Slightly less accurate |
| `google/gemini-2.5-pro` | expensive | Best quality |
| `openai/gpt-4o-mini` | low | OpenAI budget option |
| `openai/gpt-4o` | expensive | OpenAI flagship |
| `nvidia/nemotron-nano-12b-v2-vl:free` | free ⚡ | Rate-limited, auto-batched |
| `google/gemma-4-31b-it:free` | free ⚡ | Rate-limited, may be slow |

Free models auto-fallback to the cheapest paid model on failure.

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
  --provider PROVIDER   ollama (local, default) or openrouter (cloud)
  --model MODEL         Model slug — Ollama name or OpenRouter slug
  --max-frames N        Max frames to send (default 60, hard cap 200)
  --start SECONDS       Analyze from this timestamp
  --end SECONDS         Analyze up to this timestamp
  --scene THRESHOLD     Scene-change sensitivity 0–1 (default 0.1)
  --min-interval SECS   Minimum seconds between frames (default 2)
  --fps FPS             Extract at fixed FPS instead of scene-change
  --scale PX            Frame width in pixels (default 512)
  --batch-size N        Frames per API call (0 = auto; Ollama forces 1)
  --dedup-threshold N   Perceptual dedup Hamming distance (default 8, 0 = off)
  --no-transcript       Skip audio transcription
  --max-cost USD        Abort if spend exceeds this (default 1.00)
  --timeout SECS        Per-request timeout (default 600)
  -v, --verbose         Debug output
```

---

## 🔧 Environment variables

Copy `env.sample` to `.env`:

```bash
# Provider: ollama (default, local) or openrouter (cloud)
PROVIDER=ollama

# Ollama (local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:3b

# OpenRouter (cloud)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemini-2.5-flash

# Frame extraction
SCENE_THRESHOLD=0.1     # lower = more frames
MIN_INTERVAL=2          # seconds between forced frames
MAX_FRAMES=60
FRAME_WIDTH=512
MAX_COST_USD=1.00
BATCH_SIZE=0            # Ollama always uses 1 (one frame per request)
REQUEST_TIMEOUT=600
```

---

## 🔌 MCP Server

Use vidlizer from any MCP-compatible agent — Claude Code, Cursor, Gemini CLI, PI Code, Claude Desktop.

### Install

```bash
pip install -e ".[mcp]"   # adds mcp package + vidlizer-mcp entry point
```

### Configure

Use the **absolute path** to the venv binary — no shell activation needed.

**Claude Code** (adds to `~/.claude.json`):
```bash
claude mcp add vidlizer /path/to/.venv/bin/vidlizer-mcp \
  -e PROVIDER=openrouter \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -e OPENROUTER_MODEL=google/gemini-2.5-flash
```

**Other clients** (Cursor `.cursor/mcp.json`, Claude Desktop, Bolt, etc.):
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

Local (Ollama, no API key):
```json
{
  "mcpServers": {
    "vidlizer": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/vidlizer-mcp",
      "env": {
        "PROVIDER": "ollama",
        "OLLAMA_MODEL": "qwen2.5vl:3b"
      }
    }
  }
}
```

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
| `search_analysis(id, query)` | steps matching text | only hits |
| `get_transcript(id, start_s, end_s)` | transcript slice | scaled |
| `get_full_analysis(id)` | full flow + transcript | full |
| `delete_analysis(id)` | confirmation | ~10 |

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

Fully automated test suite — **103 unit + integration tests, 3 e2e tests**.

```bash
make install-dev    # installs pytest, pytest-html, pytest-mock
make test           # runs unit + integration (no network) → HTML report
make test-e2e       # also runs YouTube download + full pipeline e2e
```

Reports land in `reports/test-report.html`. Tests cover:

- Frame extraction (ffmpeg), perceptual dedup, cache TTL
- Audio detection, transcript merge (no duplicates)
- PDF → frames, image encoding, URL detection
- Output formatter: json/summary/markdown correctness
- Full pipeline with mocked OpenRouter (fake HTTP server)
- Real CLI subprocess invocations against real media
- Real YouTube download + full analysis (opt-in `-m e2e`)

---

## 📝 License

MIT — see [LICENSE](LICENSE).
