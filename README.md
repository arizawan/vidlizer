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

vidlizer extracts frames with ffmpeg, sends them to a vision model via [OpenRouter](https://openrouter.ai), and writes a `flow` array describing every scene, action, and visible text. For videos with audio it automatically transcribes speech with Apple MLX Whisper and merges it into each step.

```bash
vidlizer demo.mp4
vidlizer "https://youtube.com/watch?v=..."
vidlizer screenshot.png
vidlizer document.pdf
```

---

## ✨ Features

- **Any input** — local video, image (jpg/png/webp/…), PDF, or URL (YouTube, Loom, Vimeo, Twitter)
- **3 output formats** — `--format json` (default), `summary` (plain text by phase), `markdown` (step-per-section doc)
- **Auto transcript** — detects audio, transcribes with Apple MLX Whisper (Neural Engine), merges speech into each flow step
- **Perceptual dedup** — removes near-duplicate frames before sending (saves tokens)
- **analyze_moment** — `--start`/`--end` flags to focus on a time range
- **In-memory cache** — repeat runs on the same file skip the API call
- **Multi-model** — 7 curated models with live pricing; free models auto-fallback to cheapest paid
- **Cost guard** — aborts if spend exceeds `MAX_COST_USD` (default $1.00)
- **Live progress** — Rich streaming indicator shows elapsed time and token count per batch
- **Auto-install** — missing `ffmpeg` is brew-installed; `mlx-whisper` is pip-installed on first audio video
- **Mac-native** — file picker dialog, Apple MLX transcription, osascript integration

---

## 📦 Requirements

- macOS (Apple Silicon recommended for transcription speed)
- Python 3.10+
- An [OpenRouter API key](https://openrouter.ai/keys)

`ffmpeg` is installed automatically via Homebrew on first run if missing.

---

## 🚀 Install

```bash
git clone https://github.com/arizawan/vidlizer.git
cd vidlizer
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp env.sample .env        # paste your OpenRouter key
```

---

## ⚡ Quick start

```bash
# Analyze a local video
vidlizer demo.mp4

# Analyze a YouTube video
vidlizer "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Analyze a single image
vidlizer screenshot.png

# Analyze a PDF
vidlizer report.pdf

# Focus on a time range (analyze_moment)
vidlizer demo.mp4 --start 30 --end 90

# Pick model + output path explicitly
vidlizer demo.mp4 --model google/gemini-2.5-flash -o result.json

# Output as Markdown or plain-text summary
vidlizer demo.mp4 --format markdown -o result.md
vidlizer demo.mp4 --format summary -o result.txt
```

Run with no arguments to get an interactive file picker and model selector.

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

Models are fetched live from OpenRouter with current pricing. Seven curated defaults:

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

Override via env var: `OPENROUTER_MODEL=google/gemini-2.5-flash`

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
  --model MODEL         OpenRouter model slug
  --max-frames N        Max frames to send (default 60, hard cap 200)
  --start SECONDS       Analyze from this timestamp
  --end SECONDS         Analyze up to this timestamp
  --scene THRESHOLD     Scene-change sensitivity 0–1 (default 0.1)
  --min-interval SECS   Minimum seconds between frames (default 2)
  --fps FPS             Extract at fixed FPS instead of scene-change
  --scale PX            Frame width in pixels (default 512)
  --batch-size N        Frames per API call (0 = auto)
  --dedup-threshold N   Perceptual dedup Hamming distance (default 8, 0 = off)
  --no-transcript       Skip audio transcription
  --max-cost USD        Abort if spend exceeds this (default 1.00)
  --timeout SECS        Per-request timeout (default 600)
  -v, --verbose         Debug output
```

---

## 🔧 Environment variables

Copy `env.sample` to `.env` and fill in your key. All CLI flags can also be set here:

```bash
OPENROUTER_API_KEY=sk-or-v1-...   # required
OPENROUTER_MODEL=google/gemini-2.5-flash

SCENE_THRESHOLD=0.1     # lower = more frames
MIN_INTERVAL=2          # seconds between forced frames
MAX_FRAMES=60
FRAME_WIDTH=512
MAX_COST_USD=1.00
BATCH_SIZE=0
REQUEST_TIMEOUT=600
```

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
