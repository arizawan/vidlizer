# Contributing to vidlizer

## Setup

```bash
git clone https://github.com/arizawan/vidlizer.git
cd vidlizer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp env.sample .env   # fill in at least one provider
```

## Running tests

```bash
make test          # unit tests only (no network, fast)
make test-e2e      # includes real API calls
make smoke         # interactive smoke test against live providers
```

## Code style

- Python 3.10+, ruff for linting: `ruff check vidlizer/`
- Functions ≤ 50 lines, files ≤ 300 lines
- No comments unless the **why** is non-obvious

## Project layout

| Path | Purpose |
|---|---|
| `vidlizer/cli.py` | CLI entry point, interactive prompts |
| `vidlizer/core.py` | Main `run()` orchestration |
| `vidlizer/batch.py` | Frame batching, JSON parsing, parallel dispatch |
| `vidlizer/frames.py` | ffmpeg frame extraction |
| `vidlizer/http.py` | API calls, cost tracking |
| `vidlizer/detect.py` | Provider detection (read-only) |
| `vidlizer/bootstrap.py` | Auto-install ffmpeg / mlx-whisper |
| `scripts/smoke.py` | Interactive smoke test |

## Submitting a PR

1. Branch from `main`
2. Make your change + add tests
3. `make test` passes, `ruff check vidlizer/` clean
4. Open a PR — the template will guide you

## Reporting bugs

Use the [GitHub issue tracker](https://github.com/arizawan/vidlizer/issues).
Include your provider, model, macOS version, and the full error output.
