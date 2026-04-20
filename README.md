# vidlizer

Frame-by-frame video analyzer → JSON user-journey map via OpenRouter vision models.

Feed it a screen recording. It samples scene-change frames with ffmpeg, sends them to
a vision model, and writes a structured `flow` array describing every step the user
took, the UI feedback they saw, and any logic bugs or media-load failures.

## Install

```bash
git clone git@github.com:arizawan/vidlizer.git
cd vidlizer
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp env.sample .env         # then paste your OpenRouter key into .env
```

Requires `ffmpeg` on `PATH`.

## Use

```bash
vidlizer path/to/recording.mp4
```

Any omitted option is asked interactively. CLI flags override `.env`:

```bash
vidlizer demo.mp4 --model google/gemini-2.5-flash --max-cost 0.25
```

Output goes to `<video>.analysis.json` by default, or pass `-o path.json`.

## Models

| Model                                     | Pricing       | Notes                           |
|-------------------------------------------|---------------|---------------------------------|
| `google/gemini-2.5-flash`                 | ~$0.001/run   | Recommended default             |
| `google/gemini-2.5-flash-lite`            | cheaper       | Slightly less accurate          |
| `nvidia/nemotron-nano-12b-v2-vl:free`     | free          | Slow, 10-image cap (auto-batched) |
| `google/gemma-4-31b-it:free`              | free          | May rate-limit                  |
| `google/gemini-2.5-pro`, `openai/gpt-4o`  | expensive     | Warns before running            |

## Cost safeguards

- **`MAX_COST_USD`** (default `1.00`): aborts mid-run if spend exceeds this.
- **`MAX_FRAMES`** (default `60`, hard-capped at `200`): caps frames sent.
- Batching recursion is depth-limited — can't infinite-loop on image-limit errors.
- Expensive models show a pre-run warning with their per-token rates.

## Output schema

See `sample-output.json`. Each `flow[]` step has `step`, `phase`, `page`,
`text_context`, `input`, `screen_data` (`timer_state`, `score_state`,
`media_status`, `ui_feedback`), and `next_screen`.

## License

Private. Unauthorized redistribution not permitted.
