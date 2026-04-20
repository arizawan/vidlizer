# vidlizer

Frame-by-frame video analyzer → structured JSON event map via OpenRouter vision models.

Feed it any video. It samples scene-change frames with ffmpeg, sends them to a vision
model, and writes a structured `flow` array describing every event, action, subject,
and piece of visible text — works for screen recordings, tutorials, interviews,
product demos, marketing content, or any other video type.

## Install

```bash
git clone git@github.com:arizawan/vidlizer.git
cd vidlizer
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp env.sample .env         # then paste your OpenRouter key into .env
```

Requires `ffmpeg` on `PATH`. For PDF support add the optional extra (57 MB):

```bash
pip install -e ".[pdf]"
```

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

Each `flow[]` step has:

| Field | Description |
|---|---|
| `step` | Sequential integer |
| `phase` | Logical section (Introduction, Action, Conclusion, …) |
| `scene` | What is currently visible |
| `subjects` | Key people, objects, or UI elements present |
| `action` | What is happening (interaction, movement, narration, …) |
| `text_visible` | All readable text on screen |
| `context` | Persistent state (timer, score, topic, brand, …) |
| `observations` | Errors, anomalies, emotions, key facts |
| `next_scene` | Brief description of what follows |

## License

Private. Unauthorized redistribution not permitted.
