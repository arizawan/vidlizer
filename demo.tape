# vidlizer demo — runs real commands against OpenRouter
Output assets/demo.gif

Set Theme "Catppuccin Mocha"
Set FontSize 14
Set Width 1020
Set Height 620
Set Padding 24
Set Framerate 24
Set PlaybackSpeed 1
Set TypingSpeed 40ms

Env PATH "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# ── health check ─────────────────────────────────────────────────────────────
Sleep 500ms
Type "vidlizer doctor"
Enter
Sleep 6s

# ── analyze a video ──────────────────────────────────────────────────────────
Sleep 800ms
Type "vidlizer assets/demo_src.mp4 --max-frames 6 -o /tmp/out.json"
Enter
Sleep 25s

# ── show structured output ───────────────────────────────────────────────────
Sleep 500ms
Type "cat /tmp/out.json | python3 -m json.tool | head -28"
Enter
Sleep 2s

Sleep 3s
