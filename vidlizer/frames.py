"""Frame extraction, PDF rendering, and image encoding."""
from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path

from rich.console import Console

_console = Console(stderr=True, highlight=False)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_PDF_EXTS = {".pdf"}

_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}


def _info(msg: str) -> None:
    _console.print(f"[cyan]→[/cyan] {msg}")


def _dbg(msg: str) -> None:
    _console.print(f"[dim]{msg}[/dim]", markup=False)


def extract_frames(
    video: Path,
    out_dir: Path,
    scale: int,
    max_frames: int,
    scene_threshold: float,
    fps: float | None,
    min_interval: float,
    verbose: bool,
    start: float | None = None,
    end: float | None = None,
) -> list[Path]:
    """Hybrid extraction: scene-change OR time-based minimum interval."""
    if fps is not None:
        vf = f"fps={fps},scale={scale}:-2:flags=lanczos,format=yuvj420p"
        mode_desc = f"fixed fps={fps}"
    else:
        vf = (
            f"select='gt(scene\\,{scene_threshold})+isnan(prev_selected_t)"
            f"+gte(t-prev_selected_t\\,{min_interval})',"
            f"scale={scale}:-2:flags=lanczos,format=yuvj420p"
        )
        mode_desc = f"scene>{scene_threshold} or every {min_interval}s"

    ts_file = out_dir / ".timestamps.txt"
    vf_with_ts = vf + f",metadata=print:file={ts_file}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning" if verbose else "error",
        "-y", "-i", str(video),
    ]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += [
        "-vf", vf_with_ts,
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        str(out_dir / "f_%05d.jpg"),
    ]

    _info(f"ffmpeg mode: [bold]{mode_desc}[/bold]")
    if verbose:
        _dbg(f"[ffmpeg] cmd: {' '.join(cmd)}")

    with _console.status("[dim]extracting frames…[/dim]", spinner="dots2"):
        subprocess.run(cmd, check=True)

    frames = sorted(out_dir.glob("f_*.jpg"))

    if ts_file.exists():
        raw_times = re.findall(r'pts_time:([\d.]+)', ts_file.read_text())
        ts_map = {frames[i].name: float(raw_times[i]) for i in range(min(len(frames), len(raw_times)))}
        (out_dir / ".timestamps.json").write_text(json.dumps(ts_map))

    if verbose:
        total_kb = sum(f.stat().st_size for f in frames) / 1024
        _dbg(f"[frames] extracted {len(frames)} frames, total {total_kb:.1f} KB")
        for i, f in enumerate(frames):
            _dbg(f"  [{i+1:03d}] {f.name}  {f.stat().st_size/1024:.1f} KB")
    return frames


def pdf_to_frames(pdf: Path, out_dir: Path, scale: int, max_frames: int) -> list[Path]:
    """Render PDF pages to JPEG images using pymupdf."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise RuntimeError("pymupdf not installed — run: pip install pymupdf")

    doc = fitz.open(str(pdf))
    total = len(doc)
    n = min(total, max_frames)
    _info(f"PDF: [bold]{total} pages[/bold]  (rendering {n})")

    zoom = scale / 595.0
    mat = fitz.Matrix(zoom, zoom)

    paths: list[Path] = []
    for i in range(n):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        out = out_dir / f"p_{i+1:05d}.jpg"
        pix.save(str(out))
        paths.append(out)

    doc.close()
    return paths


def encode_frame(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = _MIME.get(path.suffix.lower(), "image/jpeg")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
