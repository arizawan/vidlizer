#!/usr/bin/env python3
"""Perceptual frame deduplication via dHash (uses pymupdf — already a dep)."""
from __future__ import annotations

from pathlib import Path

DEFAULT_THRESHOLD = 8  # Hamming distance; 0 = disabled


def _dhash(path: Path, size: int = 8) -> int:
    """Compute dHash for a JPEG/image via pymupdf. Returns 64-bit int."""
    import fitz

    doc = fitz.open(str(path))
    page = doc[0]
    w = max(page.rect.width, 1)
    h = max(page.rect.height, 1)
    mat = fitz.Matrix((size + 1) / w, size / h)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    doc.close()

    s = pix.samples  # bytes, one per pixel, (size+1)*size values
    bits = [
        s[row * (size + 1) + col] > s[row * (size + 1) + col + 1]
        for row in range(size)
        for col in range(size)
    ]
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_frames(frames: list[Path], threshold: int = DEFAULT_THRESHOLD) -> list[Path]:
    """Return frames with near-duplicates removed (keeps first of each group)."""
    if threshold <= 0 or len(frames) <= 1:
        return frames
    kept: list[Path] = []
    hashes: list[int] = []
    for f in frames:
        try:
            h = _dhash(f)
        except Exception:
            kept.append(f)
            continue
        if not hashes or all(_hamming(h, prev) >= threshold for prev in hashes):
            kept.append(f)
            hashes.append(h)
    return kept
