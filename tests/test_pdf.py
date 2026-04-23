"""Integration tests for PDF → frame conversion."""
from __future__ import annotations


from vidlizer.core import pdf_to_frames


def test_pdf_to_frames_count(test_pdf, tmp_path):
    frames = pdf_to_frames(test_pdf, tmp_path, scale=512, max_frames=10)
    assert len(frames) == 2  # fixture has 2 pages


def test_pdf_max_frames_cap(test_pdf, tmp_path):
    frames = pdf_to_frames(test_pdf, tmp_path, scale=512, max_frames=1)
    assert len(frames) == 1


def test_pdf_outputs_jpgs(test_pdf, tmp_path):
    frames = pdf_to_frames(test_pdf, tmp_path, scale=512, max_frames=10)
    assert all(f.suffix == ".jpg" for f in frames)
    assert all(f.exists() for f in frames)


def test_pdf_frames_named_sequentially(test_pdf, tmp_path):
    frames = pdf_to_frames(test_pdf, tmp_path, scale=512, max_frames=10)
    names = [f.stem for f in frames]
    assert names == ["p_00001", "p_00002"]
