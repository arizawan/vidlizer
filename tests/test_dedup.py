"""Unit tests for vidlizer.dedup."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vidlizer.dedup import DEFAULT_THRESHOLD, _hamming, dedup_frames


@pytest.fixture
def identical_images(asset_dir, different_images) -> list[Path]:
    """5 byte-for-byte identical copies of the first test image."""
    src = different_images[0]
    copies = []
    for i in range(5):
        dst = asset_dir / f"dup_{i}.jpg"
        shutil.copy(src, dst)
        copies.append(dst)
    return copies


def test_identical_frames_deduplicated(identical_images):
    kept = dedup_frames(identical_images, threshold=DEFAULT_THRESHOLD)
    assert len(kept) == 1
    assert kept[0] == identical_images[0]


def test_all_different_frames_kept(different_images):
    kept = dedup_frames(different_images, threshold=DEFAULT_THRESHOLD)
    assert len(kept) == len(different_images)


def test_threshold_zero_disables_dedup(identical_images):
    kept = dedup_frames(identical_images, threshold=0)
    assert kept == identical_images


def test_single_frame_unchanged(different_images):
    single = [different_images[0]]
    assert dedup_frames(single) == single


def test_empty_list():
    assert dedup_frames([]) == []


def test_corrupted_file_kept(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    kept = dedup_frames([bad], threshold=DEFAULT_THRESHOLD)
    assert kept == [bad]


def test_hamming_identical():
    assert _hamming(0, 0) == 0
    assert _hamming(0xFF, 0xFF) == 0


def test_hamming_all_bits_differ():
    assert _hamming(0xFFFFFFFFFFFFFFFF, 0) == 64


def test_hamming_one_bit():
    assert _hamming(0b01, 0b11) == 1
