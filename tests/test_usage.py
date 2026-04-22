"""Tests for vidlizer.usage — record_run, get_stats, clear_stats."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import vidlizer.usage as usage_mod


@pytest.fixture
def isolated_usage(tmp_path, monkeypatch):
    """Point usage module at a temp file so tests don't pollute the real log."""
    temp_log = tmp_path / "usage.jsonl"
    monkeypatch.setattr(usage_mod, "_USAGE_PATH", temp_log)
    return temp_log


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# record_run
# ---------------------------------------------------------------------------

def test_record_run_skips_in_pytest_context(isolated_usage):
    """PYTEST_CURRENT_TEST is set by pytest — record_run must not write."""
    assert os.getenv("PYTEST_CURRENT_TEST"), "pytest must set PYTEST_CURRENT_TEST"
    usage_mod.record_run(
        model="test/model", provider="openrouter",
        tokens_in=100, tokens_out=50, cost_usd=0.001,
        source="test.mp4", steps=5,
    )
    assert not isolated_usage.exists()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

def test_get_stats_empty_log(isolated_usage):
    stats = usage_mod.get_stats()
    assert stats["total_runs"] == 0
    assert stats["by_model"] == []
    assert stats["total_cost_usd"] == 0.0


def test_get_stats_aggregates_correctly(isolated_usage):
    _write_records(isolated_usage, [
        {"model": "m1", "provider": "or", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01, "steps": 3},
        {"model": "m1", "provider": "or", "tokens_in": 200, "tokens_out": 100, "cost_usd": 0.02, "steps": 7},
        {"model": "m2", "provider": "ol", "tokens_in": 50, "tokens_out": 25, "cost_usd": 0.0, "steps": 2},
    ])
    stats = usage_mod.get_stats()
    assert stats["total_runs"] == 3
    assert stats["total_tokens_in"] == 350
    assert stats["total_tokens_out"] == 175
    assert stats["total_cost_usd"] == pytest.approx(0.03)
    assert stats["total_steps"] == 12


def test_get_stats_by_model_sorted_by_runs(isolated_usage):
    _write_records(isolated_usage, [
        {"model": "popular", "provider": "or", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
        {"model": "popular", "provider": "or", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
        {"model": "rare", "provider": "or", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
    ])
    stats = usage_mod.get_stats()
    assert stats["by_model"][0]["model"] == "popular"
    assert stats["by_model"][0]["runs"] == 2


def test_get_stats_skips_corrupt_lines(isolated_usage):
    isolated_usage.parent.mkdir(parents=True, exist_ok=True)
    isolated_usage.write_text(
        '{"model":"m1","tokens_in":10,"tokens_out":5,"cost_usd":0.0,"steps":1}\n'
        'CORRUPT LINE\n'
        '{"model":"m1","tokens_in":20,"tokens_out":10,"cost_usd":0.0,"steps":2}\n',
        encoding="utf-8",
    )
    stats = usage_mod.get_stats()
    assert stats["total_runs"] == 2


def test_get_stats_no_log_file(isolated_usage):
    assert not isolated_usage.exists()
    stats = usage_mod.get_stats()
    assert stats["total_runs"] == 0


def test_get_stats_includes_log_path(isolated_usage):
    stats = usage_mod.get_stats()
    assert "log_path" in stats


# ---------------------------------------------------------------------------
# clear_stats
# ---------------------------------------------------------------------------

def test_clear_stats_returns_record_count(isolated_usage):
    _write_records(isolated_usage, [
        {"model": "m", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
        {"model": "m", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
        {"model": "m", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
    ])
    count = usage_mod.clear_stats()
    assert count == 3


def test_clear_stats_deletes_log_file(isolated_usage):
    _write_records(isolated_usage, [
        {"model": "m", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
    ])
    usage_mod.clear_stats()
    assert not isolated_usage.exists()


def test_clear_stats_no_file_returns_zero(isolated_usage):
    assert not isolated_usage.exists()
    assert usage_mod.clear_stats() == 0


def test_clear_stats_then_get_stats_empty(isolated_usage):
    _write_records(isolated_usage, [
        {"model": "m", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "steps": 1},
    ])
    usage_mod.clear_stats()
    stats = usage_mod.get_stats()
    assert stats["total_runs"] == 0
