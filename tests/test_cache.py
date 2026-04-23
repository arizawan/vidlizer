"""Unit tests for vidlizer.cache."""
from __future__ import annotations

import time

import pytest

from vidlizer import cache as c


@pytest.fixture(autouse=True)
def _clear():
    c.clear()
    yield
    c.clear()


def test_put_and_get(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"data")
    result = {"flow": [{"step": 1}]}
    c.put(f, {"model": "test"}, result)
    assert c.get(f, {"model": "test"}) == result


def test_miss_on_empty(tmp_path):
    f = tmp_path / "v.mp4"
    assert c.get(f, {"model": "test"}) is None


def test_ttl_expiry(tmp_path, monkeypatch):
    import vidlizer.cache as cache_mod

    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    c.put(f, {"model": "test"}, {"flow": []})

    monkeypatch.setattr(cache_mod, "_TTL", 0)
    assert c.get(f, {"model": "test"}) is None


def test_file_change_invalidates(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"original")
    c.put(f, {"model": "test"}, {"flow": [{"step": 1}]})

    time.sleep(0.02)
    f.write_bytes(b"modified_content_different_size")
    # Different size → different key → miss
    assert c.get(f, {"model": "test"}) is None


def test_different_params_different_keys(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    c.put(f, {"model": "a"}, {"flow": [{"step": 1}]})
    c.put(f, {"model": "b"}, {"flow": [{"step": 2}]})
    assert c.get(f, {"model": "a"}) == {"flow": [{"step": 1}]}
    assert c.get(f, {"model": "b"}) == {"flow": [{"step": 2}]}


def test_none_path():
    result = {"flow": []}
    c.put(None, {"tag": "urlbased"}, result)
    assert c.get(None, {"tag": "urlbased"}) == result


def test_clear():
    c.put(None, {"k": "1"}, {"flow": []})
    c.clear()
    assert c.get(None, {"k": "1"}) is None
