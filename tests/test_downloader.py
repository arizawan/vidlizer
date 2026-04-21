"""Unit tests for vidlizer.downloader — URL detection and validation."""
from __future__ import annotations

import pytest

from vidlizer.downloader import is_supported, is_url, _is_youtube


@pytest.mark.parametrize("url", [
    "http://example.com/video.mp4",
    "https://example.com/video.mp4",
    "HTTPS://EXAMPLE.COM/VIDEO.MP4",
])
def test_is_url_valid(url):
    assert is_url(url)


@pytest.mark.parametrize("s", [
    "/home/user/video.mp4",
    "video.mp4",
    "ftp://example.com/video.mp4",
    "",
    "not-a-url",
])
def test_is_url_invalid(s):
    assert not is_url(s)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.loom.com/share/abc123",
    "https://vimeo.com/123456",
    "https://twitter.com/user/status/123",
    "https://x.com/user/status/123",
])
def test_is_supported_known_platforms(url):
    assert is_supported(url)


@pytest.mark.parametrize("url", [
    "https://example.com/video.mp4",
    "https://dailymotion.com/video/123",
    "https://tiktok.com/@user/video/123",
])
def test_is_supported_unknown_platform(url):
    assert not is_supported(url)


def test_is_supported_requires_http():
    assert not is_supported("youtube.com/watch?v=abc")


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc",
    "https://youtu.be/abc",
])
def test_is_youtube_true(url):
    assert _is_youtube(url)


@pytest.mark.parametrize("url", [
    "https://www.loom.com/share/abc",
    "https://vimeo.com/123",
    "https://twitter.com/user/status/123",
])
def test_is_youtube_false_for_other_platforms(url):
    assert not _is_youtube(url)
