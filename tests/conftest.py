"""Shared fixtures and mock helpers for the vidlizer test suite."""
from __future__ import annotations

import http.server
import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# HTML report hooks (pytest-html)
# ---------------------------------------------------------------------------

def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring real network / API key")
    if hasattr(config, "_metadata"):
        config._metadata["Project"] = "vidlizer"
        config._metadata["Version"] = "0.2.2"
        config._metadata["Docs"] = "https://github.com/arizawan/vidlizer"


def pytest_html_report_title(report) -> None:
    report.title = "vidlizer — Test Suite Report"


# ---------------------------------------------------------------------------
# Canonical mock flow returned by the fake OpenRouter server
# ---------------------------------------------------------------------------

MOCK_FLOW = {
    "flow": [
        {
            "step": 1,
            "timestamp_s": 0.0,
            "phase": "Test",
            "scene": "Test scene with blue background.",
            "subjects": ["Background"],
            "action": "Static frame displayed.",
            "text_visible": "",
            "context": "Automated test fixture.",
            "observations": "Solid color — no anomalies.",
            "next_scene": None,
        }
    ]
}


def make_mock_post(flow_data: dict = MOCK_FLOW):
    """Return a callable that mimics requests.post with SSE streaming."""
    flow_json = json.dumps(flow_data)
    content_escaped = json.dumps(flow_json)  # embeds flow JSON as a JSON string value

    lines = [
        f'data: {{"choices":[{{"delta":{{"content":{content_escaped}}}}}],"usage":null}}'.encode(),
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
        b'data: [DONE]',
    ]

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)

    return MagicMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# Mock HTTP server for subprocess / CLI tests
# ---------------------------------------------------------------------------

class _MockOpenRouterHandler(http.server.BaseHTTPRequestHandler):
    """Returns a minimal SSE response for any POST."""

    def do_POST(self):
        flow_json = json.dumps(MOCK_FLOW)
        content_escaped = json.dumps(flow_json)

        body = (
            f'data: {{"choices":[{{"delta":{{"content":{content_escaped}}}}}],"usage":null}}\n'
            '\n'
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n'
            '\n'
            'data: [DONE]\n'
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress access logs


@pytest.fixture(scope="session")
def mock_openrouter_server():
    """Start a local mock OpenRouter HTTP server; yield its base URL."""
    server = http.server.HTTPServer(("localhost", 0), _MockOpenRouterHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}/v1/chat/completions"
    server.shutdown()


# ---------------------------------------------------------------------------
# Media fixtures generated on-the-fly via ffmpeg / pymupdf
# ---------------------------------------------------------------------------

def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


@pytest.fixture(scope="session")
def asset_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("assets")


@pytest.fixture(scope="session")
def test_video(asset_dir) -> Path:
    """5-second 320×240 MP4 with audio (colour bars + 440 Hz tone)."""
    out = asset_dir / "test.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "5",
        "-c:v", "libx264", "-c:a", "aac",
        str(out),
    )
    return out


@pytest.fixture(scope="session")
def test_video_silent(asset_dir) -> Path:
    """5-second 320×240 MP4 without audio."""
    out = asset_dir / "test_silent.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=10",
        "-t", "5", "-an",
        str(out),
    )
    return out


@pytest.fixture(scope="session")
def test_image_png(asset_dir) -> Path:
    """Single 320×240 PNG (green frame from ffmpeg)."""
    out = asset_dir / "test.png"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=green:size=320x240",
        "-frames:v", "1",
        str(out),
    )
    return out


@pytest.fixture(scope="session")
def test_pdf(asset_dir) -> Path:
    """2-page PDF created with pymupdf."""
    import fitz

    out = asset_dir / "test.pdf"
    doc = fitz.open()
    for i, color in enumerate([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(page.rect, color=color, fill=color)
        page.insert_text((100, 420), f"Page {i + 1}", fontsize=24, color=(1, 1, 1))
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture(scope="session")
def different_images(asset_dir) -> list[Path]:
    """5 visually distinct 64×64 JPEGs: vertical red bar at different x positions.

    Each image has a red stripe on a white background; the stripe shifts 12px
    each time, creating distinct horizontal gradient patterns for dHash.
    """
    import fitz

    paths = []
    for i in range(5):
        p = asset_dir / f"diff_{i}.jpg"
        doc = fitz.open()
        page = doc.new_page(width=64, height=64)
        page.draw_rect(fitz.Rect(0, 0, 64, 64), color=(1, 1, 1), fill=(1, 1, 1))
        x = i * 12  # 0, 12, 24, 36, 48
        page.draw_rect(fitz.Rect(x, 0, x + 12, 64), color=(1, 0, 0), fill=(1, 0, 0))
        pix = page.get_pixmap(colorspace=fitz.csGRAY)
        pix.save(str(p))
        doc.close()
        paths.append(p)
    return paths
