"""CLI tests — both unit-level and subprocess invocations."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Unit: filename normalization (no subprocess needed)
# ---------------------------------------------------------------------------

def _normalize(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "output"


@pytest.mark.parametrize("stem,expected", [
    ("My Video 2024", "my-video-2024"),
    ("demo.mp4", "demo-mp4"),
    ("hello_world", "hello-world"),
    ("---bad---name---", "bad-name"),
    ("UPPER CASE FILE", "upper-case-file"),
    ("file!@#$%name", "file-name"),
    ("", "output"),
    ("123", "123"),
])
def test_filename_normalization(stem, expected):
    assert _normalize(stem) == expected


# ---------------------------------------------------------------------------
# Unit: output path uses CWD for temp-dir inputs
# ---------------------------------------------------------------------------

def test_output_path_cwd_for_url_input(tmp_path):
    import tempfile
    fake_tmp = Path(tempfile.gettempdir()) / "vidlizer_dl_xyz" / "video.mp4"
    import re as _re
    import tempfile as _tf

    safe_stem = _re.sub(r"[^a-z0-9]+", "-", fake_tmp.stem.lower()).strip("-") or "output"
    out_dir = Path.cwd() if str(fake_tmp).startswith(_tf.gettempdir()) else fake_tmp.parent
    result = out_dir / f"{safe_stem}.analysis.json"
    assert result.parent == Path.cwd()


def test_output_path_same_dir_for_local_file(tmp_path):
    local = tmp_path / "my video.mp4"
    import re as _re, tempfile as _tf
    safe_stem = _re.sub(r"[^a-z0-9]+", "-", local.stem.lower()).strip("-") or "output"
    out_dir = Path.cwd() if str(local).startswith(_tf.gettempdir()) else local.parent
    result = out_dir / f"{safe_stem}.analysis.json"
    assert result.parent == tmp_path


# ---------------------------------------------------------------------------
# Subprocess: --help
# ---------------------------------------------------------------------------

def test_help_exits_zero():
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert r.returncode == 0
    assert "vidlizer" in r.stdout.lower() or "vidlizer" in r.stderr.lower()


# ---------------------------------------------------------------------------
# Subprocess: missing API key → non-zero exit
# ---------------------------------------------------------------------------

def test_missing_api_key_fails(test_video, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env.pop("OPENROUTER_MODEL", None)
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli", str(test_video),
         "-o", str(tmp_path / "out.json"), "--model", "google/gemini-2.5-flash"],
        capture_output=True, text=True, timeout=30,
        env={**env, "OPENROUTER_API_KEY": ""},
    )
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# Subprocess: file not found → non-zero exit
# ---------------------------------------------------------------------------

def test_missing_file_fails(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli",
         str(tmp_path / "nonexistent.mp4"),
         "--model", "google/gemini-2.5-flash"],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "OPENROUTER_API_KEY": "sk-fake"},
    )
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# Subprocess: full analysis run against mock OpenRouter server
# ---------------------------------------------------------------------------

def test_full_run_via_cli(test_video, tmp_path, mock_openrouter_server):
    out = tmp_path / "cli-result.json"
    env = {
        **os.environ,
        "PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-test-key",
        "OPENROUTER_BASE_URL": mock_openrouter_server,
        "OPENROUTER_MODEL": "google/gemini-2.5-flash",
    }
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli",
         str(test_video), "-o", str(out), "--no-transcript"],
        capture_output=True, text=True, timeout=60,
        env=env,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
    assert out.exists(), "output file not written"
    data = json.loads(out.read_text())
    assert "flow" in data
    assert len(data["flow"]) >= 1


def test_cli_image_input(test_image_png, tmp_path, mock_openrouter_server):
    out = tmp_path / "image-result.json"
    env = {
        **os.environ,
        "PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-test-key",
        "OPENROUTER_BASE_URL": mock_openrouter_server,
        "OPENROUTER_MODEL": "google/gemini-2.5-flash",
    }
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli",
         str(test_image_png), "-o", str(out)],
        capture_output=True, text=True, timeout=30,
        env=env,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
    data = json.loads(out.read_text())
    assert "flow" in data


def test_cli_pdf_input(test_pdf, tmp_path, mock_openrouter_server):
    out = tmp_path / "pdf-result.json"
    env = {
        **os.environ,
        "PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-test-key",
        "OPENROUTER_BASE_URL": mock_openrouter_server,
        "OPENROUTER_MODEL": "google/gemini-2.5-flash",
    }
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli",
         str(test_pdf), "-o", str(out)],
        capture_output=True, text=True, timeout=30,
        env=env,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
    data = json.loads(out.read_text())
    assert "flow" in data


def test_cli_start_end_flags(test_video, tmp_path, mock_openrouter_server):
    out = tmp_path / "range-result.json"
    env = {
        **os.environ,
        "PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-test-key",
        "OPENROUTER_BASE_URL": mock_openrouter_server,
        "OPENROUTER_MODEL": "google/gemini-2.5-flash",
    }
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli",
         str(test_video), "-o", str(out),
         "--start", "1", "--end", "3", "--no-transcript"],
        capture_output=True, text=True, timeout=60,
        env=env,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr[-2000:]}"
    assert out.exists()


# ---------------------------------------------------------------------------
# Subprocess: vidlizer doctor
# ---------------------------------------------------------------------------

def test_doctor_exits_nonzero_when_env_missing(tmp_path):
    """doctor returns exit code 1 when no .env exists."""
    project_root = str(Path(__file__).parent.parent)
    env = {k: v for k, v in os.environ.items()
           if k not in {"PROVIDER", "OPENROUTER_API_KEY", "OLLAMA_MODEL", "DOTENV_PATH"}}
    env["PYTHONPATH"] = project_root
    env["VIDLIZER_CONFIG_DIR"] = str(tmp_path)  # empty dir, no .env
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli", "doctor"],
        capture_output=True, text=True, timeout=10,
        env=env,
        cwd=str(tmp_path),
    )
    assert r.returncode == 1


def test_doctor_output_contains_expected_sections():
    """doctor prints ffmpeg and provider check headers."""
    project_root = str(Path(__file__).parent.parent)
    env = {**os.environ, "PYTHONPATH": project_root}
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli", "doctor"],
        capture_output=True, text=True, timeout=10,
        env=env,
    )
    combined = r.stdout + r.stderr
    assert "ffmpeg" in combined.lower()
    assert ".env" in combined.lower()


# ---------------------------------------------------------------------------
# Subprocess: vidlizer setup
# ---------------------------------------------------------------------------

def test_setup_writes_env_file(tmp_path):
    """setup detects OpenRouter via fake key, writes .env to VIDLIZER_CONFIG_DIR."""
    # 3× Enter = skip port retries for offline local providers
    # "1" + Enter = select OR (detected candidate)
    # Enter = keep current key, Enter = pick model 1 (gemini-2.5-flash)
    piped = "\n\n\n1\n\n\n"
    project_root = str(Path(__file__).parent.parent)
    env = {
        **os.environ,
        "OPENROUTER_API_KEY": "sk-test-fake",
        "PYTHONPATH": project_root,
        "VIDLIZER_CONFIG_DIR": str(tmp_path),  # redirect config write
    }
    r = subprocess.run(
        [sys.executable, "-m", "vidlizer.cli", "setup"],
        input=piped, capture_output=True, text=True, timeout=15,
        env=env, cwd=str(tmp_path),
    )
    env_file = tmp_path / ".env"
    assert env_file.exists(), f"No .env written. stderr:\n{r.stderr[-1000:]}"
    assert "PROVIDER" in env_file.read_text()
