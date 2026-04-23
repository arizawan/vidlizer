#!/usr/bin/env bash
# Usage: ./scripts/release.sh [patch|minor|major]
# Bumps version in pyproject.toml, builds, uploads to PyPI, tags, and pushes.
set -euo pipefail

BUMP=${1:-patch}
PYPROJECT="pyproject.toml"

# ── 1. Read current version ────────────────────────────────────────────────
current=$(grep '^version = ' "$PYPROJECT" | head -1 | sed 's/version = "\(.*\)"/\1/')
IFS='.' read -r major minor patch <<< "$current"

case "$BUMP" in
  major) major=$((major+1)); minor=0; patch=0 ;;
  minor) minor=$((minor+1)); patch=0 ;;
  patch) patch=$((patch+1)) ;;
  *) echo "Usage: $0 [patch|minor|major]"; exit 1 ;;
esac

new="${major}.${minor}.${patch}"
echo "▶ Bumping $current → $new"

# ── 2. Update pyproject.toml ───────────────────────────────────────────────
sed -i '' "s/^version = \"${current}\"/version = \"${new}\"/" "$PYPROJECT"

# ── 3. Lint ────────────────────────────────────────────────────────────────
echo "▶ ruff check"
ruff check vidlizer scripts

# ── 4. Tests ───────────────────────────────────────────────────────────────
echo "▶ pytest (unit only)"
pytest -q -m "not e2e" --tb=short

# ── 5. Build ───────────────────────────────────────────────────────────────
echo "▶ build"
rm -rf dist/
python -m build --sdist --wheel

# ── 6. Upload to PyPI ──────────────────────────────────────────────────────
echo "▶ twine upload"
if [[ -z "${PYPI_PROD_TOKEN:-}" ]]; then
  echo "ERROR: PYPI_PROD_TOKEN not set. Run: source ~/.zshrc" >&2
  exit 1
fi
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_PROD_TOKEN" \
  twine upload dist/*

# ── 7. Commit, tag, push ───────────────────────────────────────────────────
echo "▶ git commit + tag v${new}"
git add "$PYPROJECT"
git commit -m "chore: release v${new}"
git tag "v${new}"
git push origin main --tags

echo "✓ Released v${new} → https://pypi.org/project/vidlizer/${new}/"
