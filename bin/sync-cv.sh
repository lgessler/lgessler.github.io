#!/usr/bin/env bash
#
# Push the canonical publications.bib to the Overleaf CV repo.
#
# content/static/publications.bib is the single source of truth: the site
# renders it via plugins/bibliography_plugin.py, and the Overleaf CV renders
# the same file via biblatex. This script copies it across and commits.
#
# It stops short of pushing. Review, then run the printed git push yourself.
#
# Usage:  bin/sync-cv.sh [path-to-cv-clone]     (default: ./cv)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$REPO_ROOT/content/static/publications.bib"
CV_DIR="${1:-$REPO_ROOT/cv}"
OVERLEAF_URL="https://git@git.overleaf.com/66d32e59e71b2e3bf7c06491"

if [[ ! -f "$SOURCE" ]]; then
    echo "error: canonical bib not found at $SOURCE" >&2
    exit 1
fi

if [[ ! -d "$CV_DIR/.git" ]]; then
    echo "error: no git clone at $CV_DIR" >&2
    echo "clone it first:  git clone $OVERLEAF_URL \"$CV_DIR\"" >&2
    exit 1
fi

# Refuse to clobber unrelated local work in the CV clone.
if ! git -C "$CV_DIR" diff --quiet -- ':!publications.bib' \
   || ! git -C "$CV_DIR" diff --cached --quiet -- ':!publications.bib'; then
    echo "error: $CV_DIR has uncommitted changes outside publications.bib." >&2
    echo "commit or stash them first, so a sync can't bury them." >&2
    exit 1
fi

echo "pulling latest from Overleaf..."
git -C "$CV_DIR" pull --ff-only

cp "$SOURCE" "$CV_DIR/publications.bib"

if git -C "$CV_DIR" diff --quiet -- publications.bib; then
    echo "publications.bib already up to date; nothing to sync."
    exit 0
fi

echo
echo "=== changes to publications.bib ==="
git -C "$CV_DIR" --no-pager diff --stat -- publications.bib
echo

git -C "$CV_DIR" add publications.bib
git -C "$CV_DIR" commit -m "Sync publications.bib from lgessler.com site repo"

echo
echo "committed locally. to publish to Overleaf, run:"
echo "    git -C \"$CV_DIR\" push"
