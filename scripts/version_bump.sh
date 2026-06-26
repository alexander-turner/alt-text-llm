#!/bin/bash
# Decide the next release version from Conventional Commits and write it to
# alt_text_llm/__init__.py, so releases don't require a manual version edit.
#
# Adapted from alexander-turner/punctilio's auto-version workflow for this
# Python/PyPI package. The bump level is decided deterministically by parsing
# Conventional Commits since the last release tag; PyPI is the source of truth
# for the current version. This script only *computes and writes* the version
# and emits GitHub Actions outputs — building, publishing (OIDC trusted
# publishing), tagging, and the release-docs commit happen in the workflow.
#
# Outputs (to $GITHUB_OUTPUT when set, else stdout):
#   version=<X.Y.Z>          the new version written to __init__.py
#   should_publish=true|false whether the workflow should build & publish
#
# All diagnostics go to stderr so stdout stays clean.
set -euo pipefail

log() { echo "$@" >&2; }

VERSION_FILE="alt_text_llm/__init__.py"

emit() {
  # Append a key=value line to the GitHub Actions output file when present.
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "$1=$2" >> "$GITHUB_OUTPUT"
  else
    echo "$1=$2"
  fi
}

# Print the semver bump level. $1: commit subject lines (`%s`, one per line) —
# only these are checked for type prefixes, so prose in a commit body that
# happens to start with `feat:` can't inflate the bump. $2: full messages
# (`%B`), scanned only for BREAKING CHANGE footers. Per Conventional Commits:
# - any `type!:` / `type(scope)!:` subject or `BREAKING CHANGE:` footer -> major
# - else any `feat:` / `feat(scope):` subject -> minor
# - else (including commits with no conventional prefix at all) -> patch
determine_bump() {
  local subjects="$1" full_messages="$2"
  if grep -Eq '^[a-zA-Z]+(\([^)]*\))?!:' <<< "$subjects" \
    || grep -Eq '^BREAKING[- ]CHANGE:' <<< "$full_messages"; then
    echo "major"
  elif grep -Eq '^feat(\([^)]*\))?:' <<< "$subjects"; then
    echo "minor"
  else
    if ! grep -Eq '^[a-zA-Z]+(\([^)]*\))?:' <<< "$subjects"; then
      log "No Conventional Commits prefixes found; defaulting to patch."
    fi
    echo "patch"
  fi
}

PACKAGE_NAME=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['name'])")
COMMITTED_VERSION=$(grep -E '^__version__' "$VERSION_FILE" | sed -E 's/.*"([^"]+)".*/\1/')

# Current version: the highest of PyPI's latest, the latest release tag, and
# the committed __version__.
#
# Using the MAX (not just PyPI) is essential: PyPI's JSON API lags behind
# uploads (Fastly CDN), so when two releases land back-to-back the second run
# can still read the pre-first-release version from PyPI and recompute a
# version that was *just* published — which then fails upload with
# "400 File already exists". The git tag created at release time does not lag,
# so folding it into the base prevents that race.
PYPI_JSON=$(curl -fsS "https://pypi.org/pypi/${PACKAGE_NAME}/json" 2>/dev/null || echo "")
PYPI_VERSION=""
if [ -n "$PYPI_JSON" ]; then
  PYPI_VERSION=$(echo "$PYPI_JSON" | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])")
  log "Latest PyPI version: ${PYPI_VERSION:-<none>}"
else
  log "PyPI unreachable or package unpublished."
fi
LATEST_TAG_VERSION=$(git tag --list 'v*' | sed 's/^v//' \
  | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
log "Latest release tag version: ${LATEST_TAG_VERSION:-<none>}"
CURRENT_VERSION=$(printf '%s\n%s\n%s\n' \
  "$PYPI_VERSION" "$LATEST_TAG_VERSION" "$COMMITTED_VERSION" \
  | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
log "Basing next version on: $CURRENT_VERSION"

# Determine which commits to analyze, from the last release tag (vX.Y.Z).
LAST_TAG=$(git describe --tags --match "v*" --abbrev=0 HEAD 2>/dev/null || echo "")
if [ -n "$LAST_TAG" ]; then
  if [ "$(git rev-list -1 "$LAST_TAG")" = "$(git rev-parse HEAD)" ]; then
    log "No new commits since $LAST_TAG. Skipping."
    emit version "$CURRENT_VERSION"
    emit should_publish "false"
    exit 0
  fi
  COMMIT_SUBJECTS=$(git log "$LAST_TAG"..HEAD --pretty=format:%s --no-merges)
  COMMIT_MESSAGES=$(git log "$LAST_TAG"..HEAD --pretty=format:%B --no-merges)
else
  log "No vX.Y.Z tags found; analyzing the most recent commits."
  COMMIT_SUBJECTS=$(git log --pretty=format:%s --no-merges -20)
  COMMIT_MESSAGES=$(git log --pretty=format:%B --no-merges -20)
fi

if [ -z "$COMMIT_SUBJECTS" ]; then
  log "No commits to analyze. Skipping."
  emit version "$CURRENT_VERSION"
  emit should_publish "false"
  exit 0
fi

BUMP=$(determine_bump "$COMMIT_SUBJECTS" "$COMMIT_MESSAGES")
log "Conventional Commits bump level: $BUMP"

IFS='.' read -r MAJOR MINOR PATCH_NUM <<< "$CURRENT_VERSION"
case "$BUMP" in
  major) NEW_VERSION="$((MAJOR + 1)).0.0" ;;
  minor) NEW_VERSION="${MAJOR}.$((MINOR + 1)).0" ;;
  patch) NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH_NUM + 1))" ;;
esac

if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  log "Error: computed an invalid version: $NEW_VERSION"
  exit 1
fi
log "New version: $NEW_VERSION"

# Safety net: if the computed version somehow already exists on PyPI, skip
# rather than fail the publish step.
if [ -n "$PYPI_JSON" ] && echo "$PYPI_JSON" \
  | python -c "import json,sys; sys.exit(0 if '${NEW_VERSION}' in json.load(sys.stdin).get('releases',{}) else 1)"; then
  log "Version $NEW_VERSION already exists on PyPI. Skipping."
  emit version "$NEW_VERSION"
  emit should_publish "false"
  exit 0
fi

# Write the new version into the package so `python -m build` picks it up.
NEW_VERSION="$NEW_VERSION" python - "$VERSION_FILE" <<'PY'
import os, re, sys
path = sys.argv[1]
new = os.environ["NEW_VERSION"]
text = open(path, encoding="utf-8").read()
text, n = re.subn(r'^__version__ = "[^"]+"', f'__version__ = "{new}"', text, count=1, flags=re.M)
if n != 1:
    sys.exit(f"Could not update __version__ in {path}")
open(path, "w", encoding="utf-8").write(text)
PY
log "Wrote __version__ = $NEW_VERSION to $VERSION_FILE"

emit version "$NEW_VERSION"
emit should_publish "true"
