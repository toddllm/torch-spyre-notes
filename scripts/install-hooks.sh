#!/usr/bin/env bash
# Install the repo's git hooks by pointing `core.hooksPath` at
# `.githooks/`. Run once per fresh clone; it is idempotent.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"

git config core.hooksPath .githooks

echo "install-hooks: core.hooksPath set to .githooks"
echo "install-hooks: hooks available in .githooks:"
ls -la .githooks
