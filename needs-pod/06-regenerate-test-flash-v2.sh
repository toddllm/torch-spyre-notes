#!/usr/bin/env bash
#
# needs-pod: regenerate measurements/2026-08-20/data/test_flash.jsonl with
# the v2 instrumentation schema.
#
# Background
# ----------
# The checked-in test_flash.jsonl was produced against an older version of
# measurements/2026-08-20/patches/instrument_read_writes.py whose records
# lacked the identity + bookkeeping fields required by the current
# measurements/2026-08-20/scripts/analyze.py:
#
#   - op_pyid, operation_name, op_type, seq, depth
#   - kind ∈ {raw_get_read_writes, memo_wrapper_hit,
#             memo_wrapper_miss_inclusive, memo_wrapper_overhead}
#   - a leading `instrument_installed` record carrying schema_version=2
#
# analyze.py runs the old JSONL in legacy/inclusive-only mode. To get the
# exclusive-vs-inclusive tables the finding docs quote, the log must be
# regenerated with the v2 patch. This script does that on a pod that
# actually has torch_spyre installed.
#
# Laptop cannot run this — no Spyre device, no torch-spyre install. Run on
# a dev pod with the pinned Torch-Spyre build at
#   fea0c4be901e1383b1f700dbad8887128b0fcb27
# (or compare on newer tips; record the actual git rev in the summary).
#
# Environment variables:
#   TORCH_SPYRE_ROOT   path to a torch-spyre worktree at the pinned SHA
#                      (default: $HOME/torch-spyre-work/torch-spyre)
#   OUT_DIR            where to write test_flash_v2.jsonl
#                      (default: $PWD/measurements/$(date +%Y-%m-%d)/data)
#   AUDIT_ROOT         torch-spyre-notes checkout root (auto-detected
#                      from this script's location)
#
# Cold-compile hygiene: TORCHINDUCTOR_CACHE_DIR is set to a fresh mktemp
# directory inside this script and exported to the harness.

set -euo pipefail

# --- paths -------------------------------------------------------------------

: "${AUDIT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${TORCH_SPYRE_ROOT:=$HOME/torch-spyre-work/torch-spyre}"
: "${OUT_DIR:=$PWD/measurements/$(date +%Y-%m-%d)/data}"

PATCH_SRC="$AUDIT_ROOT/measurements/2026-08-20/patches/instrument_read_writes.py"
HARNESS="$AUDIT_ROOT/measurements/2026-08-20/scripts/run_test_flash.py"
ANALYZER="$AUDIT_ROOT/measurements/2026-08-20/scripts/analyze.py"

echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "TORCH_SPYRE_ROOT=$TORCH_SPYRE_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "PATCH_SRC=$PATCH_SRC"
echo "HARNESS=$HARNESS"

for f in "$PATCH_SRC" "$HARNESS" "$ANALYZER"; do
    if [[ ! -f "$f" ]]; then
        echo "expected file missing: $f" >&2
        exit 2
    fi
done

mkdir -p "$OUT_DIR"

# --- step 1: prove torch_spyre._C loads --------------------------------------
#
# On dev pods where the flex-ABI symbol mismatch is present, `import
# torch_spyre._C` raises an undefined-symbol ImportError even though
# `import torch_spyre` at the pure-Python level succeeds. Fail closed
# with the recipe rather than silently running against a stub.

echo "[1/6] verifying torch_spyre._C import"
if ! python3 -c "import torch_spyre._C" 2>/tmp/_ts_c_import.err; then
    cat /tmp/_ts_c_import.err >&2 || true
    cat >&2 <<'EOF'

FATAL: `import torch_spyre._C` failed. This usually means the flex-ABI
symbol mismatch has bitten again — the venv's torch_spyre wheel was
built against a torch whose C++ ABI no longer matches the runtime.

Rebuild recipe (run inside the torch-spyre venv on the dev pod):

    cd "$TORCH_SPYRE_ROOT"
    source .venv/bin/activate
    pip install --force-reinstall --no-deps -e .
    # if that still fails, rebuild the wheel from scratch:
    pip uninstall -y torch_spyre
    pip install --no-build-isolation -e .

Then re-run this script.
EOF
    exit 3
fi
echo "[1/6] torch_spyre._C import OK"

# --- step 2: confirm the pinned SHA -----------------------------------------

echo "[2/6] verifying torch-spyre SHA in $TORCH_SPYRE_ROOT"
if [[ ! -d "$TORCH_SPYRE_ROOT/.git" ]]; then
    echo "FATAL: $TORCH_SPYRE_ROOT is not a git checkout" >&2
    exit 4
fi
HEAD_SHA=$(cd "$TORCH_SPYRE_ROOT" && git rev-parse HEAD)
if [[ "$HEAD_SHA" != fea0c4b* ]]; then
    cat >&2 <<EOF
FATAL: expected torch-spyre HEAD to start with fea0c4b, got:
    $HEAD_SHA

Checkout recipe (run inside $TORCH_SPYRE_ROOT):

    cd "$TORCH_SPYRE_ROOT"
    git fetch origin fea0c4be901e1383b1f700dbad8887128b0fcb27
    git checkout fea0c4be901e1383b1f700dbad8887128b0fcb27
    # rebuild the extension if you switched SHAs:
    pip install --force-reinstall --no-deps -e .

Then re-run this script.
EOF
    exit 5
fi
echo "[2/6] HEAD=$HEAD_SHA (matches fea0c4b)"

# --- step 3: stage patch + harness ------------------------------------------
#
# The harness at measurements/2026-08-20/scripts/run_test_flash.py imports
# instrument_read_writes.py from ../patches at module-import time (it
# inserts the sibling patches/ directory onto sys.path — see
# `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
# "patches"))` near the top of run_test_flash.py). So we don't need
# TORCH_INDUCTOR_PATCH_FN or any --instrument flag — pointing the harness
# at the AUDIT_ROOT copy of patches/instrument_read_writes.py is enough.
#
# For extra safety, if TORCH_SPYRE_ROOT is being used as a staging dir
# for the patch, drop a working copy there as well. This is a no-op when
# the harness already resolves the patch through AUDIT_ROOT.

echo "[3/6] staging v2 instrument patch"
echo "     patch source: $PATCH_SRC"
grep -q 'schema_version.*2' "$PATCH_SRC" || {
    echo "FATAL: patch at $PATCH_SRC does not declare schema_version=2" >&2
    exit 6
}
echo "[3/6] patch declares schema_version=2"

# --- step 4: fresh Inductor cache dir ---------------------------------------

TORCHINDUCTOR_CACHE_DIR="$(mktemp -d -t torchinductor_test_flash_v2.XXXXXX)"
export TORCHINDUCTOR_CACHE_DIR
echo "[4/6] TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR (fresh, cold-compile hygiene)"

# --- step 5: run harness, land JSONL under $OUT_DIR -------------------------

OUT_JSONL="$OUT_DIR/test_flash_v2.jsonl"
export TORCH_SPYRE_INSTRUMENT_LOG="$OUT_JSONL"
echo "[5/6] running harness -> $OUT_JSONL"
python3 "$HARNESS"

if [[ ! -s "$OUT_JSONL" ]]; then
    echo "FATAL: harness completed but $OUT_JSONL is empty" >&2
    exit 7
fi

# --- step 6: one-line summary ------------------------------------------------
#
# No jq. Use grep + head + awk + sort. The `instrument_installed`
# record is always the first record and carries schema_version.

REC_COUNT=$(wc -l < "$OUT_JSONL" | tr -d ' ')
FIRST_SCHEMA=$(head -n 1 "$OUT_JSONL" | grep -o '"schema_version": *[0-9]*' | head -n1 | awk -F: '{print $2}' | tr -d ' ')
if [[ -z "$FIRST_SCHEMA" ]]; then
    FIRST_SCHEMA="MISSING"
fi

echo "[6/6] summary:"
echo "     records=$REC_COUNT  first_schema_version=$FIRST_SCHEMA"

echo "     top 3 op_pyid hotspots (raw_get_read_writes only):"
# Extract op_pyid from raw_get_read_writes records, count, sort desc, top 3.
grep '"kind": "raw_get_read_writes"' "$OUT_JSONL" \
    | grep -o '"op_pyid": "[^"]*"' \
    | sort \
    | uniq -c \
    | sort -rn \
    | head -n 3 \
    | awk '{printf "       %6d  %s\n", $1, $2 " " $3}'

# --- next step: how to derive exclusive/inclusive tables --------------------

cat <<EOF

next step — derive exclusive/inclusive tables:

    python3 "$ANALYZER" "$OUT_JSONL" | tee "${OUT_JSONL%.jsonl}_analysis.txt"

The v2 log will print an exclusive_total_ms line and a per-kind
breakdown (raw / memo_hit / memo_miss_inclusive / memo_overhead) in
addition to the legacy inclusive-only summary that the v1 log
produces.
EOF
