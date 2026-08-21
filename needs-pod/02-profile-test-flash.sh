#!/usr/bin/env bash
#
# needs-pod: wrap the test_flash cold compile with the whole-compile
# profiler at measurements/2026-08-20/scripts/profile_whole_compile.py.
#
# Laptop cannot run this — no Spyre device, no torch-spyre install. Run
# on a dev pod with the pinned Torch-Spyre build at
#   fea0c4be901e1383b1f700dbad8887128b0fcb27
# (or compare on newer tips; record the actual git rev in the summary).
#
# What this produces (under $TORCH_SPYRE_PROFILE_DIR):
#   - profile.prof         cProfile output over the whole compile
#   - phase_timings.json   per-phase call counts + elapsed_us for each
#                          of the 8 phases that resolved on this build
#                          (Dynamo, AOTAutograd, Inductor GraphLowering,
#                           torch-spyre pre-scheduling, scheduler, Spyre
#                           codegen, DeepTools/backend, first device
#                           execution). Any phase whose hook point did
#                           not resolve is logged in the "hooks" block
#                           with attached=null and a list of skipped
#                           candidates.
#   - analyze_profile.txt  Top-30 exclusive / inclusive time tables,
#                          plus HOT-AND-REPEATEDLY-CALLED cross-ref
#                          against scans/results/repeated_analysis.json.
#
# The harness driven here is measurements/2026-08-20/scripts/run_test_flash.py,
# whose closure body is a verbatim copy of the torch-spyre
# tests/inductor/test_opspec_tiling.py::TestOpSpecTiling::test_flash
# workload at the pinned SHA. Cold-compile hygiene is enforced via
# TORCHINDUCTOR_CACHE_DIR.
#
# NOTE: run_test_flash.py currently imports instrument_read_writes.py
# for its own instrumentation. That instrument still logs to
# $TORCH_SPYRE_INSTRUMENT_LOG (default /tmp/torch_spyre_read_writes.jsonl)
# and does NOT interfere with the whole-compile cProfile output — the
# two logs are independent. If you want the profile.prof to be
# free of monkeypatch overhead, run this script with
#   PROFILE_NO_INSTRUMENT=1 ./02-profile-test-flash.sh
# and swap the harness to a copy that skips the instrument_read_writes
# import (or add a guard to run_test_flash.py). Left as-is by default
# so the two data streams line up in one run.

set -euo pipefail

# --- environment guards ------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found" >&2
    exit 1
fi

: "${AUDIT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${TORCH_SPYRE_PROFILE_DIR:=/tmp/torch_spyre_whole_compile_profile}"
: "${TORCHINDUCTOR_CACHE_DIR:=/tmp/torchinductor_test_flash_profile}"
: "${TORCH_SPYRE_INSTRUMENT_LOG:=/tmp/torch_spyre_read_writes_profile.jsonl}"

export TORCH_SPYRE_PROFILE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_SPYRE_INSTRUMENT_LOG

echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "TORCH_SPYRE_PROFILE_DIR=$TORCH_SPYRE_PROFILE_DIR"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "TORCH_SPYRE_INSTRUMENT_LOG=$TORCH_SPYRE_INSTRUMENT_LOG"

mkdir -p "$TORCH_SPYRE_PROFILE_DIR"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

python3 - <<'PYCHECK'
import sys
try:
    import torch
    import torch_spyre  # noqa: F401
except ImportError as e:
    sys.stderr.write(f"needs-pod prerequisite missing: {e}\n")
    sys.exit(1)
sys.stdout.write(f"torch={torch.__version__} torch_spyre={getattr(torch_spyre, '__file__', '?')}\n")
PYCHECK

# --- run the profiler --------------------------------------------------------

PROFILER="$AUDIT_ROOT/measurements/2026-08-20/scripts/profile_whole_compile.py"
HARNESS="$AUDIT_ROOT/measurements/2026-08-20/scripts/run_test_flash.py"

if [[ ! -f "$PROFILER" ]]; then
    echo "profiler script not found: $PROFILER" >&2
    exit 2
fi
if [[ ! -f "$HARNESS" ]]; then
    echo "harness script not found: $HARNESS" >&2
    exit 2
fi

echo "beginning whole-compile profile of test_flash"
python3 "$PROFILER" --harness "$HARNESS"

# --- summarize ---------------------------------------------------------------

ANALYZER="$AUDIT_ROOT/measurements/2026-08-20/scripts/analyze_profile.py"
if [[ -f "$ANALYZER" ]]; then
    ANALYZE_OUT="$TORCH_SPYRE_PROFILE_DIR/analyze_profile.txt"
    echo "writing analyze report to $ANALYZE_OUT"
    python3 "$ANALYZER" \
        "$TORCH_SPYRE_PROFILE_DIR/profile.prof" \
        "$TORCH_SPYRE_PROFILE_DIR/phase_timings.json" \
        > "$ANALYZE_OUT" 2>&1 || {
            echo "analyze_profile.py exited non-zero (see $ANALYZE_OUT)" >&2
            exit 3
        }
    tail -n 40 "$ANALYZE_OUT" || true
else
    echo "note: analyzer script not present at $ANALYZER; skipping report" >&2
fi

echo "done. inspect:"
echo "  $TORCH_SPYRE_PROFILE_DIR/profile.prof"
echo "  $TORCH_SPYRE_PROFILE_DIR/phase_timings.json"
echo "  $TORCH_SPYRE_PROFILE_DIR/analyze_profile.txt (if analyzer was present)"
