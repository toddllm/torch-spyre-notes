#!/usr/bin/env bash
#
# needs-pod: run the test_flash cold compile with the three focused
# instrumentation stubs from measurements/2026-08-20/patches/ installed
# simultaneously:
#
#   - instrument_build_indirect_load_subs.py
#       target: torch_spyre._inductor.pass_utils._build_indirect_load_subs
#   - instrument_get_fill_order.py
#       target: torch._inductor.ir.ComputedBuffer.get_fill_order
#   - instrument_iteration_space.py
#       target: torch_spyre._inductor.pass_utils.iteration_space_from_op
#
# Each stub writes ONE JSONL row per call carrying elapsed_us + input
# signature (arity, kwarg_names, op_pyid, operation_name, op_type),
# plus one-frame-up caller info. The stubs share a schema so the
# resulting three logs can be concatenated and consumed by
# measurements/2026-08-20/scripts/analyze_profile.py in compat mode.
#
# Laptop cannot run this — no Spyre device, no torch-spyre install. Run
# on a dev pod with the pinned Torch-Spyre build at
#   fea0c4be901e1383b1f700dbad8887128b0fcb27
# (or compare on newer tips; record the actual git rev in the summary).
#
# Log destinations (override with env vars if needed):
#   $LOG_BUILD_INDIRECT     default /tmp/torch_spyre_build_indirect_load_subs.jsonl
#   $LOG_GET_FILL_ORDER     default /tmp/torch_spyre_get_fill_order.jsonl
#   $LOG_ITERATION_SPACE    default /tmp/torch_spyre_iteration_space.jsonl
#
# NOTE on how the three stubs are loaded together:
#   Each stub reads TORCH_SPYRE_INSTRUMENT_LOG at module-import time
#   and hard-codes that path for its file handle. To get three separate
#   logs in one process we cannot just set one env var — we import each
#   stub in a separate step of a small bootstrap that rewrites the
#   module-level LOG_PATH before install(). The bootstrap Python here
#   does exactly that. If you would rather run three separate compiles
#   (one per stub), just export TORCH_SPYRE_INSTRUMENT_LOG=... and
#   `python -c "import instrument_<name>"` three times.

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found" >&2
    exit 1
fi

: "${AUDIT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${TORCHINDUCTOR_CACHE_DIR:=/tmp/torchinductor_hot_callers}"
: "${LOG_BUILD_INDIRECT:=/tmp/torch_spyre_build_indirect_load_subs.jsonl}"
: "${LOG_GET_FILL_ORDER:=/tmp/torch_spyre_get_fill_order.jsonl}"
: "${LOG_ITERATION_SPACE:=/tmp/torch_spyre_iteration_space.jsonl}"

export TORCHINDUCTOR_CACHE_DIR

echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "LOG_BUILD_INDIRECT=$LOG_BUILD_INDIRECT"
echo "LOG_GET_FILL_ORDER=$LOG_GET_FILL_ORDER"
echo "LOG_ITERATION_SPACE=$LOG_ITERATION_SPACE"

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

PATCH_DIR="$AUDIT_ROOT/measurements/2026-08-20/patches"
HARNESS="$AUDIT_ROOT/measurements/2026-08-20/scripts/run_test_flash.py"

for f in \
    "$PATCH_DIR/instrument_build_indirect_load_subs.py" \
    "$PATCH_DIR/instrument_get_fill_order.py" \
    "$PATCH_DIR/instrument_iteration_space.py" \
    "$HARNESS"; do
    if [[ ! -f "$f" ]]; then
        echo "expected file missing: $f" >&2
        exit 2
    fi
done

# Bootstrap: pre-load the three stubs, each pointed at its own log
# file, THEN exec runpy on the harness so the harness's own
# imports (including its own instrument_read_writes side-effect)
# find the wrappers already installed.
export PATCH_DIR HARNESS LOG_BUILD_INDIRECT LOG_GET_FILL_ORDER LOG_ITERATION_SPACE

python3 - <<'PYRUN'
import importlib.util
import os
import runpy
import sys

PATCH_DIR = os.environ["PATCH_DIR"]
HARNESS = os.environ["HARNESS"]
LOGS = {
    "instrument_build_indirect_load_subs": os.environ["LOG_BUILD_INDIRECT"],
    "instrument_get_fill_order":            os.environ["LOG_GET_FILL_ORDER"],
    "instrument_iteration_space":           os.environ["LOG_ITERATION_SPACE"],
}

sys.path.insert(0, PATCH_DIR)

for mod_name, log_path in LOGS.items():
    # Set the env var each stub reads at import time.
    os.environ["TORCH_SPYRE_INSTRUMENT_LOG"] = log_path
    # Fresh import each time so the top-level install() side-effect
    # runs against the freshly-set env var and target.
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        sys.stderr.write(f"[bootstrap] cannot find {mod_name}\n")
        sys.exit(3)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)   # runs install() at import time
    sys.stderr.write(f"[bootstrap] installed {mod_name}, log_path={log_path}\n")

# Restore a sensible fallback for the harness's own instrument path.
os.environ["TORCH_SPYRE_INSTRUMENT_LOG"] = "/tmp/torch_spyre_read_writes_hotcallers.jsonl"

sys.stderr.write(f"[bootstrap] running harness {HARNESS}\n")
runpy.run_path(HARNESS, run_name="__main__")
PYRUN

# --- summarize per-log via analyze_profile.py compat mode --------------------

ANALYZER="$AUDIT_ROOT/measurements/2026-08-20/scripts/analyze_profile.py"
if [[ -f "$ANALYZER" ]]; then
    for log in "$LOG_BUILD_INDIRECT" "$LOG_GET_FILL_ORDER" "$LOG_ITERATION_SPACE"; do
        out="${log%.jsonl}.analyze.txt"
        if [[ -s "$log" ]]; then
            echo "analyzing $log -> $out"
            python3 "$ANALYZER" "$log" > "$out" 2>&1 || {
                echo "analyzer failed on $log (see $out)" >&2
            }
            tail -n 20 "$out" || true
            echo "----"
        else
            echo "log is empty: $log (target may not have been reached)"
        fi
    done
else
    echo "note: analyzer script not present at $ANALYZER; skipping compat report" >&2
fi

echo "done. inspect:"
echo "  $LOG_BUILD_INDIRECT"
echo "  $LOG_GET_FILL_ORDER"
echo "  $LOG_ITERATION_SPACE"
