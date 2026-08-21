#!/usr/bin/env bash
#
# needs-pod: reproduction candidate for
#   findings/correctness/01-dedup-constants-graph-output-not-observed.md
#
# Laptop cannot run this — no Spyre device, no torch-spyre install. Run on a
# dev pod with the pinned Torch-Spyre build at
# fea0c4be901e1383b1f700dbad8887128b0fcb27 (or compare on newer tips).
#
# What we are testing:
#   Whether a user-authored torch.compile function that directly calls
#   torch.ops.spyre.constant.default(...) at output positions produces
#   SpyreConstantFallback entries in V.graph.graph_outputs, and whether
#   dedup_and_promote_constants then corrupts the return contract by
#   running _drop_constant on a constant whose name is in
#   V.graph.get_output_names().
#
# Expected outcomes to record (any one is a data point):
#   1. Dynamo folds the scalar-returning custom_op at trace time and the
#      call never reaches Inductor. -> invariant preserved by upstream
#      layer, document that dependency.
#   2. Inductor's output routing wraps the scalar in a different IR node
#      (SymFloat, ComputedBuffer, etc.) before graph_outputs is
#      populated. -> invariant preserved by output routing, document
#      that dependency.
#   3. graph_outputs holds SpyreConstantFallback IR nodes and dedup runs
#      without the guard-vs-drop mismatch firing (e.g. because
#      value-groups keep each output-position node in its own group).
#      -> invariant preserved by dedup's grouping, document that.
#   4. graph_outputs holds SpyreConstantFallback IR nodes AND dedup
#      merges them AND _drop_constant runs on a graph-output constant.
#      -> genuine bug; upgrade finding status from "not-observed" to
#      "open" (or "reproduced" if a runtime failure follows).
#
# Instrumentation strategy:
#   Patch torch_spyre/_inductor/dedup_constants.py to log when the guard
#   fires vs when _drop_constant is invoked. Do NOT modify behavior in
#   the observation run.

set -euo pipefail

# --- environment guards ------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found" >&2
    exit 1
fi

python3 - <<'PYCHECK'
import sys
try:
    import torch
    import torch_spyre  # noqa: F401
except ImportError as e:
    sys.stderr.write(f"needs-pod prerequisite missing: {e}\n")
    sys.exit(1)

# Confirm the pinned SHA is what we are running against (soft check).
try:
    import torch_spyre
    ts_path = torch_spyre.__file__
    sys.stdout.write(f"torch_spyre at: {ts_path}\n")
except Exception as e:
    sys.stderr.write(f"cannot introspect torch_spyre path: {e}\n")

# Confirm spyre::constant is a public custom op at this build.
assert hasattr(torch.ops.spyre, "constant"), (
    "torch.ops.spyre.constant not registered — is torch_spyre imported?"
)
sys.stdout.write("torch.ops.spyre.constant is reachable\n")
PYCHECK

# --- the actual repro --------------------------------------------------------

python3 - <<'PYREPRO'
import logging
import torch
import torch_spyre  # ensure the custom op is registered

# Turn on the dedup-pass debug logger — its "skipping output constant %s"
# message is the direct signal that the guard fired.
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("torch_spyre._inductor.dedup_constants").setLevel(logging.DEBUG)


def f(x):
    # Two direct calls with identical (value, dtype, device). If Dynamo
    # does not fold these and Inductor keeps them as separate FX nodes,
    # dedup_and_promote_constants should group them and try to merge
    # them. Returning both places their IR outputs into graph_outputs.
    c1 = torch.ops.spyre.constant.default(
        1.0, torch.float16, torch.device("spyre")
    )
    c2 = torch.ops.spyre.constant.default(
        1.0, torch.float16, torch.device("spyre")
    )
    return c1, c2, x + c1


compiled = torch.compile(f, backend="spyre")
x = torch.zeros((4,), dtype=torch.float16, device="spyre")

try:
    out = compiled(x)
    print("OK: compiled call returned", out)
except Exception as e:
    # Codegen failure or runtime error on a retired buffer name is the
    # bug fingerprint. Capture the full traceback so we can classify.
    import traceback
    print("FAIL:")
    traceback.print_exc()
    raise
PYREPRO

# --- optional follow-up ------------------------------------------------------
#
# If the run above did not surface the bug, try variants:
#   - Force distinct FX nodes by wrapping each call in a no-op op that
#     Inductor cannot fuse (e.g. explicit .detach() at a different
#     rank).
#   - Emit N > 2 duplicates so any single-slot canonicalization is
#     bypassed.
#   - Emit the pair inside a subgraph whose outputs are then re-emitted
#     as the outer graph's outputs (HigherOrderOp shape).
#
# Each variant should be reproduced with the debug logger on and its
# output attached to a comment on the finding file.
