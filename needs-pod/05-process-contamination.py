#!/usr/bin/env python3
"""
Process-contamination metamorphic test.

Companion to findings/upstream-fragility/04-monkey-patches-outside-patches-py.md
and findings/upstream-fragility/03-lowering-registry-lock.md.

The audit claim under test: torch_spyre installs process-global mutations
into upstream Inductor at *import time* and inside ``enable_spyre_context``,
which means a CPU-only compile in a process that has imported torch_spyre —
or that has just finished a Spyre compile — sees a different Inductor
state than a fresh process would.

Three sub-scenarios, each run in a fresh child process for isolation:

  1. baseline-fresh          — no torch_spyre imported. Snapshot Inductor
                                state, compile a CPU graph, snapshot again.
  2. imported-only           — ``import torch_spyre`` (triggers _autoload).
                                Snapshot Inductor state, compile a CPU graph,
                                snapshot again.
  3. after-spyre-compile     — import torch_spyre, run a *Spyre* compile
                                (this enters and exits ``enable_spyre_context``,
                                which is supposed to restore state on exit),
                                then compile a CPU graph. Snapshot before
                                and after the CPU compile.

For each scenario we snapshot a fixed set of *upstream* Inductor state
before and after the CPU compile and diff each field. The state fields:

  torch._inductor.lowering.lowerings           (dict: op → callable)
  torch._inductor.lowering.fallbacks           (set of ops)
  torch._inductor.fx_passes.joint_graph.pass_patterns   (list length)
  torch._inductor.fx_passes.post_grad.pass_patterns     (list length)
  torch._inductor.fx_passes.reinplace.inplaceable_ops   (dict length)
  torch._prims_common._computation_dtype_map   (dict)
  torch._dynamo.config.cache_size_limit        (int)
  torch._inductor.ir.Loops.has_large_inner_fn  (callable identity)
  torch._inductor.graph.GraphLowering._update_scheduler (callable identity)
  torch._inductor.scheduler.SchedulerNode.has_side_effects (callable identity)

The identity we care about is the address of the function object
(``id(fn)``) — if torch_spyre replaced it and never restored, the id
after CPU compile differs from a fresh baseline.

Results are written to ``results/process-contamination.jsonl`` next to
this file (or to $TORCH_SPYRE_NOTES_RESULTS if set).

Dry mode:
  --dry runs each scenario in-process without importing torch_spyre or
  torch. Fakes are used for snapshot targets. Verifies the diffing and
  child-process JSONL plumbing.

Usage (on pod):
    python3 needs-pod/05-process-contamination.py

Usage (laptop, dry):
    python3 needs-pod/05-process-contamination.py --dry
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = HERE / "results" / "process-contamination.jsonl"


# ─── snapshot target definitions ─────────────────────────────────────

# Each snapshot function returns a JSON-serializable dict. All snapshot
# functions guard reads with try/except so a missing symbol on an
# unusual torch build reports 'missing' rather than crashing.

SNAPSHOT_CODE = textwrap.dedent(r"""
    def snapshot_inductor_state():
        import json
        result = {}

        def _get(path):
            try:
                obj = __import__(path.split('.', 1)[0])
                for p in path.split('.')[1:]:
                    obj = getattr(obj, p)
                return obj
            except Exception as e:
                return f"MISSING: {e!r}"

        # (name, callable-that-produces-a-JSON-friendly-value)
        specs = [
            ("lowerings_count",
                lambda: len(_get('torch._inductor.lowering.lowerings'))),
            ("fallbacks_count",
                lambda: len(_get('torch._inductor.lowering.fallbacks'))),
            ("joint_pass_patterns_len",
                lambda: len(_get('torch._inductor.fx_passes.joint_graph.pass_patterns'))),
            ("post_grad_pass_patterns_len",
                lambda: len(_get('torch._inductor.fx_passes.post_grad.pass_patterns'))),
            ("inplaceable_ops_count",
                lambda: len(_get('torch._inductor.fx_passes.reinplace.inplaceable_ops'))),
            ("computation_dtype_map",
                lambda: {str(k): str(v) for k, v in _get('torch._prims_common._computation_dtype_map').items()}),
            ("dynamo_cache_size_limit",
                lambda: _get('torch._dynamo.config.cache_size_limit')),
            ("Loops_has_large_inner_fn_id",
                lambda: id(_get('torch._inductor.ir.Loops.has_large_inner_fn'))),
            ("GraphLowering_update_scheduler_id",
                lambda: id(_get('torch._inductor.graph.GraphLowering._update_scheduler'))),
            ("SchedulerNode_has_side_effects_id",
                lambda: id(_get('torch._inductor.scheduler.SchedulerNode.has_side_effects'))),
            ("Tensor_to_id",
                lambda: id(_get('torch.Tensor.to'))),
            ("torch_empty_id",
                lambda: id(_get('torch.empty'))),
            ("Tensor_repr_id",
                lambda: id(_get('torch.Tensor.__repr__'))),
            ("FxGraphHashDetails_init_id",
                lambda: id(_get('torch._inductor.codecache.FxGraphHashDetails.__init__'))),
        ]

        for name, thunk in specs:
            try:
                result[name] = thunk()
            except Exception as e:
                result[name] = f"THUNK-ERROR: {e!r}"
        return result
""").strip()


# ─── child-process runners ────────────────────────────────────────────

RUNNER_TEMPLATE = textwrap.dedent(r"""
    import json, sys, traceback

    {snapshot_code}

    scenario = {scenario!r}
    output_path = {output_path!r}

    def _emit(kind, **fields):
        rec = {{"scenario": scenario, "kind": kind, **fields}}
        with open(output_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    try:
        {setup}
        _emit("snapshot", when="before-cpu-compile", state=snapshot_inductor_state())

        import torch
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(16, 32, bias=False)
                self.b = torch.nn.Linear(32, 16, bias=False)
            def forward(self, x):
                return self.b(torch.nn.functional.relu(self.a(x)))
        m = M()
        x = torch.randn(4, 16)
        y = torch.compile(m)(x)
        _emit("cpu-compile", ok=True, out_sum=float(y.sum().item()))
        _emit("snapshot", when="after-cpu-compile", state=snapshot_inductor_state())
    except Exception:
        _emit("cpu-compile", ok=False, traceback=traceback.format_exc())
""").strip()


# For scenario 3, run a Spyre compile before the CPU compile.
SPYRE_COMPILE_SETUP = textwrap.dedent(r"""
    import torch
    import torch_spyre  # noqa: F401
    class SpyreM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(16, 32, bias=False)
        def forward(self, x):
            return torch.nn.functional.relu(self.a(x))
    _sm = SpyreM().to("spyre")
    _sx = torch.randn(4, 16, device="spyre")
    try:
        _ = torch.compile(_sm)(_sx)
        _emit("spyre-compile", ok=True)
    except Exception:
        _emit("spyre-compile", ok=False, traceback=traceback.format_exc())
""").strip()


def _script_for(scenario: str, output_path: Path) -> str:
    if scenario == "baseline-fresh":
        setup = "# no torch_spyre import"
    elif scenario == "imported-only":
        setup = "import torch_spyre  # noqa: F401  # triggers _autoload_impl"
    elif scenario == "after-spyre-compile":
        setup = SPYRE_COMPILE_SETUP
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return RUNNER_TEMPLATE.format(
        snapshot_code=SNAPSHOT_CODE,
        scenario=scenario,
        output_path=str(output_path),
        setup=setup,
    )


def _run_scenario_child(scenario: str, output_path: Path) -> int:
    """Run one scenario in a fresh Python child process for isolation."""
    script = _script_for(scenario, output_path)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        script_path = f.name

    try:
        # Inherit env; the pod is expected to have torch_spyre importable.
        completed = subprocess.run(
            [sys.executable, script_path],
            check=False,
            timeout=600,
            capture_output=True,
        )
        with open(output_path, "a") as fh:
            fh.write(json.dumps({
                "scenario": scenario,
                "kind": "child-process-exit",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-2000:],
            }) + "\n")
        return completed.returncode
    except subprocess.TimeoutExpired:
        with open(output_path, "a") as fh:
            fh.write(json.dumps({
                "scenario": scenario,
                "kind": "child-process-timeout",
            }) + "\n")
        return 124
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ─── diff computation ────────────────────────────────────────────────

def _diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    diff = {}
    for k in keys:
        b = before.get(k)
        a = after.get(k)
        if b != a:
            diff[k] = {"before": b, "after": a}
    return diff


def _diff_across_scenarios(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a cross-scenario diff: baseline vs imported-only vs after-spyre."""
    by_scenario_when = {}
    for r in records:
        if r.get("kind") == "snapshot":
            by_scenario_when.setdefault(r["scenario"], {})[r["when"]] = r["state"]

    result = {}
    for when in ("before-cpu-compile", "after-cpu-compile"):
        baseline = by_scenario_when.get("baseline-fresh", {}).get(when)
        if baseline is None:
            continue
        for scen in ("imported-only", "after-spyre-compile"):
            other = by_scenario_when.get(scen, {}).get(when)
            if other is None:
                continue
            result[f"{scen}_vs_baseline_{when}"] = _diff_snapshots(baseline, other)
    return result


# ─── dry mode ────────────────────────────────────────────────────────

def _run_dry(output_path: Path) -> None:
    """Fake three scenarios end-to-end without importing torch."""

    def fake_snapshot(scenario_label: str) -> dict[str, Any]:
        # Baseline values.
        s = {
            "lowerings_count": 500,
            "fallbacks_count": 40,
            "joint_pass_patterns_len": 2,
            "post_grad_pass_patterns_len": 3,
            "inplaceable_ops_count": 10,
            "computation_dtype_map": {"torch.bfloat16": "torch.float32"},
            "dynamo_cache_size_limit": 8,
            "Loops_has_large_inner_fn_id": 111111,
            "GraphLowering_update_scheduler_id": 222222,
            "SchedulerNode_has_side_effects_id": 333333,
            "Tensor_to_id": 444444,
            "torch_empty_id": 555555,
            "Tensor_repr_id": 666666,
            "FxGraphHashDetails_init_id": 777777,
        }
        # Fake the effects of import-only vs after-spyre-compile.
        if scenario_label == "imported-only":
            s["lowerings_count"] = 520          # inplaceable_ops write etc.
            s["inplaceable_ops_count"] = 11
            s["computation_dtype_map"] = {"torch.bfloat16": "torch.bfloat16"}
            s["dynamo_cache_size_limit"] = 1024
            s["Tensor_to_id"] = 999999          # monkey-patched
            s["torch_empty_id"] = 888888        # monkey-patched
            s["Tensor_repr_id"] = 777770        # monkey-patched
            s["FxGraphHashDetails_init_id"] = 666660  # monkey-patched
        elif scenario_label == "after-spyre-compile":
            # Same as imported-only after teardown, IF teardown is clean.
            s["lowerings_count"] = 520
            s["inplaceable_ops_count"] = 11
            s["computation_dtype_map"] = {"torch.bfloat16": "torch.bfloat16"}
            s["dynamo_cache_size_limit"] = 1024
            s["Tensor_to_id"] = 999999
            s["torch_empty_id"] = 888888
            s["Tensor_repr_id"] = 777770
            s["FxGraphHashDetails_init_id"] = 666660
            # Simulate a leaked mutation (e.g., pass_patterns not restored).
            s["joint_pass_patterns_len"] = 1
        return s

    for scenario in ("baseline-fresh", "imported-only", "after-spyre-compile"):
        with open(output_path, "a") as fh:
            fh.write(json.dumps({
                "scenario": scenario,
                "kind": "snapshot",
                "when": "before-cpu-compile",
                "state": fake_snapshot(scenario),
            }) + "\n")
            fh.write(json.dumps({
                "scenario": scenario,
                "kind": "cpu-compile",
                "ok": True,
                "out_sum": 3.14,
            }) + "\n")
            fh.write(json.dumps({
                "scenario": scenario,
                "kind": "snapshot",
                "when": "after-cpu-compile",
                "state": fake_snapshot(scenario),
            }) + "\n")


# ─── entrypoint ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="run each scenario in-process with a fake snapshot (no torch)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("TORCH_SPYRE_NOTES_RESULTS", DEFAULT_RESULTS_PATH)),
        help="output JSONL path",
    )
    parser.add_argument(
        "--only",
        choices=("baseline-fresh", "imported-only", "after-spyre-compile"),
        help="run just one scenario (live mode only)",
    )
    args = parser.parse_args(argv)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    # Truncate for a clean run.
    open(args.results, "w").close()

    with open(args.results, "a") as fh:
        fh.write(json.dumps({
            "kind": "run-header",
            "mode": "dry" if args.dry else "live",
            "python": sys.version.split()[0],
            "argv": argv or sys.argv,
            "cwd": os.getcwd(),
        }) + "\n")

    if args.dry:
        _run_dry(args.results)
        print(f"[dry] wrote {args.results}", file=sys.stderr)
    else:
        scenarios = [args.only] if args.only else [
            "baseline-fresh", "imported-only", "after-spyre-compile"
        ]
        for scen in scenarios:
            rc = _run_scenario_child(scen, args.results)
            if rc != 0:
                print(f"scenario {scen} exited {rc}", file=sys.stderr)

    # Compute cross-scenario diff and write it as one final record.
    records = []
    with open(args.results) as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    diff = _diff_across_scenarios(records)
    with open(args.results, "a") as fh:
        fh.write(json.dumps({"kind": "cross-scenario-diff", "diff": diff}, default=str) + "\n")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
