"""Unbiased whole-compile profiler for torch-spyre.

Wraps a torch.compile invocation with:

- **cProfile** around the whole compile call (writes ``profile.prof``).
- **Phase-boundary timers** around a curated set of upstream/torch-spyre
  entry points that map to the eight phases of a Spyre compile:

    1. Dynamo               -> ``torch._dynamo.convert_frame.convert_frame``
    2. AOTAutograd          -> ``torch._functorch.aot_autograd.aot_module_simplified``
    3. Inductor GraphLowering -> ``torch._inductor.graph.GraphLowering.compile_to_module``
    4. torch-spyre pre-scheduling
                             -> ``torch_spyre._inductor.pass_wrapper.run_passes``
                                (fallback: ``torch_spyre._inductor.spyre_scheduling.pre_scheduling``)
    5. Inductor Scheduler   -> ``torch._inductor.scheduler.Scheduler.__init__``
    6. Spyre codegen        -> ``torch_spyre._inductor.codegen.wrapper.SpyreWrapperCodegen.generate``
                                (fallback: ``torch_spyre._inductor.codegen.SpyreScheduling.codegen``)
    7. DeepTools / backend  -> ``torch_spyre._inductor.deeptools_backend.compile``
                                (fallback: ``torch_spyre.compiler.deeptools.run``)
    8. First device execution
                             -> the first call of the returned compiled function.

Every hook point is *best-effort*: at the pinned SHA
``fea0c4be901e1383b1f700dbad8887128b0fcb27`` some of the fallbacks are
what actually exist; on newer tips the primary names may replace them.
When a hook cannot be attached we log a WARNING to stderr and continue —
the profile will simply have that phase merged into an adjacent one.

Outputs (both in ``$TORCH_SPYRE_PROFILE_DIR`` — default cwd):

- ``profile.prof``          — cProfile stats, readable with ``pstats`` / ``analyze_profile.py``.
- ``phase_timings.json``    — {phase: [{start_us, end_us, elapsed_us, meta}, ...], "hooks": {...}}

Usage
-----
::

    export TORCH_SPYRE_PROFILE_DIR=/tmp/whole_compile_profile
    mkdir -p "$TORCH_SPYRE_PROFILE_DIR"
    python profile_whole_compile.py \\
        --harness measurements/2026-08-20/scripts/run_test_flash.py

The ``--harness`` argument is a Python file whose top-level module the
profiler will execute via ``runpy.run_path``. Everything the harness
does — importing torch_spyre, building the closure, moving tensors to
device, and calling ``torch.compile`` on the workload — runs inside
the outer cProfile.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import runpy
import sys
import time
from typing import Any, Callable


PROFILE_DIR = os.environ.get("TORCH_SPYRE_PROFILE_DIR", os.getcwd())
os.makedirs(PROFILE_DIR, exist_ok=True)

PROFILE_PATH = os.path.join(PROFILE_DIR, "profile.prof")
PHASES_PATH = os.path.join(PROFILE_DIR, "phase_timings.json")


# Phase name -> list of (module_dotted, attr_or_qualname) candidates.
# The first candidate that resolves wins; the rest are recorded as
# "fallbacks_skipped" in phase_timings.json so a reader can see what
# the code looked for.
HOOKS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "dynamo",
        [
            ("torch._dynamo.convert_frame", "convert_frame"),
        ],
    ),
    (
        "aot_autograd",
        [
            ("torch._functorch.aot_autograd", "aot_module_simplified"),
            ("torch._functorch.aot_autograd", "aot_export_module"),
        ],
    ),
    (
        "inductor_graph_lowering",
        [
            ("torch._inductor.graph", "GraphLowering.compile_to_module"),
            ("torch._inductor.graph", "GraphLowering.run"),
        ],
    ),
    (
        "torch_spyre_pre_scheduling",
        [
            ("torch_spyre._inductor.pass_wrapper", "run_passes"),
            ("torch_spyre._inductor.spyre_scheduling", "pre_scheduling"),
            ("torch_spyre._inductor.passes", "run_pre_scheduling_passes"),
        ],
    ),
    (
        "scheduler",
        [
            ("torch._inductor.scheduler", "Scheduler.__init__"),
        ],
    ),
    (
        "spyre_codegen",
        [
            ("torch_spyre._inductor.codegen.wrapper", "SpyreWrapperCodegen.generate"),
            ("torch_spyre._inductor.codegen", "SpyreScheduling.codegen"),
            ("torch_spyre._inductor.codegen.spyre", "SpyreCodegen.generate"),
        ],
    ),
    (
        "deeptools_backend",
        [
            ("torch_spyre._inductor.deeptools_backend", "compile"),
            ("torch_spyre.compiler.deeptools", "run"),
            ("torch_spyre._inductor.backend", "run_deeptools"),
        ],
    ),
    # first_device_exec is not hooked via monkeypatch; the harness
    # driver marks it explicitly around the compiled-fn first call
    # (see mark_first_device_exec_{begin,end} below). If the harness
    # does not mark it, the phase will simply be absent.
]


_PROCESS_T0 = time.perf_counter()


def _now_us() -> float:
    return (time.perf_counter() - _PROCESS_T0) * 1e6


# phase_timings holds one list per phase; each entry is a call record.
phase_timings: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in HOOKS}
phase_timings["first_device_exec"] = []
hook_status: dict[str, dict[str, Any]] = {}


def _resolve_target(module_path: str, qualname: str) -> tuple[Any, str, Callable] | None:
    """Resolve ``module_path.qualname`` to (owner, attr, orig_callable).

    Supports one level of class qualification (``Cls.method``).
    Returns None if any component is missing.
    """
    try:
        module = __import__(module_path, fromlist=["__name__"])
    except Exception as e:  # noqa: BLE001
        return None
    parts = qualname.split(".")
    owner: Any = module
    for part in parts[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    attr = parts[-1]
    orig = getattr(owner, attr, None)
    if orig is None or not callable(orig):
        return None
    return owner, attr, orig


def _make_wrapper(phase: str, module_path: str, qualname: str, orig: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = _now_us()
        try:
            return orig(*args, **kwargs)
        finally:
            end = _now_us()
            phase_timings[phase].append(
                {
                    "start_us": start,
                    "end_us": end,
                    "elapsed_us": end - start,
                    "meta": {
                        "module": module_path,
                        "qualname": qualname,
                    },
                }
            )

    wrapper.__wrapped__ = orig  # so pstats keeps the real qualname readable
    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapper


def install_hooks() -> None:
    """Install one wrapper per phase; record which candidate won.

    Called BEFORE the harness runs so its ``torch.compile`` call hits
    the wrapped entry points.
    """
    for phase, candidates in HOOKS:
        attached = None
        skipped: list[str] = []
        for module_path, qualname in candidates:
            resolved = _resolve_target(module_path, qualname)
            if resolved is None:
                skipped.append(f"{module_path}.{qualname}")
                continue
            owner, attr, orig = resolved
            try:
                setattr(owner, attr, _make_wrapper(phase, module_path, qualname, orig))
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{module_path}.{qualname} (setattr failed: {e})")
                continue
            attached = f"{module_path}.{qualname}"
            break
        if attached is None:
            sys.stderr.write(
                f"[profile_whole_compile] WARNING: no hook point resolved for phase "
                f"{phase!r}; tried {skipped}. Phase will be absent from phase_timings.\n"
            )
            hook_status[phase] = {"attached": None, "skipped": skipped}
        else:
            hook_status[phase] = {"attached": attached, "skipped": skipped}


def mark_first_device_exec_begin() -> None:
    phase_timings["first_device_exec"].append(
        {"start_us": _now_us(), "end_us": None, "elapsed_us": None, "meta": {}}
    )


def mark_first_device_exec_end() -> None:
    if not phase_timings["first_device_exec"]:
        return
    rec = phase_timings["first_device_exec"][-1]
    if rec["end_us"] is not None:
        return
    rec["end_us"] = _now_us()
    rec["elapsed_us"] = rec["end_us"] - rec["start_us"]


def write_timings() -> None:
    payload = {
        "hooks": hook_status,
        "process_start_perf_counter": _PROCESS_T0,
        "phases": phase_timings,
    }
    with open(PHASES_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def run_harness(harness_path: str, harness_argv: list[str]) -> int:
    """Run ``harness_path`` via runpy under cProfile.

    ``harness_argv`` becomes ``sys.argv`` while the harness runs; sys.argv[0]
    is forced to the harness path so ``if __name__ == '__main__'`` blocks
    behave normally.
    """
    saved_argv = sys.argv
    sys.argv = [harness_path, *harness_argv]

    # Expose the two markers into the harness's globals via a stub
    # module. Harnesses that need to mark first-device-exec can do:
    #   from torch_spyre_profile_marks import (
    #       mark_first_device_exec_begin, mark_first_device_exec_end,
    #   )
    import types
    marks_mod = types.ModuleType("torch_spyre_profile_marks")
    marks_mod.mark_first_device_exec_begin = mark_first_device_exec_begin
    marks_mod.mark_first_device_exec_end = mark_first_device_exec_end
    sys.modules["torch_spyre_profile_marks"] = marks_mod

    profiler = cProfile.Profile()
    profiler.enable()
    exit_code = 0
    try:
        runpy.run_path(harness_path, run_name="__main__")
    except SystemExit as e:  # harness may call sys.exit(...)
        exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        profiler.disable()
        sys.argv = saved_argv
        profiler.dump_stats(PROFILE_PATH)
        write_timings()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wrap a torch-spyre compile harness with cProfile + phase timers."
    )
    parser.add_argument(
        "--harness",
        required=True,
        help="Path to a Python harness file (executed via runpy.run_path).",
    )
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="Args forwarded to the harness as sys.argv[1:].",
    )
    args = parser.parse_args()

    install_hooks()
    exit_code = run_harness(args.harness, args.harness_args or [])
    sys.stderr.write(
        f"[profile_whole_compile] wrote {PROFILE_PATH} and {PHASES_PATH}\n"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
