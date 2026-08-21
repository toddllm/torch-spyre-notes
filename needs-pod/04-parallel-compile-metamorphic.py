#!/usr/bin/env python3
"""
Parallel-compile metamorphic test for the ``enable_spyre_lowerings`` RLock.

Companion to findings/upstream-fragility/03-lowering-registry-lock.md.

The lock at torch_spyre/_inductor/lowering.py:60 (module-level RLock,
`_lowerings_lock`) is held across the entire yielded region of
``enable_spyre_lowerings`` (see :183 acquire, :242 yield, :279 release
in the pinned tree). torch_spyre calls the CM from
_inductor/patches.py:163 inside ``enable_spyre_context``, so the lock
is held across the whole Inductor compile that runs inside the caller's
``with`` block. This test measures the wall-clock cost.

Three scenarios, submitted as three JSONL records per run:

  (a) two threads, each doing torch.compile of the SAME Spyre model
      (both hit ``enable_spyre_lowerings`` → contend on _lowerings_lock).

  (b) two threads: one Spyre compile, one CPU-only compile
      (only the Spyre thread acquires _lowerings_lock; but the
      CPU thread reads a lowering.lowerings that the Spyre thread
      has mutated — the *contamination* metamorphic).

  (c) nested Spyre compile inside Spyre-compile lowering
      (a lowering triggers another torch.compile of a Spyre-device
      graph on the same thread — re-enters _lowerings_lock via the
      RLock's reentrancy; nesting counter should increment).

Instrumentation:
  - We monkey-patch ``_lowerings_lock.acquire``/``release`` to record
    (t_acquire, t_release) per thread; hold times are computed.
  - We monkey-patch ``enable_spyre_lowerings`` to log (t_enter,
    t_exit, nesting_before, nesting_after) per invocation.

Outputs one line of JSON per (scenario, thread, event) to
``results/parallel-compile-metamorphic.jsonl`` next to this file
(or to $TORCH_SPYRE_NOTES_RESULTS if set).

Dry mode:
  --dry runs the harness itself (thread scheduling, JSONL writer,
  instrumentation plumbing) without importing torch_spyre. It fabricates
  a fake ``_lowerings_lock`` and a fake ``enable_spyre_lowerings`` CM so
  the coordinator can validate the framework on a laptop. Dry mode
  writes to a ``.jsonl`` file that carries ``"scenario":"dry-…"``.

Usage (on pod):
    python3 needs-pod/04-parallel-compile-metamorphic.py

Usage (laptop, dry):
    python3 needs-pod/04-parallel-compile-metamorphic.py --dry
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── file locations ───────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = HERE / "results" / "parallel-compile-metamorphic.jsonl"


# ─── JSONL writer (thread-safe) ───────────────────────────────────────

class JsonlWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Truncate at open so re-running produces a clean file per invocation.
        self._fh = open(self.path, "w", buffering=1)  # line-buffered

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


# ─── instrumentation state ────────────────────────────────────────────

@dataclass
class HoldEvent:
    thread_id: int
    thread_name: str
    t_acquire_ns: int
    t_release_ns: int
    hold_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "lock-hold",
            "thread_id": self.thread_id,
            "thread_name": self.thread_name,
            "t_acquire_ns": self.t_acquire_ns,
            "t_release_ns": self.t_release_ns,
            "hold_ns": self.hold_ns,
        }


@dataclass
class CMEvent:
    thread_id: int
    thread_name: str
    t_enter_ns: int
    t_exit_ns: int
    nesting_before: int
    nesting_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "cm-window",
            "thread_id": self.thread_id,
            "thread_name": self.thread_name,
            "t_enter_ns": self.t_enter_ns,
            "t_exit_ns": self.t_exit_ns,
            "hold_ns": self.t_exit_ns - self.t_enter_ns,
            "nesting_before": self.nesting_before,
            "nesting_after": self.nesting_after,
        }


class _WrappingLock:
    """Proxy around a threading.RLock that records acquire/release timings.

    ``_thread.RLock`` disallows direct attribute assignment on its
    ``acquire``/``release`` methods (they are read-only C descriptors), so
    we replace the module-level lock *object* with this proxy. The proxy
    supports the same ``with``-statement interface and forwards through to
    the underlying lock.
    """

    def __init__(self, inner: Any, writer: JsonlWriter, scenario: str) -> None:
        self._inner = inner
        self._writer = writer
        self._scenario = scenario
        self._per_thread_acquires: dict[int, int] = {}

    def acquire(self, *args, **kwargs):
        r = self._inner.acquire(*args, **kwargs)
        self._per_thread_acquires[threading.get_ident()] = time.perf_counter_ns()
        return r

    def release(self, *args, **kwargs):
        tid = threading.get_ident()
        t_release = time.perf_counter_ns()
        t_acq = self._per_thread_acquires.pop(tid, None)
        if t_acq is not None:
            ev = HoldEvent(
                thread_id=tid,
                thread_name=threading.current_thread().name,
                t_acquire_ns=t_acq,
                t_release_ns=t_release,
                hold_ns=t_release - t_acq,
            )
            self._writer.write({"scenario": self._scenario, **ev.to_dict()})
        return self._inner.release(*args, **kwargs)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class Instrumentation:
    """Wraps _lowerings_lock and enable_spyre_lowerings to log hold windows."""

    def __init__(self, writer: JsonlWriter, scenario: str) -> None:
        self.writer = writer
        self.scenario = scenario

    def wrap_lock_on(self, holder: Any, attr_name: str) -> None:
        """Replace ``holder.attr_name`` (an RLock) with a WrappingLock proxy.

        In dry mode ``holder`` is the fake lowering module; live mode it is
        ``torch_spyre._inductor.lowering``. Idempotent — replacing twice
        chains proxies but that only doubles timing overhead, not correctness.
        """
        current = getattr(holder, attr_name)
        proxy = _WrappingLock(current, self.writer, self.scenario)
        setattr(holder, attr_name, proxy)

    def wrap_cm(self, module: Any) -> None:
        """Wrap module.enable_spyre_lowerings with instrumentation."""
        original_cm = module.enable_spyre_lowerings

        @contextlib.contextmanager
        def _instrumented():
            tid = threading.get_ident()
            nesting_before = getattr(module, "_lowerings_nesting", -1)
            t_enter = time.perf_counter_ns()
            with original_cm():
                nesting_inside = getattr(module, "_lowerings_nesting", -1)
                yield
            t_exit = time.perf_counter_ns()
            nesting_after = getattr(module, "_lowerings_nesting", -1)
            ev = CMEvent(
                thread_id=tid,
                thread_name=threading.current_thread().name,
                t_enter_ns=t_enter,
                t_exit_ns=t_exit,
                nesting_before=nesting_before,
                nesting_after=nesting_after,
            )
            rec = {
                "scenario": self.scenario,
                **ev.to_dict(),
                "nesting_inside": nesting_inside,
            }
            self.writer.write(rec)

        module.enable_spyre_lowerings = _instrumented


# ─── the three scenarios ──────────────────────────────────────────────

def _tiny_model_and_inputs(device: str):
    """A minimal MLP compile fixture. Uses the given device."""
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(16, 32, bias=False)
            self.b = torch.nn.Linear(32, 16, bias=False)

        def forward(self, x):
            return self.b(torch.nn.functional.relu(self.a(x)))

    model = Tiny().to(device)
    example = torch.randn(4, 16, device=device)
    return model, example


def _compile_and_run_once(device: str, writer: JsonlWriter, scenario: str, tag: str) -> None:
    import torch

    t_start = time.perf_counter_ns()
    exc_info = None
    try:
        model, example = _tiny_model_and_inputs(device)
        compiled = torch.compile(model, fullgraph=False, dynamic=False)
        _ = compiled(example)
    except Exception:  # pragma: no cover
        exc_info = traceback.format_exc()
    t_end = time.perf_counter_ns()
    writer.write({
        "scenario": scenario,
        "kind": "compile-run",
        "tag": tag,
        "device": device,
        "thread_id": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "t_start_ns": t_start,
        "t_end_ns": t_end,
        "wall_ns": t_end - t_start,
        "exception": exc_info,
    })


def scenario_a_two_spyre(writer: JsonlWriter) -> None:
    """Two threads, both compile a Spyre model."""
    scenario = "a-two-spyre-threads"
    threads = [
        threading.Thread(
            target=_compile_and_run_once,
            args=("spyre", writer, scenario, f"thread-{i}"),
            name=f"spyre-{i}",
        )
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def scenario_b_spyre_and_cpu(writer: JsonlWriter) -> None:
    """One Spyre compile, one CPU compile, in parallel."""
    scenario = "b-spyre-plus-cpu"
    t_spyre = threading.Thread(
        target=_compile_and_run_once,
        args=("spyre", writer, scenario, "spyre"),
        name="spyre",
    )
    t_cpu = threading.Thread(
        target=_compile_and_run_once,
        args=("cpu", writer, scenario, "cpu"),
        name="cpu",
    )
    t_spyre.start()
    t_cpu.start()
    t_spyre.join()
    t_cpu.join()


def scenario_c_nested(writer: JsonlWriter) -> None:
    """
    Nested Spyre compile inside a Spyre compile lowering.

    We create a torch.compile'd function whose graph contains a call to
    another torch.compile'd function; if Inductor's tracing runs into
    the inner compile *during* the outer compile's lowering pass, the
    RLock should re-enter (RLock reentrancy) rather than deadlock.
    """
    import torch

    scenario = "c-nested-spyre-compile"

    # Inner: a Spyre-compiled callable that gets called from the outer's forward.
    @torch.compile(fullgraph=False)
    def inner(y):
        return torch.tanh(y)

    def outer_fn(x):
        return inner(x) * 2.0

    # If this raises, the traceback ends up on the record.
    t_start = time.perf_counter_ns()
    exc_info = None
    try:
        example = torch.randn(4, 16, device="spyre")
        compiled_outer = torch.compile(outer_fn, fullgraph=False)
        _ = compiled_outer(example)
    except Exception:
        exc_info = traceback.format_exc()
    t_end = time.perf_counter_ns()
    writer.write({
        "scenario": scenario,
        "kind": "compile-run",
        "tag": "outer",
        "thread_id": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "t_start_ns": t_start,
        "t_end_ns": t_end,
        "wall_ns": t_end - t_start,
        "exception": exc_info,
    })


# ─── dry-mode fake substrate ──────────────────────────────────────────

class _FakeLoweringModule:
    """A stand-in for torch_spyre._inductor.lowering used only in --dry mode."""

    def __init__(self) -> None:
        self._lowerings_lock = threading.RLock()
        self._lowerings_nesting = 0

        @contextlib.contextmanager
        def enable_spyre_lowerings():
            with self._lowerings_lock:
                self._lowerings_nesting += 1
                try:
                    # Simulate the cost of a compile inside the yield.
                    time.sleep(0.02)
                    yield
                finally:
                    self._lowerings_nesting -= 1

        self.enable_spyre_lowerings = enable_spyre_lowerings  # type: ignore[assignment]


def _run_dry(writer: JsonlWriter) -> None:
    """Exercise the harness without importing torch or torch_spyre."""
    fake = _FakeLoweringModule()
    instr = Instrumentation(writer, scenario="dry-harness-self-test")
    instr.wrap_lock_on(fake, "_lowerings_lock")
    instr.wrap_cm(fake)

    def _spin(tag: str) -> None:
        t_start = time.perf_counter_ns()
        with fake.enable_spyre_lowerings():
            time.sleep(0.01)
        t_end = time.perf_counter_ns()
        writer.write({
            "scenario": "dry-harness-self-test",
            "kind": "compile-run",
            "tag": tag,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "t_start_ns": t_start,
            "t_end_ns": t_end,
            "wall_ns": t_end - t_start,
            "exception": None,
        })

    # Two-thread contention exercise.
    threads = [
        threading.Thread(target=_spin, args=(f"dry-t{i}",), name=f"dry-t{i}")
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Nested-reentry exercise on one thread.
    t_start = time.perf_counter_ns()
    with fake.enable_spyre_lowerings():
        with fake.enable_spyre_lowerings():
            pass
    t_end = time.perf_counter_ns()
    writer.write({
        "scenario": "dry-harness-self-test",
        "kind": "compile-run",
        "tag": "dry-nested",
        "thread_id": threading.get_ident(),
        "thread_name": threading.current_thread().name,
        "t_start_ns": t_start,
        "t_end_ns": t_end,
        "wall_ns": t_end - t_start,
        "exception": None,
    })


# ─── entrypoint ───────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="run the harness itself without importing torch_spyre",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("TORCH_SPYRE_NOTES_RESULTS", DEFAULT_RESULTS_PATH)),
        help="output JSONL path",
    )
    args = parser.parse_args(argv)

    writer = JsonlWriter(args.results)
    writer.write({
        "kind": "run-header",
        "mode": "dry" if args.dry else "live",
        "python": sys.version.split()[0],
        "argv": argv or sys.argv,
        "cwd": os.getcwd(),
    })

    try:
        if args.dry:
            _run_dry(writer)
            print(f"[dry] wrote {args.results}", file=sys.stderr)
            return 0

        # Live path — pod-only from here down.
        try:
            import torch_spyre  # noqa: F401  # trigger _autoload_impl
            from torch_spyre._inductor import lowering as ts_lowering
        except Exception as e:
            writer.write({"kind": "fatal", "message": f"torch_spyre import failed: {e!r}"})
            print(f"torch_spyre import failed: {e!r}", file=sys.stderr)
            return 2

        # Wrap the RLock and the CM for each scenario. Because we run
        # scenarios sequentially we can rewrap between them to prefix
        # records with the scenario name.
        for scenario_fn, name in (
            (scenario_a_two_spyre, "a-two-spyre-threads"),
            (scenario_b_spyre_and_cpu, "b-spyre-plus-cpu"),
            (scenario_c_nested, "c-nested-spyre-compile"),
        ):
            instr = Instrumentation(writer, scenario=name)
            instr.wrap_lock_on(ts_lowering, "_lowerings_lock")
            instr.wrap_cm(ts_lowering)
            try:
                scenario_fn(writer)
            except Exception:
                writer.write({
                    "scenario": name,
                    "kind": "fatal",
                    "traceback": traceback.format_exc(),
                })

        print(f"[live] wrote {args.results}", file=sys.stderr)
        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
