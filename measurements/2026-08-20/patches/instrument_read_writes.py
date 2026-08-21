"""Monkey-patch instrumentation for get_read_writes / op_read_writes.

Purpose
-------
Log every call to `ComputedBuffer.get_read_writes` and to
`torch_spyre._inductor.pass_utils.op_read_writes` during a compile, with:

- caller file:line (the direct call site)
- enclosing torch-spyre function (one frame up if inside pass_utils.op_read_writes)
- op type + short name (best-effort)
- cache hit / miss for op_read_writes
- wall-clock us for the underlying dependency extraction

Exclusive-vs-inclusive timing
-----------------------------
On an op_read_writes cache MISS, the wrapped op_read_writes body calls
ComputedBuffer.get_read_writes, which we also wrap. If both wrappers
just record their own elapsed time, the inner get_read_writes work is
counted twice when summed. We fix this by tagging each record with a
`kind` and by tracking a per-thread nesting counter so the analyzer
can compute exclusive time correctly:

- ``raw_get_read_writes`` — the inner ComputedBuffer.get_read_writes wrapper.
  ``elapsed_us`` is the wall time spent in ``orig(self, ...)``.
- ``memo_wrapper_hit`` — op_read_writes call that returned the cached
  value without invoking get_read_writes. Exclusive == inclusive.
- ``memo_wrapper_miss_inclusive`` — op_read_writes call that missed the
  cache; ``elapsed_us`` is the full wall time (including the nested
  raw_get_read_writes it triggered). The paired
  ``memo_wrapper_overhead`` record carries the exclusive portion
  (miss_inclusive - inner raw for the call).
- ``memo_wrapper_overhead`` — synthetic record emitted alongside every
  miss_inclusive; ``elapsed_us`` is (miss_inclusive - captured inner raw).
  Summing raw_get_read_writes + memo_wrapper_hit + memo_wrapper_overhead
  gives the true exclusive total. Summing miss_inclusive on top of that
  would double-count.

Additional identity fields on every record:

- ``op_pyid`` — hex ``id(op)``; distinguishes distinct Python objects
  that happen to reuse the same ``operation_name``.
- ``operation_name`` — canonical op name (or ``<no-name>``).
- ``op_buffer_name`` — ``op.name`` when present (buffers), else null.
- ``op_type`` — ``type(op).__name__``.
- ``seq`` — monotonically-incrementing per-process call sequence number.
- ``depth`` — nesting depth of the wrapper at emission time (0 = outer).

Records land as JSONL at $TORCH_SPYRE_INSTRUMENT_LOG (default:
/tmp/torch_spyre_read_writes.jsonl). One line per call.

Enable by importing this module BEFORE any torch-spyre compilation
begins. Example:

    import measurements.2026-08-20.patches.instrument_read_writes  # noqa
    # ... normal torch.compile flow ...

Or, easier from a shell script:

    PYTHONPATH=/path/to/torch-spyre-notes/measurements/2026-08-20/patches \\
        python -c 'import instrument_read_writes; import runpy; runpy.run_path("path/to/harness.py")'

No modification of torch-spyre source is performed. Patches are
installed on the class object and on the module-level function; the
process only needs to survive one compile. Import order: torch first
(to fully populate torch._inductor.ir), then torch_spyre (so
pass_utils is loaded and op_read_writes exists), then this module.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
from typing import Any


LOG_PATH = os.environ.get(
    "TORCH_SPYRE_INSTRUMENT_LOG", "/tmp/torch_spyre_read_writes.jsonl"
)

# Truncate on install so each run starts fresh. If two runs need to
# coexist, set TORCH_SPYRE_INSTRUMENT_LOG to different paths.
with open(LOG_PATH, "w") as _f:
    _f.write("")

_log_fh = open(LOG_PATH, "a", buffering=1)  # line-buffered

# Monotonic per-process call-sequence counter. Emitted on every record
# so downstream analysis can reconstruct call order even if records
# are reordered on disk (they shouldn't be, but line-buffered writes
# from multiple threads could interleave).
_seq_counter = itertools.count()


class _NestState(threading.local):
    """Per-thread wrapper nesting state.

    ``depth`` is the current wrapper-call depth (0 outside any wrapper).
    ``inner_raw_us`` is a stack; when a memo wrapper enters, it pushes
    ``0.0``. Every raw_get_read_writes that finishes while a memo
    wrapper is active adds its elapsed_us to the top-of-stack. On
    memo-wrapper exit we pop the accumulated inner time and record
    the exclusive overhead.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.inner_raw_us: list[float] = []


_state = _NestState()


def _emit(record: dict[str, Any]) -> None:
    record.setdefault("seq", next(_seq_counter))
    _log_fh.write(json.dumps(record, default=str))
    _log_fh.write("\n")


def _caller_frame(depth: int) -> tuple[str, int, str]:
    """Return (file, line, function) for the frame `depth` levels above the caller."""
    frame = sys._getframe(depth)
    return frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name


def _op_identity(op: Any) -> dict[str, Any]:
    """Extract identity fields for an op/buffer."""
    operation_name = getattr(op, "operation_name", None)
    buffer_name = getattr(op, "name", None)
    # Prefer operation_name (canonical for Operations); fall back to buffer name.
    canonical = operation_name or buffer_name or "<no-name>"
    return {
        "op_pyid": hex(id(op)),
        "operation_name": str(canonical),
        "op_buffer_name": str(buffer_name) if buffer_name is not None else None,
        "op_type": type(op).__name__,
    }


def _install_get_read_writes() -> None:
    from torch._inductor.ir import ComputedBuffer

    orig = ComputedBuffer.get_read_writes

    def wrapped(self, *args, **kwargs):
        # _caller_frame(2) skips both _caller_frame's frame AND this wrapper's frame,
        # landing on the actual caller of get_read_writes.
        try:
            caller_file, caller_line, caller_func = _caller_frame(2)
        except ValueError:
            caller_file, caller_line, caller_func = ("<unknown>", 0, "<unknown>")
        ident = _op_identity(self)
        depth_at_entry = _state.depth
        _state.depth += 1
        t0 = time.perf_counter()
        try:
            result = orig(self, *args, **kwargs)
        finally:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            _state.depth -= 1
            # If a memo wrapper is on the stack, credit this raw call to
            # its inner-raw accumulator so we can compute exclusive time.
            if _state.inner_raw_us:
                _state.inner_raw_us[-1] += elapsed_us
        _emit(
            {
                "kind": "raw_get_read_writes",
                "caller_file": caller_file,
                "caller_line": caller_line,
                "caller_func": caller_func,
                "elapsed_us": elapsed_us,
                "cache": "n/a",
                "depth": depth_at_entry,
                **ident,
            }
        )
        return result

    ComputedBuffer.get_read_writes = wrapped


def _install_op_read_writes() -> None:
    # Import the module by string so this file is safe to import even
    # if torch_spyre is not yet installed at that path.
    import torch_spyre._inductor.pass_utils as pu

    orig = pu.op_read_writes

    def wrapped(op):
        # _caller_frame(2) skips both _caller_frame's frame AND this wrapper's frame.
        try:
            caller_file, caller_line, caller_func = _caller_frame(2)
        except ValueError:
            caller_file, caller_line, caller_func = ("<unknown>", 0, "<unknown>")
        ident = _op_identity(op)
        # Cache-hit check: op_read_writes stores under `_ts_cached_read_writes`
        # in the op's __dict__ (verified at torch-spyre fea0c4be pass_utils.py:114-116).
        cached = op.__dict__.get("_ts_cached_read_writes")
        cache_status = "hit" if cached is not None else "miss"

        depth_at_entry = _state.depth
        _state.depth += 1
        # Push an inner-raw accumulator so any raw_get_read_writes that
        # fires below can be attributed to this call.
        _state.inner_raw_us.append(0.0)
        t0 = time.perf_counter()
        try:
            result = orig(op)
        finally:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            inner_raw = _state.inner_raw_us.pop()
            _state.depth -= 1
            # If we ourselves are nested inside another memo wrapper,
            # our inclusive time is inner-raw to that parent. (Highly
            # unusual for op_read_writes, but keep the bookkeeping honest.)
            if _state.inner_raw_us:
                _state.inner_raw_us[-1] += elapsed_us

        base = {
            "caller_file": caller_file,
            "caller_line": caller_line,
            "caller_func": caller_func,
            "cache": cache_status,
            "depth": depth_at_entry,
            **ident,
        }
        if cache_status == "hit":
            _emit(
                {
                    "kind": "memo_wrapper_hit",
                    "elapsed_us": elapsed_us,
                    **base,
                }
            )
        else:
            _emit(
                {
                    "kind": "memo_wrapper_miss_inclusive",
                    "elapsed_us": elapsed_us,
                    "inner_raw_us": inner_raw,
                    **base,
                }
            )
            # Synthetic exclusive-overhead record. Summing
            # raw_get_read_writes + memo_wrapper_hit + memo_wrapper_overhead
            # gives the true exclusive total across all wrappers.
            overhead = elapsed_us - inner_raw
            _emit(
                {
                    "kind": "memo_wrapper_overhead",
                    "elapsed_us": overhead,
                    "inner_raw_us": inner_raw,
                    "inclusive_us": elapsed_us,
                    **base,
                }
            )
        return result

    pu.op_read_writes = wrapped


def install() -> None:
    _install_get_read_writes()
    _install_op_read_writes()
    _emit(
        {
            "kind": "instrument_installed",
            "log_path": LOG_PATH,
            "pid": os.getpid(),
            "argv": sys.argv,
            "schema_version": 2,
        }
    )


install()
