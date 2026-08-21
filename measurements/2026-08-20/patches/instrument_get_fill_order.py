"""Focused instrumentation for ``ComputedBuffer.get_fill_order``.

Target
------
``torch._inductor.ir.ComputedBuffer.get_fill_order``

At pinned torch-spyre ``fea0c4be901e1383b1f700dbad8887128b0fcb27``,
``get_fill_order`` is an upstream Inductor method that in turn calls
``self.get_read_writes()`` (see the ``caller_func = "get_fill_order"``
rows in ``measurements/2026-08-20/data/test_flash.jsonl``). It runs
inside torch-spyre's lowering / scheduling flow, so wrapping it here
gives an unbiased count of how often the fill-order recomputation
kicks in per op.

What we record
--------------
One JSONL row per call:

- ``kind = "get_fill_order"``
- ``elapsed_us``            wall-clock time in the wrapped body
- ``op_pyid``                hex ``id(self)``
- ``operation_name``          ``self.operation_name``
- ``op_buffer_name``          ``self.name``
- ``op_type``                ``type(self).__name__``
- ``arity``                  positional arg count (self excluded)
- ``kwarg_names``            sorted list of kwarg keys
- ``caller_file``, ``caller_line``, ``caller_func``
- ``seq``                    monotonic per-process counter
- ``depth``                  wrapper nesting depth at emission

Enable by importing this module BEFORE any torch-spyre compilation
begins.
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
    "TORCH_SPYRE_INSTRUMENT_LOG",
    "/tmp/torch_spyre_get_fill_order.jsonl",
)

with open(LOG_PATH, "w") as _f:
    _f.write("")
_log_fh = open(LOG_PATH, "a", buffering=1)
_seq_counter = itertools.count()


class _State(threading.local):
    def __init__(self) -> None:
        self.depth = 0


_state = _State()


def _emit(record: dict[str, Any]) -> None:
    record.setdefault("seq", next(_seq_counter))
    _log_fh.write(json.dumps(record, default=str))
    _log_fh.write("\n")


def _caller_frame(depth: int) -> tuple[str, int, str]:
    frame = sys._getframe(depth)
    return frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name


def _op_identity(op: Any) -> dict[str, Any]:
    operation_name = getattr(op, "operation_name", None)
    buffer_name = getattr(op, "name", None)
    canonical = operation_name or buffer_name or "<no-name>"
    return {
        "op_pyid": hex(id(op)),
        "operation_name": str(canonical),
        "op_buffer_name": str(buffer_name) if buffer_name is not None else None,
        "op_type": type(op).__name__,
    }


def install() -> None:
    try:
        from torch._inductor.ir import ComputedBuffer
    except Exception as e:  # noqa: BLE001
        _emit(
            {
                "kind": "instrument_skipped",
                "target": "torch._inductor.ir.ComputedBuffer.get_fill_order",
                "reason": f"import failed: {e}",
            }
        )
        return

    orig = getattr(ComputedBuffer, "get_fill_order", None)
    if orig is None:
        _emit(
            {
                "kind": "instrument_skipped",
                "target": "torch._inductor.ir.ComputedBuffer.get_fill_order",
                "reason": "method missing at this torch build",
            }
        )
        return

    def wrapped(self, *args, **kwargs):
        try:
            caller_file, caller_line, caller_func = _caller_frame(2)
        except ValueError:
            caller_file, caller_line, caller_func = ("<unknown>", 0, "<unknown>")
        ident = _op_identity(self)
        depth_at_entry = _state.depth
        _state.depth += 1
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            _state.depth -= 1
            _emit(
                {
                    "kind": "get_fill_order",
                    "caller_file": caller_file,
                    "caller_line": caller_line,
                    "caller_func": caller_func,
                    "elapsed_us": elapsed_us,
                    "arity": len(args),
                    "kwarg_names": sorted(kwargs.keys()),
                    "depth": depth_at_entry,
                    **ident,
                }
            )

    ComputedBuffer.get_fill_order = wrapped
    _emit(
        {
            "kind": "instrument_installed",
            "target": "torch._inductor.ir.ComputedBuffer.get_fill_order",
            "log_path": LOG_PATH,
            "pid": os.getpid(),
            "argv": sys.argv,
            "schema_version": 2,
        }
    )


install()
