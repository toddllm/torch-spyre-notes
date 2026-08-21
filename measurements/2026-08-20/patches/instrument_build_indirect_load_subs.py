"""Focused instrumentation for ``_build_indirect_load_subs``.

Target
------
``torch_spyre._inductor.pass_utils._build_indirect_load_subs``

Signature at pinned SHA ``fea0c4be901e1383b1f700dbad8887128b0fcb27``::

    def _build_indirect_load_subs(op: ComputedBuffer) -> dict[sympy.Symbol, sympy.Expr]:

Called (per ``scans/results/repeated_analysis.json``) from
``indirect_info_from_op`` in the same module. Repeated-analysis scan
count: 1 site, but the function itself is a hot inner loop.

What we record
--------------
One JSONL row per call with:

- ``kind = "build_indirect_load_subs"``
- ``elapsed_us``      wall-clock time in the wrapped body
- ``op_pyid``          hex ``id(op)`` — distinct-Python-object identity
- ``operation_name``   ``op.operation_name`` (or ``<no-name>``)
- ``op_buffer_name``   ``op.name`` when present
- ``op_type``          ``type(op).__name__``
- ``arity``            positional arg count (always 1 for this signature,
                       included for a stable schema across the three stubs)
- ``kwarg_names``      sorted list of kwarg keys (for the same reason)
- ``caller_file``, ``caller_line``, ``caller_func``
                       one frame up (the direct call site)
- ``seq``              monotonic per-process counter
- ``depth``            wrapper nesting depth at emission (0 = outer)

Enable by importing this module BEFORE any torch-spyre compilation
begins. See instrument_read_writes.py for background on the pattern
(same log-path env var, same schema conventions).

If the target module or function is missing at this build, the module
still imports cleanly and logs one ``kind = "instrument_skipped"``
record with the resolve error; the compile can proceed unmodified.
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
    "/tmp/torch_spyre_build_indirect_load_subs.jsonl",
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
        import torch_spyre._inductor.pass_utils as pu
    except Exception as e:  # noqa: BLE001
        _emit(
            {
                "kind": "instrument_skipped",
                "target": "torch_spyre._inductor.pass_utils._build_indirect_load_subs",
                "reason": f"import failed: {e}",
            }
        )
        return

    orig = getattr(pu, "_build_indirect_load_subs", None)
    if orig is None:
        _emit(
            {
                "kind": "instrument_skipped",
                "target": "torch_spyre._inductor.pass_utils._build_indirect_load_subs",
                "reason": "attribute missing at this build",
            }
        )
        return

    def wrapped(op, *args, **kwargs):
        try:
            caller_file, caller_line, caller_func = _caller_frame(2)
        except ValueError:
            caller_file, caller_line, caller_func = ("<unknown>", 0, "<unknown>")
        ident = _op_identity(op)
        depth_at_entry = _state.depth
        _state.depth += 1
        t0 = time.perf_counter()
        try:
            return orig(op, *args, **kwargs)
        finally:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            _state.depth -= 1
            _emit(
                {
                    "kind": "build_indirect_load_subs",
                    "caller_file": caller_file,
                    "caller_line": caller_line,
                    "caller_func": caller_func,
                    "elapsed_us": elapsed_us,
                    "arity": 1 + len(args),
                    "kwarg_names": sorted(kwargs.keys()),
                    "depth": depth_at_entry,
                    **ident,
                }
            )

    pu._build_indirect_load_subs = wrapped
    _emit(
        {
            "kind": "instrument_installed",
            "target": "torch_spyre._inductor.pass_utils._build_indirect_load_subs",
            "log_path": LOG_PATH,
            "pid": os.getpid(),
            "argv": sys.argv,
            "schema_version": 2,
        }
    )


install()
