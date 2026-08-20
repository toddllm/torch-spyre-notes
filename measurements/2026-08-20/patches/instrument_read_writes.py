"""Monkey-patch instrumentation for get_read_writes / op_read_writes.

Purpose
-------
Log every call to `ComputedBuffer.get_read_writes` and to
`torch_spyre._inductor.pass_utils.op_read_writes` during a compile, with:

- caller file:line (the direct call site)
- enclosing torch-spyre function (one frame up if inside pass_utils.op_read_writes)
- op type + short name (best-effort)
- cache hit / miss for op_read_writes
- wall-clock ms for the underlying dependency extraction

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

import json
import os
import sys
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


def _emit(record: dict[str, Any]) -> None:
    _log_fh.write(json.dumps(record, default=str))
    _log_fh.write("\n")


def _caller_frame(depth: int) -> tuple[str, int, str]:
    """Return (file, line, function) for the frame `depth` levels above the caller."""
    frame = sys._getframe(depth)
    return frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name


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
        op_name = getattr(self, "operation_name", None) or getattr(
            self, "name", "<no-name>"
        )
        t0 = time.perf_counter()
        result = orig(self, *args, **kwargs)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        _emit(
            {
                "kind": "get_read_writes_raw",
                "caller_file": caller_file,
                "caller_line": caller_line,
                "caller_func": caller_func,
                "op_type": type(self).__name__,
                "op_name": str(op_name),
                "elapsed_us": elapsed_us,
                "cache": "n/a",
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
        op_name = getattr(op, "operation_name", None) or getattr(
            op, "name", "<no-name>"
        )
        # Cache-hit check: op_read_writes stores under `_ts_cached_read_writes`
        # in the op's __dict__ (verified at torch-spyre fea0c4be pass_utils.py:114-116).
        cached = op.__dict__.get("_ts_cached_read_writes")
        cache_status = "hit" if cached is not None else "miss"
        t0 = time.perf_counter()
        result = orig(op)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        _emit(
            {
                "kind": "op_read_writes_memo",
                "caller_file": caller_file,
                "caller_line": caller_line,
                "caller_func": caller_func,
                "op_type": type(op).__name__,
                "op_name": str(op_name),
                "elapsed_us": elapsed_us,
                "cache": cache_status,
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
        }
    )


install()
