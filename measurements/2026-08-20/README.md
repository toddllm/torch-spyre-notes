# measurements/2026-08-20

Instrumentation data and analysis for the `test_flash` cold-compile
workload at torch-spyre@fea0c4b.

## Schema note — test_flash.jsonl is v1 (historical)

`data/test_flash.jsonl` was produced against an older revision of
`patches/instrument_read_writes.py` whose records lacked the identity
and bookkeeping fields (`op_pyid`, `operation_name`, `op_type`, `seq`,
`depth`) and the four `kind` values (`raw_get_read_writes`,
`memo_wrapper_hit`, `memo_wrapper_miss_inclusive`,
`memo_wrapper_overhead`) that `scripts/analyze.py` now requires for
exclusive-vs-inclusive breakdowns. When fed the v1 log,
`analyze.py` auto-detects the legacy kinds and falls back to
inclusive-only totals — the exclusive tables the finding docs quote are
not available from this file. Treat it as historical. Regenerate with
schema v2 via `needs-pod/06-regenerate-test-flash-v2.sh` on a dev pod
that has torch_spyre installed; the new log lands at
`data/test_flash_v2.jsonl` and downstream analysis switches to v2 mode
automatically.

## Layout

- `patches/` — monkey-patch instrumentation stubs loaded before
  `torch.compile`. `instrument_read_writes.py` declares
  `schema_version=2` on its `instrument_installed` header record.
- `scripts/` — harness (`run_test_flash.py`), whole-compile profiler
  (`profile_whole_compile.py`), and analyzers (`analyze.py`,
  `analyze_profile.py`).
- `data/` — captured JSONL logs and analyzer text output.
