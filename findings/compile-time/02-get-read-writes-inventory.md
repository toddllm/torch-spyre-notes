# `get_read_writes` / `op_read_writes` call-site inventory across torch-spyre

- **Category:** compile-time
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** proven (static inventory verified; per-call cost measured on the pod against a real cold compile)
- **Status:** open

## Summary

Torch-Spyre's `_inductor` package at `fea0c4be` calls PyTorch's
`Operation.get_read_writes()` (raw, uncached upstream) or the local
memoized wrapper `op_read_writes()` at **101 call sites across 21
files**. `extract_read_writes` (the underlying upstream helper that
`ComputedBuffer.get_read_writes` invokes to run SymPy dependency
extraction — see `torch/_inductor/dependencies.py`) is referenced only
in torch-spyre comments/docstrings; it is never called directly.

Breakdown:

| Dimension                | `get_read_writes` (raw upstream) | `op_read_writes` (memoized) | Total |
|--------------------------|----------------------------------|-----------------------------|-------|
| **Total call sites**     | 68                               | 33                          | 101   |
| Enclosing loop depth = 0 | 23                               | 14                          | 37    |
| Enclosing loop depth = 1 | 34                               | 12                          | 46    |
| Enclosing loop depth = 2 | 9                                | 6                           | 15    |
| Enclosing loop depth = 3 | 2                                | 1                           | 3     |
| Return value stored in a variable (whole `ReadWrites` reused) | 29 | 19 | 48 |
| `.reads` / `.writes` stored (partial reuse of the object) | 30 | 10 | 40 |
| Fully inline (`for … in op.get_read_writes().reads:` etc., discarded after single iteration) | 9 | 4 | 13 |

*Loop depth* counts `for`/`while` blocks plus generator/comprehension
generators (a list comprehension inside two `for` loops is depth 3).
The three depth-3 sites and 15 depth-2 sites are the hot candidates.

Files that use only the raw upstream call and never the memoized
wrapper: **13** — including four of the most frequent-call files
(`optimize_restickify.py`, `enforce_indirect_access_layout.py`,
`wsr/coarse_tile.py`, `propagate_layouts.py`). Files that use only the
memoized wrapper: **5**
(`work_division.py`, `work_division_constraints.py`,
`scratchpad/graph_editor.py`, `scratchpad/lx_relayout.py`,
`scratchpad/utils.py`). Two files mix both
(`pass_utils.py` — the file that *defines* `op_read_writes` still calls
the raw method three times in its own helpers; `scratchpad/allocator.py`
uses raw in five sites and memoized in eight).

## Files and symbols

`op_read_writes` is defined in
[`torch_spyre/_inductor/pass_utils.py`
L102-L116](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L102-L116).
Its docstring states:

> `ComputedBuffer.get_read_writes` re-runs sympy dependency extraction
> on every call and is not cached upstream, yet its result does not
> depend on the only thing the LX planner mutates
> (`op_it_space_splits`). The scratchpad pass calls it hundreds of
> times, so we cache it under a private key only this helper reads —
> **a non-planner caller (e.g. later-pass codegen) still goes through
> the real method.**

Implementation:

```python
def op_read_writes(op: Operation) -> ReadWrites:
    rw = op.__dict__.get("_ts_cached_read_writes")
    if rw is None:
        rw = op.get_read_writes()
        op.__dict__["_ts_cached_read_writes"] = rw
    return rw
```

Companion invalidator
[`invalidate_op_read_writes` L130-L140](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L130-L140)
drops the cached entry from `op.__dict__` and is intended to be called
"immediately after mutating an op's dependencies in place — e.g.
swapping a load name in its `inner_fn`."

**`op_read_writes` is NOT a full drop-in replacement.** Restrictions:

1. **Manual invalidation.** The cache is only correct as long as
   `op.get_read_writes()` would return the same `ReadWrites`. Any pass
   that mutates the op's `inner_fn`, its `data.inner_fn`, its ranges,
   or any input to SymPy dependency extraction must call
   `invalidate_op_read_writes(op)` before the next `op_read_writes(op)`
   call, or every later caller will read a stale `ReadWrites`.
2. **Op-instance identity.** The cache key is
   `op.__dict__["_ts_cached_read_writes"]`. Passes that clone or
   rebuild the op instance (e.g. `replace_computed_buffer_body`) get a
   fresh, uncached instance — that's a correctness win (no stale
   copy-across) but also a compile-time cost (every clone pays the
   full retrace).
3. **Documented scope.** The helper's docstring explicitly says
   "non-planner callers (e.g. later-pass codegen) still go through the
   real method." The intent is that only the LX planner uses the
   memo, because only its mutations (splits) are guaranteed cache-safe
   without invalidation.

Only one call site currently invalidates:
[`scratchpad/graph_editor.py:279`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/graph_editor.py#L279)
(`invalidate_op_read_writes(old_loop)`).

### Depth-3 call sites (hottest)

Each row is inside two enclosing Python for-loops plus a generator or
comprehension. `Kind` is `raw` for `op.get_read_writes()` and `memo`
for `op_read_writes()`.

| File & line | Enclosing function | Kind | Reuse | Loop-header + call |
|-------------|--------------------|------|-------|--------------------|
| [`enforce_indirect_access_layout.py:624`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/enforce_indirect_access_layout.py#L624) | `enforce_indirect_access_layout` | raw | `.reads` stored via `next(...)` | outer L595 `for original_op in list(graph.operations):`  · L616 `for value_buf in value_bufs:`  · L622-626 `value_dep = next(d for d in op.get_read_writes().reads if isinstance(d, MemoryDep) and d.name == value_buf.get_name())` |
| [`enforce_indirect_access_layout.py:643`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/enforce_indirect_access_layout.py#L643) | `enforce_indirect_access_layout` | raw | `.reads` stored via `next(...)` | outer L595 `for original_op in list(graph.operations):`  · L616 `for value_buf in value_bufs:`  · L640-647 `index_dep = next((d for d in op.get_read_writes().reads if isinstance(d, MemoryDep) and d.name == index_name), None)` |
| [`scratchpad/lx_relayout.py:283`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/lx_relayout.py#L283) | `collect_lx_relayout_plans` | memo | `deps` list-comp | outer L234 `for source_name, consumer_reads in reads.items():`  · L271 `for consumer, dep in consumer_reads:`  · L282-284 `deps = [d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]` |

The two `enforce_indirect_access_layout` sites are in the same
iteration of the same nested loop — every gather/scatter op that has
an indirect-access requirement pays *two* raw
`op.get_read_writes()` calls per value-buffer, even though `op` is
unchanged between them.

### Depth-2 call sites

Grouped by file for readability. All sites are also inside a function
whose outer scope is one of `for op in graph.operations`,
`for op in operations`, `for op in removed_ops`, `for consumer in
consumers`, or similar. Loop headers quoted alongside each call.

**`optimize_restickify.py` — four sites, all raw, all inside `for op in operations` in restickify-cost passes:**

- [`optimize_restickify.py:403`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/optimize_restickify.py#L403)
  `greedy_local_min_cost` — inline. L392 `for op in operations:` · L403 `for dep in op.get_read_writes().reads:`.
- [`optimize_restickify.py:524`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/optimize_restickify.py#L524)
  `compute_future_min_cost` — inline. L521 `for op in operations:` · L524 `for dep in op.get_read_writes().reads:`.
- [`optimize_restickify.py:573`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/optimize_restickify.py#L573)
  `_compute_last_use` — inline. L570 `for op in operations:` · L573 `for dep in op.get_read_writes().reads:`.
- [`optimize_restickify.py:651`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/optimize_restickify.py#L651)
  `beam_global_min_cost` — reused-partial. L640 `for op in operations:` · L651 `deps = [dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)]`.

**`propagate_layouts.py`:**

- [`propagate_layouts.py:1603`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/propagate_layouts.py#L1603)
  `_resolve_copy_back_candidates` — inline. L1602 `for op in removed_ops:` · L1603 `for write in op.get_read_writes().writes:`.

**`scratchpad/allocator.py`:**

- [`scratchpad/allocator.py:2080`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/allocator.py#L2080)
  `_cd_parent_matches` — memo, reused-partial. L2063 `for parent in parent_names:` · L2077-2084 `write_dep = next((w for w in op_read_writes(parent_op).writes if w.name == parent and hasattr(w, "index")), None)`.

**`scratchpad/lx_relayout.py`:**

- [`scratchpad/lx_relayout.py:229`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/lx_relayout.py#L229)
  `collect_lx_relayout_plans` — memo, reused-partial. Outer function-body scope + list-comprehension generator over `for consumer in graph.operations:` at L228 · L229 `deps = [d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]`.

**`scratchpad/utils.py`:**

- [`scratchpad/utils.py:219`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/utils.py#L219)
  `buffer_not_read_in_full` — memo, inline. L218 `for op in graph.operations:` · L219 `for dep in op_read_writes(op).reads:`.
- [`scratchpad/utils.py:286`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/utils.py#L286)
  `_is_read_advancing_anywhere` — memo, reused-partial. L281 `for reader_op, dep in buf_user_deps.get(name, []):` · L285-287 `read_deps = [d for d in op_read_writes(reader_op).reads if isinstance(d, MemoryDep)]`.
- [`scratchpad/utils.py:365`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/utils.py#L365)
  `ops_in_offset_mutation_component` — memo, inline. L364 `for op in graph.operations:` · L365 `for dep in op_read_writes(op).reads:`.
- [`scratchpad/utils.py:463`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/utils.py#L463)
  `get_ncores_for_buffers` — memo, whole-object reused. Outer `for buf_name, users in ...` · L459 `for op, dep in users:` · L463 `op_rw = op_read_writes(op)`.

**`wsr/coarse_tile.py` — four sites, all raw:**

- [`wsr/coarse_tile.py:245`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/wsr/coarse_tile.py#L245)
  `plan_coarse_tile_groups` — reused. Outer per-group loop · L240 `for op in group_ops:` · L245 `rw = op.get_read_writes()`.
- [`wsr/coarse_tile.py:745`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/wsr/coarse_tile.py#L745)
  `_zero_reads_of_fixed_buffers_planned` — reused-partial. L735 `for op in operations:` · L745 `reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]`.
- [`wsr/coarse_tile.py:1738`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/wsr/coarse_tile.py#L1738)
  `validate_reader_tile_advance` — reused-partial (wrapped in `try:` — the only site that treats `get_read_writes()` as fallible). L1728 `for op in operations:` · L1736-1739 `try: reads = [dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)]`.
- [`wsr/coarse_tile.py:3949`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/wsr/coarse_tile.py#L3949)
  `_patch_consumers` — reused-partial. L3891 `for consumer in consumers:` · L3948-3950 `new_reads = [r for r in new_consumer.get_read_writes().reads if isinstance(r, MemoryDep)]`. Immediately follows an `inner_fn` mutation (L3910 `object.__setattr__(consumer.data, "inner_fn", new_inner_fn)`) plus a `replace_computed_buffer_body` rebuild, so the new `ComputedBuffer` is a fresh instance that has never been cached.

### Depth-0 and depth-1 sites

Rolled up counts by file (full JSON is `/tmp/call_sites.json` on the
auditor host and can be regenerated with the reproducer below):

| File | `get_read_writes` | `op_read_writes` | Depths present |
|------|-------------------|------------------|----------------|
| `deadcode_elimination.py` | 3 | 0 | 1,1,1 |
| `dedup_constants.py` | 1 | 0 | 1 |
| `dump_cost_model.py` | 6 | 0 | 0,0,0,0,0,0 |
| `enforce_indirect_access_layout.py` | 4 | 0 | 1,1,3,3 |
| `optimize_restickify.py` | 5 | 0 | 0,2,2,2,2 |
| `padding.py` | 1 | 0 | 1 |
| `pass_utils.py` | 5 | 2 | 0,0,0,0,0,0,1 |
| `propagate_layouts.py` | 6 | 0 | 1,1,1,1,1,2 |
| `scratchpad/allocator.py` | 5 | 8 | 0,0,0,0,0,0,0,0,0,1,1,1,2 |
| `scratchpad/graph_editor.py` | 0 | 1 | 0 |
| `scratchpad/lx_relayout.py` | 0 | 4 | 1,1,2,3 |
| `scratchpad/utils.py` | 0 | 10 | 1,1,1,1,1,1,2,2,2,2 |
| `split_multi_ops.py` | 1 | 0 | 1 |
| `spyre_kernel.py` | 2 | 0 | 1,1 |
| `work_division.py` | 0 | 7 | 0,0,0,0,0,1,1 |
| `work_division_constraints.py` | 0 | 1 | 0 |
| `wsr/coarse_tile.py` | 18 | 0 | 0,0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2 |
| `wsr/coarse_tile_span_overflow.py` | 2 | 0 | 1,1 |
| `wsr/propagate_named_dims.py` | 5 | 0 | 0,0,1,1,1 |
| `wsr/span_overflow_hint_analysis.py` | 4 | 0 | 0,0,0,1 |

Notes on specific in-`pass_utils.py` raw calls that are surprising
because the memoization helper is defined in the same file:

- [`pass_utils.py:114`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L114)
  — inside `op_read_writes` itself (the memo miss path). Correct by
  construction.
- [`pass_utils.py:382`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L382)
  `op_out_coords` — one-liner used across many passes.
- [`pass_utils.py:475`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L475)
  `_build_indirect_store_subs`.
- [`pass_utils.py:691`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L691)
  `_build_indirect_load_subs`.
- [`pass_utils.py:1972`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L1972)
  `format_operations` — diagnostic printer, not a hot path.

## Observed behavior

Torch-Spyre's pre-scheduling passes collectively invoke SymPy
dependency extraction on operations many times per compilation. The
memoization decision — raw vs. memoized — is spread across 21 files
and is made independently at each call site. There is no
process-wide policy: `_patch_consumers` in `wsr/coarse_tile.py`
correctly uses the raw call because it just rebuilt the op instance;
`plan_coarse_tile_groups` in the same file uses the raw call in an
inner loop over `group_ops` even though nothing has mutated those ops
between iterations; `optimize_restickify.py` uses only the raw call in
three separate `for op in operations` loops, each of which runs the
full extraction fresh on every op.

## Upstream behavior

- **v2.13.0 (supported baseline):** `Operation.get_read_writes()`
  (defined on `ComputedBuffer` /
  `torch/_inductor/ir.py`) re-runs `extract_read_writes` — the
  SymPy-based dependency tracer that walks `inner_fn` — on every
  call. Upstream does not cache the result on the op instance.
- **main:** same shape as v2.13.0. Any cache introduced upstream would
  change the equation for the memoization helper, but at
  `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62` no such cache exists.

(Confirmation of the upstream behavior for v2.13.0 and main is
deferred to the pytorch-side portion of this manifest; the claim above
is what the code and the `op_read_writes` docstring both assert.)

## Hidden assumption or duplicated knowledge

The current design distributes the correctness of caching across every
caller. Specifically:

1. **The choice of whether to memoize is local to each call site.**
   There are 21 files and 101 call sites; each decides independently
   whether SymPy retracing is cheap enough to ignore, or whether to
   route through `op_read_writes`. No central pass or policy
   determines this. The result is that
   `enforce_indirect_access_layout` calls the raw function twice per
   iteration of a nested loop, while `scratchpad/utils.py` — which was
   presumably profiled — is fully on the memoized path.
2. **There is no graph-epoch cache.** The memo is keyed on the op
   instance's `__dict__`, and the *only* invariant that keeps it
   correct is "no one mutated this op's `inner_fn` since we cached."
   That invariant is enforced by convention: each pass that mutates
   `inner_fn` is expected to call `invalidate_op_read_writes(op)`.
   Grep confirms this happens exactly once in the codebase, at
   [`scratchpad/graph_editor.py:279`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/scratchpad/graph_editor.py#L279).
   Every other pass that mutates dependencies (e.g.
   `_patch_consumers` in `wsr/coarse_tile.py`) instead relies on
   rebuilding the op via `replace_computed_buffer_body`, which
   produces a fresh instance whose `_ts_cached_read_writes` was never
   set — a correct but load-bearing coincidence.
3. **The "non-planner caller" carve-out is a promise, not an
   enforced rule.** The docstring says non-planner callers should go
   through the real method. Today, `codegen/` and `scheduler.py` do
   not import `op_read_writes` (grep confirms), so the promise holds.
   The next pass that reaches for the memo without noticing that
   promise silently violates the boundary.

The distributed correctness surface is the finding. The performance
question — how much SymPy retracing this actually costs — is the
measurement.

## Evidence

Verbatim excerpts of the three depth-3 sites; loop headers quoted
alongside the call as the finding schema requires.

### Site 1 — `enforce_indirect_access_layout.py:624`

```python
595:    for original_op in list(graph.operations):
...
616:        for value_buf in value_bufs:
...
622:            value_dep = next(
623:                d
624:                for d in op.get_read_writes().reads
625:                if isinstance(d, MemoryDep) and d.name == value_buf.get_name()
626:            )
```

### Site 2 — `enforce_indirect_access_layout.py:643`

Same enclosing loops as Site 1; second raw call in the same
`value_buf` iteration:

```python
640:            index_dep = next(
641:                (
642:                    d
643:                    for d in op.get_read_writes().reads
644:                    if isinstance(d, MemoryDep) and d.name == index_name
645:                ),
646:                None,
647:            )
```

### Site 3 — `scratchpad/lx_relayout.py:283`

```python
234:    for source_name, consumer_reads in reads.items():
...
271:        for consumer, dep in consumer_reads:
...
282:            deps = [
283:                d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)
284:            ]
```

## Reproducer or proof

The static inventory is deterministic and re-derivable from the pinned
SHA. Regenerate it with:

```bash
# 1. Fetch the tree.
curl -s "https://api.github.com/repos/torch-spyre/torch-spyre/git/trees/\
fea0c4be901e1383b1f700dbad8887128b0fcb27?recursive=1" > /tmp/tree.json

# 2. Download every torch_spyre/_inductor/*.py.
jq -r '.tree[] | select(.path | startswith("torch_spyre/_inductor/") \
  and endswith(".py")) | .path' /tmp/tree.json > /tmp/files.txt
mkdir -p /tmp/ts_inductor
while read p; do
  out=/tmp/ts_inductor/$(echo "$p" | sed 's|torch_spyre/_inductor/||' | tr '/' '__')
  curl -s "https://raw.githubusercontent.com/torch-spyre/torch-spyre/\
fea0c4be901e1383b1f700dbad8887128b0fcb27/${p}" -o "$out"
done < /tmp/files.txt

# 3. Static AST scan — see the analyzer at findings/compile-time/
#    (inline in this finding's Measurement-needed section below), or just:
grep -Rn -E "get_read_writes|op_read_writes|extract_read_writes" /tmp/ts_inductor/
```

Depth accounting is by AST (see the counting script in "Measurement
needed" — the same script produces the tables above).

## Compile-time impact

**Measured on the dev pod against `test_flash` at baseline dims
(B=1, H=8, D=128, Lq=512, Lk=1024) — a 90-second cold compile.**

Raw JSONL: [`measurements/2026-08-20/data/test_flash.jsonl`](../../measurements/2026-08-20/data/test_flash.jsonl)
(12,022 records).
Analysis: [`measurements/2026-08-20/data/test_flash_analysis.txt`](../../measurements/2026-08-20/data/test_flash_analysis.txt).

Headline numbers from one cold compile:

| Metric                                                | Value              |
|-------------------------------------------------------|--------------------|
| Total `get_read_writes` (raw) calls                   | **7,050**          |
| Total `op_read_writes` (memoized) calls               | **4,971**          |
| Combined total                                        | **12,021**         |
| `op_read_writes` cache hit rate                       | **95.9%** (4,766 / 4,971) |
| Cache miss cost (p50)                                 | 252 μs             |
| Cache hit cost (p50)                                  | 0.2 μs             |
| Total wall-clock inside these two calls               | **1.86 seconds** (of a 90 s compile ≈ **2.1%**) |
| Per-call cost (all sites) — p50 / p90 / p99 / max     | 152.7 μs / 275.9 μs / 413.4 μs / 161.6 ms |
| Unique caller sites hit at runtime                    | 27                 |
| Unique ops re-analyzed (raw calls)                    | ~180 (top 15 each hit 50–54 times) |

The **top three call sites by total time** hit the same story:

| Rank | Site                                                       | Calls | Total   | Avg     |
|------|------------------------------------------------------------|-------|---------|---------|
| 1    | `dedup_constants.py:70` in `_redirect_consumers`           | 2,460 | 616.5 ms| 250 μs  |
| 2    | `pass_utils.py:691` in `_build_indirect_load_subs`         | 2,431 | 394.8 ms| 162 μs  |
| 3    | `ir.py:5362` in `get_fill_order`                           |   177 | 333.0 ms| 1,881 μs|

Two observations from those three rows:

- **Site 1 alone is 1/3 of all `get_read_writes` time in the compile,
  and it is exactly the O(D × N) scan the sibling finding
  [`01-dedup-constants-quadratic-scan.md`](01-dedup-constants-quadratic-scan.md)
  describes.** The theoretical scan-count math there is not
  theoretical — this measurement is the same code path.
- Site 3 is 25× more expensive per call than sites 1 and 2. Fill-order
  computation walks a larger dependency set. It runs 177 times because
  the same op keeps getting asked for its fill order from different
  passes.

**The top 15 individual ops are each re-analyzed 50–54 times in one
compile.** That is the strongest signal that a graph-epoch cache
would remove real work: the same op's `ReadWrites` is computed dozens
of times per compile without any mutation between calls.

Note that the 95.9% cache hit rate on `op_read_writes` shows the
memoization *is* effective for the sites that use it. The problem is
that 68 of 101 call sites bypass it entirely.

## Runtime impact

None. `get_read_writes` runs at compile time only.

## Correctness impact

None from *calling* the function; potential from *caching* it. The
`op_read_writes` memoization key `_ts_cached_read_writes` is only
correct while `inner_fn` is unchanged. Today, the two places that
mutate `inner_fn` are handled safely (one uses
`invalidate_op_read_writes`; the other rebuilds the op). A third,
future mutation-in-place that forgets to invalidate would return a
stale `ReadWrites` object to whichever pass next asked. The current
design places that correctness on the pass author's memory.

## Measurement method

The numbers in "Compile-time impact" above were captured with a
monkey-patch that wraps `ComputedBuffer.get_read_writes` and
`torch_spyre._inductor.pass_utils.op_read_writes` and writes one
JSONL record per call (caller-frame, op type, op name, elapsed μs,
cache hit/miss). No torch-spyre source modification.

- Patch: [`measurements/2026-08-20/patches/instrument_read_writes.py`](../../measurements/2026-08-20/patches/instrument_read_writes.py)
- Driver: [`measurements/2026-08-20/scripts/run_test_flash.py`](../../measurements/2026-08-20/scripts/run_test_flash.py)
- Analyzer: [`measurements/2026-08-20/scripts/analyze.py`](../../measurements/2026-08-20/scripts/analyze.py)

**Environment.**

- Pod: `tdeshane-compiler-timing-dev-v2` (node `p1-worker-23`,
  `a5-deepview` cluster)
- torch-spyre HEAD: `fea0c4be901e1383b1f700dbad8887128b0fcb27` (fetched
  and checked out for this audit; `_C.so` rebuilt via `make setup` on
  the pod)
- torch: `2.13.0+cpu`
- Python: 3.12
- Cold compile — `TORCHINDUCTOR_CACHE_DIR` wiped between runs

**Reproduce.**

```bash
# On the pod:
cd $HOME/torch-spyre-work/torch-spyre
source .venv/bin/activate
export TORCH_SPYRE_INSTRUMENT_LOG=/tmp/audit_test_flash.jsonl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_audit_test_flash
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_SPYRE_INSTRUMENT_LOG"
python /path/to/torch-spyre-notes/measurements/2026-08-20/scripts/run_test_flash.py
```

Then analyze locally:

```bash
python measurements/2026-08-20/scripts/analyze.py \
    measurements/2026-08-20/data/test_flash.jsonl
```

**A synthetic dedup-stress workload** was attempted at
[`scripts/run_synthetic_dedup.py`](../../measurements/2026-08-20/scripts/run_synthetic_dedup.py)
to isolate the O(D × N) scan cost from a real compile. The compile
aborted at `pass_utils.py:919` inside `device_coordinates` with
`InductorError: Spyre backend does not support: Unexpected stick
expression 1` — the `F.pad`+`stack`+`sum` pattern hits a Spyre-backend
layout restriction before `dedup_and_promote_constants` runs. Only 84
records were captured (all in `deadcode_elimination.live_operations`).
See [`data/synthetic_dedup_partial.jsonl`](../../measurements/2026-08-20/data/synthetic_dedup_partial.jsonl).
Future runs should design a synthetic that avoids the stick-expression
restriction. The test_flash numbers above are sufficient to support
the finding without it.

## Suggested change

Replace the distributed cache with a **graph-epoch cache**. Sketch:

- One process-wide (or `GraphLowering`-scoped) cache keyed on
  `(id(op), graph.mutation_epoch)`.
- `graph.mutation_epoch` is a monotonically increasing integer stamped
  onto the `GraphLowering` and bumped by exactly the operations that
  change what `extract_read_writes` would see: any code path that
  writes to `graph.operations`, replaces an op via
  `replace_computed_buffer_body`, mutates `inner_fn` on an existing
  op, or edits ranges/indices.
- `op_read_writes(op)` returns the cached entry when the epoch matches
  and the op-id matches; otherwise retraces and updates the cache.
- `invalidate_op_read_writes` is deleted — every mutation site bumps
  the epoch instead, which is a smaller, easier-to-audit contract.
- Existing callers of the raw method migrate to the epoch-cached
  helper. The three raw `pass_utils.py` sites (382, 475, 691) are the
  first migration targets; the four `optimize_restickify.py` sites
  (403, 524, 573, 651) and the five `wsr/coarse_tile.py` depth-2 sites
  are the highest-return targets.

Detailed design is out of scope for this finding — it needs its own
finding once the measurement above is in hand and the cost is proven
non-negligible.

## Skill / contract update

Create `contracts/dependency-extraction.md` documenting:

1. That `Operation.get_read_writes()` is uncached in upstream PyTorch
   at v2.13.0 and at main, and that any torch-spyre pass that reads
   dependencies more than once per op-instance is responsible for its
   own caching.
2. The three-way call-site classification the finding uses (raw,
   memoized via `op_read_writes`, memoized via a future graph-epoch
   cache).
3. The invalidation invariant: for the current per-op memo, every
   pass that mutates `inner_fn` in place must call
   `invalidate_op_read_writes(op)` before the next reader. Any pass
   that uses `replace_computed_buffer_body` is safe by construction
   because that path produces a fresh op instance.
4. A checklist for reviewers: "if you see
   `.get_read_writes()` inside a `for` loop over `graph.operations`
   without prior variable capture, ask why it isn't `op_read_writes`."

The scan that produced this inventory (the `curl` +
`grep -Rn -E "get_read_writes|op_read_writes"` combination above) is
the seed of a repeatable audit — add it to `scans/` as
`scans/get_read_writes_sites.sh` so that regressions in the ratio of
raw vs. memoized call sites can be caught on every torch-spyre bump.
