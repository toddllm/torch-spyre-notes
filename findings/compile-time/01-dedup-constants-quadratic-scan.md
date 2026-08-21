# `dedup_and_promote_constants` performs O(D·N) full-operations scans with un-memoized `get_read_writes`

- **Id:** CT-01
- **Category:** compile-time
- **Created:** 2026-08-20
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** confirmed (reproduced against `test_flash.jsonl`; 2,460 calls / 616.5 ms measured at `dedup_constants.py:70` `_redirect_consumers`)
- **Status:** open

## Summary

The step-2 dedup loop in `dedup_and_promote_constants` scans the
entire `operations` list once per duplicate. For each duplicate `dup`,
`_redirect_consumers` iterates every op in `operations` and calls
`op.get_read_writes()` — not the memoizing
`pass_utils.op_read_writes` helper that already exists in this
codebase for exactly this workload — to test whether the op reads
`dup.get_name()`. With `D` total duplicates across all groups and `N`
operations, this is `Θ(D·N)` calls to `get_read_writes()`, each of
which re-runs SymPy dependency extraction inside upstream
`ComputedBuffer.get_read_writes()`. In addition, `_drop_constant`
runs `operations.remove(dup)` once per duplicate, contributing
another `Θ(D·N)` cost in list-linear-scan comparisons. The natural
shape is a single pre-pass that builds a `name → consumers` reverse
index once and reuses it for every group; the memoization helper
already exists but is not applied here.

## Files and symbols

- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `_redirect_consumers` (lines 52–79, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L52-L79>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `_drop_constant` (lines 82–106, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L82-L106>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `dedup_and_promote_constants` step 2 (lines 131–138, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L131-L138>)
- torch-spyre: `torch_spyre/_inductor/pass_utils.py` — `op_read_writes` (lines 102–116, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L102-L116>)
- torch-spyre: `torch_spyre/_inductor/pass_utils.py` — `invalidate_op_read_writes` (lines 130–140, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/pass_utils.py#L130-L140>)
- upstream v2.13.0: `torch/_inductor/ir.py` — `ComputedBuffer.get_read_writes` (uncached; recomputes via `dependencies.extract_read_writes`).
- upstream main: `torch/_inductor/ir.py` — `ComputedBuffer.get_read_writes` (unchanged shape at `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`; still uncached at the class level, hence the private memo on the torch-spyre side).

## Observed behavior

The relevant snippet from `dedup_and_promote_constants`:

```python
# --- Step 2: dedup ---
for key, group in groups.items():
    if len(group) <= 1:
        continue
    canonical = group[0]
    for dup in group[1:]:
        _redirect_consumers(operations, dup, canonical)
        _drop_constant(operations, dup, canonical)
```

`_redirect_consumers` (verbatim):

```python
def _redirect_consumers(
    operations: list[Operation],
    dup: SpyreConstantFallback,
    canonical: SpyreConstantFallback,
) -> None:
    """Rewrite every ComputedBuffer consumer of dup to read canonical instead."""
    D = dup.get_name()
    C = canonical.get_name()
    name_map = {D: C}

    # Do not dedup a constant that is itself a graph output.
    if D in V.graph.get_output_names():
        logger.debug("dedup_and_promote_constants: skipping output constant %s", D)
        return

    for op in operations:
        if op is dup or op is canonical:
            continue
        rw = op.get_read_writes()
        if not any(dep.name == D for dep in rw.reads):
            continue
        if isinstance(op, ComputedBuffer):
            _patch_inner_fn(op, name_map)
        else:
            raise AssertionError(
                f"dedup_and_promote_constants: unsupported consumer type "
                f"{type(op).__name__} reads constant {D!r} — cannot rewrite"
            )
```

The inner `for op in operations` loop walks the full operations list
and calls `op.get_read_writes()` on every op that is not the current
`dup` or `canonical`. This scan is redone from scratch for every
duplicate.

`_drop_constant` also does a linear-time removal:

```python
operations.remove(dup)
```

Python's `list.remove` scans from the head until the element is
matched by identity/equality; on an `N`-element list this is `Θ(N)`
comparisons in the worst case per call.

The memoizing helper that already exists in this codebase — placed
there because upstream `ComputedBuffer.get_read_writes` re-runs SymPy
dependency extraction on every call — is:

```python
def op_read_writes(op: Operation) -> ReadWrites:
    """``op.get_read_writes()`` memoized on the op instance.

    ``ComputedBuffer.get_read_writes`` re-runs sympy dependency extraction on
    every call and is not cached upstream, yet its result does not depend on the
    only thing the LX planner mutates (``op_it_space_splits``). The scratchpad
    pass calls it hundreds of times, so we cache it under a private key only this
    helper reads -- a non-planner caller (e.g. later-pass codegen) still goes
    through the real method.
    """
    rw = op.__dict__.get("_ts_cached_read_writes")
    if rw is None:
        rw = op.get_read_writes()
        op.__dict__["_ts_cached_read_writes"] = rw
    return rw
```

`_redirect_consumers` calls the raw `op.get_read_writes()` and
therefore does not benefit from the memoization. Every duplicate's
scan re-extracts every op's reads from SymPy.

## Upstream behavior

- **v2.13.0 (supported baseline):** `ComputedBuffer.get_read_writes`
  is not cached at the class level. Each call constructs a
  `LoopBody` (or reuses `get_default_sizes_body`'s cached body) and
  calls `dependencies.extract_read_writes(self.get_store_function(),
  ...)`, which walks the `inner_fn` under a
  `RecordLoadStoreInner` handler and rebuilds `MemoryDep` objects
  through SymPy expression indexing. This is the "expensive per-call
  operation" the torch-spyre `op_read_writes` helper's docstring
  refers to.
- **main:** unchanged shape at
  `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`.
  `ComputedBuffer.get_read_writes` still has no persistent memo at
  the base-class level; the `LoopBody` and dependency extraction
  work are still performed per call.

## Hidden assumption or duplicated knowledge

Two overlapping bits of knowledge:

1. **"Consumer discovery by read-write analysis is expensive; use
   `op_read_writes`."** This lesson is already encoded in
   `pass_utils.op_read_writes` and used by the scratchpad pass, but
   `dedup_and_promote_constants` was written without applying it.
   The rule "if you call `get_read_writes()` inside a loop, call
   `op_read_writes()` instead" is folklore, not enforced by any lint
   or wrapper.

2. **"Consumer discovery over the whole operations list should happen
   once, not once per producer."** A reverse index
   (`name → consumers`) built in a single sweep would make each
   duplicate's redirect step O(|consumers(D)|) instead of O(N).
   Upstream Inductor already maintains `V.graph.name_to_users` for
   exactly this purpose — the case-study document
   (`cases/dedup-and-promote-constants.md` §5.2) notes that the pass
   uses read-write analysis instead of `name_to_users` because the
   latter can be stale during a pass invocation. That is a real
   constraint at the moment `_patch_inner_fn` has already run
   against some consumers, but nothing forces the initial
   consumer-discovery sweep to be reissued per duplicate — it can be
   built once at the top of step 2 from the original state, and
   updated in place as consumers are patched.

## Evidence

The full step-2 caller and the inner scan are already quoted verbatim
in "Observed behavior" above. Both loops are quoted rather than
paraphrased because the finding rests on their nesting.

Scan-count math at the pinned SHA:

- Let `N = len(operations)` after all creators (`convert_constant_with_graph_node`,
  `lower_pad_sequence`, `split_multi_ops` constant materialization,
  `coarse_tile` accumulator fill).
- Let `K = |groups with >= 2 duplicates|`, and for the `i`-th such
  group let `d_i = |group_i|` and `D = Σ (d_i - 1)` (total duplicates
  removed).
- The inner `for op in operations` loop in `_redirect_consumers`
  runs `N - 2` iterations per invocation. Each iteration performs one
  identity check plus `op.get_read_writes()`, which for
  `ComputedBuffer` re-runs `dependencies.extract_read_writes`.
- `_redirect_consumers` is called `D` times → total inner-loop
  iterations `≈ D · (N - 2)`.
- Total `get_read_writes()` calls: `≈ D · (N - 2)`.
- Total `operations.remove(dup)` cost: each `.remove` is `Θ(N)`
  comparisons; called `D` times → `Θ(D · N)` list scan work.

The two contributions compose. `D · N`
`get_read_writes()` calls is the dominant cost when `N` is dominated
by `ComputedBuffer`s (typical for the graphs this pass operates on,
per §3–§4 of `cases/dedup-and-promote-constants.md`).

Worked example: a graph with 500 total operations, 20 spyre.constant
ops splitting into 4 groups of 5 duplicates each (so `D = 4·4 = 16`
total duplicates, `N = 500`):
- `get_read_writes()` calls: `≈ 16 · 498 ≈ 7968`.
- `list.remove` element-scans: `≈ Σ_{i=0..15} (N - i) ≈ 7880`.

The `get_read_writes()` cost is the interesting one because each
call runs SymPy dependency extraction against an `inner_fn` that has
not changed between calls for that op.

## Reproducer or proof

Static, this finding is `plausible`. To promote to `reproduced`:

1. On the dev pod (`a5-deepview`, per the memory index), check out
   torch-spyre at `fea0c4be901e1383b1f700dbad8887128b0fcb27`. Ensure
   `torch_spyre` is importable in the `.venv` (see
   `project_torch_spyre_dev_pod_layout` memory entry).
2. Add a temporary counter around the loop, e.g.:

   ```python
   # torch_spyre/_inductor/dedup_constants.py, inside _redirect_consumers
   import itertools
   _ct = itertools.count()

   for op in operations:
       next(_ct)
       ...
   ```

   And log the total at the end of `dedup_and_promote_constants`. Or,
   more surgically, wrap `Operation.get_read_writes` with a global
   counter for the duration of `dedup_and_promote_constants`.
3. Compile any model that produces at least one duplicate group. The
   existing test `tests/inductor/test_dedup_constants.py::TestDedupConstants::test_dedup_across_same_dtype_pad_sequences`
   at `fea0c4b` already constructs a graph with 4 constants
   collapsing to 1. Run it and capture the counter.
4. Also record the elapsed wall time of the pass (torch-spyre already
   emits `time.perf_counter()` elapsed-ms lines for each
   pre-scheduling pass — see the operating brief rule about not
   calling something a perf bug without measurement).

The counter should show `D · (N-2)` calls to
`op.get_read_writes()`; the elapsed-ms line will show the pass's
wall-clock cost. Both go into an updated Compile-time impact section
and promote the finding to `reproduced`.

## Compile-time impact

**Measured on the dev pod against a `test_flash` cold compile at
baseline dims (see the sibling finding
[`02-get-read-writes-inventory.md`](02-get-read-writes-inventory.md)
for the measurement environment and method).**

The `op.get_read_writes()` call inside `_redirect_consumers` at
`dedup_constants.py:70` is the **#1 hottest `get_read_writes` site
in the whole compile**, by both count and total time:

| Site                                              | Calls | Total time | Avg per call |
|---------------------------------------------------|-------|------------|--------------|
| `dedup_constants.py:70` in `_redirect_consumers`  | **2,460** | **616.5 ms** | 250 μs |

For context, this single site accounts for **33% of the 1,856 ms
that the whole compile spends inside `get_read_writes` and
`op_read_writes` combined** (7,050 raw + 4,971 memoized = 12,021
total calls). No other site in the compile is close.

The 2,460 calls come from a modest number of duplicate constants —
`test_flash` at baseline dims does not stress the pattern
particularly hard. The scan cost scales as `D · N` where `D` is the
number of non-canonical duplicate constants and `N` is the size of
the operation list at the point the pass runs. A workload with more
constants (larger models, models with many masking / padding
operations, or wider fusion opportunities) would push both `D` and
`N` up together, and the site's contribution would grow
multiplicatively.

Analytical scaling: `Θ(D · N)` un-memoized `get_read_writes()`
calls plus `Θ(D · N)` `list.remove` element comparisons. Per-call
`get_read_writes()` cost for a `ComputedBuffer` includes a SymPy
`extract_read_writes` traversal; the measurement above confirms 250
μs per call on average at this site.

Raw data: [`measurements/2026-08-20/data/test_flash.jsonl`](../../measurements/2026-08-20/data/test_flash.jsonl).

## Runtime impact

None. This finding is purely compile-time.

## Correctness impact

None. The change described in "Suggested change" is a refactor of
the discovery pattern; it does not alter which consumers are
rewritten or how the surviving constants are re-ordered.

## Measurement method

See [`02-get-read-writes-inventory.md`](02-get-read-writes-inventory.md#measurement-method).
The same instrumentation, driver, and cold compile that produced the
inventory-wide numbers also captured the site-specific numbers used
here. `dedup_constants.py:70` was extracted from the JSONL via
[`measurements/2026-08-20/scripts/analyze.py`](../../measurements/2026-08-20/scripts/analyze.py).

## Suggested change

Rewrite step 2 to (a) build the reverse index once and (b) use the
memoized `op_read_writes` throughout. Sketch:

```python
from .pass_utils import op_read_writes, invalidate_op_read_writes

# Build name -> list of consumer ops once, over the un-mutated graph.
constant_names = {op.get_name() for op in operations if isinstance(op, SpyreConstantFallback)}
consumers_by_name: dict[str, list[ComputedBuffer]] = {}
for op in operations:
    if not isinstance(op, ComputedBuffer):
        continue
    for dep in op_read_writes(op).reads:
        if dep.name in constant_names:
            consumers_by_name.setdefault(dep.name, []).append(op)

output_names = set(V.graph.get_output_names())

# Step 2: for each group, redirect the pre-computed consumer lists.
for key, group in groups.items():
    if len(group) <= 1:
        continue
    canonical = group[0]
    for dup in group[1:]:
        D = dup.get_name()
        assert D not in output_names, (
            f"dedup_and_promote_constants: constant {D!r} is a graph output"
        )
        for consumer in consumers_by_name.get(D, []):
            _patch_inner_fn(consumer, {D: canonical.get_name()})
            invalidate_op_read_writes(consumer)
        _drop_constant(operations, dup, canonical)
```

Complexity after the change:

- One up-front sweep over `operations`, one `op_read_writes` call per
  op (memoized after the first). Total `get_read_writes` invocations:
  `N` on the first pass and 0 on subsequent passes for the same
  compilation, versus `D · N` today.
- Consumer redirect per duplicate is `O(|consumers(D)|)` instead of
  `O(N)`.
- `operations.remove(dup)` is still `Θ(N)` per duplicate; this can be
  removed by tracking the set of duplicates to drop and doing one
  filtered `operations[:] = [op for op in operations if op not in dropped]`
  at the end (`Θ(N)` total instead of `Θ(D · N)`).
- The graph-output guard, correctly analysed in the sibling
  correctness finding as unreachable, is promoted to an assertion at
  the caller.

Two secondary details:

- `invalidate_op_read_writes(consumer)` is called after
  `_patch_inner_fn` because that helper wraps `inner_fn` — the
  memoized reads are now stale for that op. `pass_utils.py`
  documents `invalidate_op_read_writes` for exactly this pattern:
  "Call this immediately after mutating an op's dependencies in place
  -- e.g. swapping a load name in its `inner_fn`."
- The reverse-index build reads dependencies on ops that have not yet
  been patched by this pass, so a per-op `op_read_writes` cache
  populated during the build remains correct for the rest of the
  pass — patched consumers get their memo invalidated at patch time.

## Skill / contract update

Contract file to update or create:

- `contracts/expensive-analysis-caching.md` (or wherever the
  read-writes memoization policy lives). Add the rule: **any pass
  that calls `op.get_read_writes()` inside a loop over
  `V.graph.operations` MUST use `pass_utils.op_read_writes` instead
  of the raw method, and MUST call `invalidate_op_read_writes(op)`
  after any mutation of `op.data.inner_fn` or the equivalent.**
- If a scan-cache lint doesn't exist, propose one under `scans/`: a
  grep for `\.get_read_writes\s*\(` inside `torch_spyre/_inductor/`
  should surface every raw call site; each match must be reviewed
  against the rule above.

Lesson for the audit: the existence of a private memoization helper
in `pass_utils.py` is a signature that upstream has an
un-cached-but-expensive method — and that any raw call to that
method inside a loop over graph operations is a compile-time smell
by default. Log this as a recurring investigation pattern (this is
also the second target of the manifest's investigation #2:
`get_read_writes()`/`op_read_writes()` call-site inventory).
