---
name: computed-buffer
pinned_sha: fea0c4be901e1383b1f700dbad8887128b0fcb27
pytorch_supported: v2.13.0
pytorch_main: c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62
status: draft
---

# ComputedBuffer — synthetic construction, identity, and handler nesting

This contract enumerates every place torch-spyre builds or rewrites a
`ComputedBuffer` outside the normal Inductor lowering path, states the
identity rules those sites imply, and flags the invariants that are
either violated in-tree or that the code is fragile against.

All citations resolve inside `/tmp/ts-pinned-scan/fea0c4b/`.  Upstream
citations are into a local checkout at pytorch main
(`c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`) — line numbers stay stable
for `torch/_inductor/ir.py` and `torch/_inductor/graph.py` on
`v2.13.0`.

## 1. Header — what `ComputedBuffer` is upstream

`ComputedBuffer` is a `Buffer` + `Operation` composite whose payload is a
`Loops` subclass (`Pointwise` or `Reduction`).  The `Operation` half is
a frozen-`__post_init__` dataclass whose sole identity attribute is
`operation_name`, initialized to `None` and assigned later by
`GraphLowering.register_operation`:

- `torch/_inductor/ir.py:855` `class Operation:` — `operation_name:
  Optional[str] = None` in `__post_init__`.
- `torch/_inductor/ir.py:871`
  ```python
  def get_operation_name(self) -> str:
      assert self.operation_name is not None
      return self.operation_name
  ```
- `torch/_inductor/graph.py:939`
  ```python
  def register_operation(self, op: ir.Operation) -> str:
      assert op.operation_name is None, f"Operation registered twice: {op}"
      assert isinstance(op, ir.Operation)
      name = self.qualify_name(f"op{len(self.operations)}")
      self.operations.append(op)
      self.name_to_op[name] = op
      op.operation_name = name
      return name
  ```

Two facts fall out of `register_operation` that the rest of this contract
turns on:

- **The name is derived from `len(self.operations)` at the moment of
  registration.**  It is a positional bookkeeping label, not a stable
  identity.  Once an operation is physically removed from
  `self.operations`, its former name is orphaned but not reserved — a
  later `register_operation` will happily reuse the same numeric suffix.
- **Registration asserts `op.operation_name is None`.**  Torch-spyre
  cannot call `register_operation` a second time on the same buffer
  object.  Every synthetic construction site therefore either goes
  through `register_operation` exactly once, or bypasses it entirely and
  assigns `operation_name` by hand.

`ComputedBuffer` itself is a frozen dataclass whose `data` field cannot
be reassigned.  Any pass that wants to rewrite the body of an existing
`ComputedBuffer` must construct a *new* `ComputedBuffer` and splice it
into `graph.operations` — this is the pattern
`replace_computed_buffer_body` codifies (§3).

## 2. Synthetic construction sites

Every direct `ComputedBuffer(...)` invocation in
`/tmp/ts-pinned-scan/fea0c4b/torch_spyre/`, clustered by purpose:

### 2a. Fully synthetic ops (a fresh op the graph did not lower)

| File:Line | Site | Purpose |
| --- | --- | --- |
| `torch_spyre/_inductor/lowering.py:1286` | `_build_mutation_lowering` | `spyre.copy_forced` / `spyre.opaque_copy_` lowering builds a `ComputedBuffer` with `MutationLayoutSHOULDREMOVE(dst)` so the mutating write into `dst` survives regardless of what the scheduler would otherwise decide. |
| `torch_spyre/_inductor/scratchpad/graph_editor.py:172` | `_clone_buffer_op` | Clones a buffer to break a false anti-dependency in scratchpad planning. |

The lowering.py site is the *only* one that goes through the upstream
registration flow:

```python
# torch_spyre/_inductor/lowering.py:1286
buffer = ir.ComputedBuffer(
    name=None,
    layout=ir.MutationLayoutSHOULDREMOVE(dst),
    data=pw.data.data,
)
buffer.name = V.graph.register_buffer(buffer)
V.graph.register_operation(buffer)
```

`register_operation` assigns `buffer.operation_name = op{N}` for
`N = len(self.operations)` (see §1).

The scratchpad `_clone_buffer_op` site also registers via the graph
(`self.lowering.register_operation(new_com_buf)` at
`torch_spyre/_inductor/scratchpad/graph_editor.py:182`) but then
physically moves the op in `self.lowering.operations` at
`graph_editor.py:226`:
```python
# graph_editor.py:225
self.lowering.operations.remove(new_com_buf)
self.lowering.operations.insert(insert_idx, new_com_buf)
```
The `operation_name` assigned by `register_operation` is preserved
across the remove/insert, but its **numeric suffix now no longer
matches the operation's positional index in `operations`** — this is
the first place object identity and name identity diverge.

### 2b. Copy / fill / combine helpers (`wsr/coarse_tile.py`)

Four sites all follow the same pattern: build a `ComputedBuffer` locally,
assign `operation_name` **by hand**, then splice into `operations`
without calling `register_operation`:

| File:Line | Var | Sets `operation_name` at |
| --- | --- | --- |
| `wsr/coarse_tile.py:2330` | `copy_buf` (write copy-out for a tiled op) | `wsr/coarse_tile.py:2336` |
| `wsr/coarse_tile.py:2921` | `copy_buf` (read copy-in shared across tiled consumers) | `wsr/coarse_tile.py:2923` |
| `wsr/coarse_tile.py:3421` | `combine_buf` (reduction combine op) | `wsr/coarse_tile.py:3427` |
| `wsr/coarse_tile.py:3520` | `copy_buf` (reduction accum_tile → accum_full) | `wsr/coarse_tile.py:3526` |
| `wsr/coarse_tile.py:3738` | `fill_buf` (fill identity for reduction accum) | `wsr/coarse_tile.py:3744` |

Representative site — write copy-out for a tiled op:

```python
# wsr/coarse_tile.py:2329
copy_name = V.graph.qualify_name(f"coarse_tile_copy_{tiled_op.get_name()}")
copy_buf = ComputedBuffer(
    name=copy_name,
    layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf))),
    data=copy_data,
)
copy_buf.origins = tiled_op.origins
copy_buf.operation_name = copy_name
```

`copy_name` is `V.graph.qualify_name("coarse_tile_copy_" +
tiled_op.get_name())`.  Since the upstream `register_operation` mints
names as `op{len(self.operations)}` (§1), the hand-assigned
`copy_name = "coarse_tile_copy_..."` cannot collide with a real
registered op name, but it also **does not participate in
`name_to_op`** — see §5 for the fallout.

Only two of these sites subsequently install the copy into
`V.graph.name_to_buffer` (a Buffer-level dict), never
`V.graph.name_to_op`:
- `wsr/coarse_tile.py:3062` `V.graph.name_to_buffer[copy_name] = copy_buf`
- `wsr/coarse_tile.py:3471` `V.graph.name_to_buffer[combine_name] = combine_buf`
- `wsr/coarse_tile.py:3754` `V.graph.name_to_buffer[fill_name] = fill_buf`

No coarse-tile helper ever calls `V.graph.register_operation` or writes
into `V.graph.name_to_op`.  A synthetic op inserted by these helpers is
visible via `graph.operations.index(op)` but *not* via `name_to_op`.

### 2c. Wrapper for a mutation-op rewrite (`insert_restickify.py`)

`insert_restickify_on_node_inputs` rebuilds the consumer as a fresh
`ComputedBuffer` object to invalidate the `get_default_sizes_body`
instance-keyed cache:

```python
# torch_spyre/_inductor/insert_restickify.py:242
new_consumer_buffer = ComputedBuffer(
    name=op.get_name(),
    layout=op.layout,
    data=op.data,
    _split_size=op._split_size,
    _original_inner_fn=op._original_inner_fn,
    _original_ranges=op._original_ranges,
    _original_reduction_ranges=op._original_reduction_ranges,
)
new_consumer_buffer.operation_name = op.operation_name
```

The reused metadata is:
- `name` (Buffer name) — reused verbatim.
- `layout` — reused verbatim.
- `data` — the same `Pointwise`/`Reduction` object, whose `inner_fn`
  has just been rewritten via `object.__setattr__` at
  `insert_restickify.py:238`.
- The four `_original_*` fields Inductor uses to reconstruct the
  pre-split loop body.
- `operation_name` — copied on the next line (`insert_restickify.py:251`).

This is a pure *identity replay*: same operation_name, same buffer
name, same layout, freshly constructed `ComputedBuffer` object.  The
new object is spliced into the operations list at
`insert_restickify.py:260`:
```python
operations[op_index] = new_consumer_buffer
V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer
```

Note this site does **not** update `V.graph.name_to_op[operation_name]
= new_consumer_buffer` — after this splice, `name_to_op` still points
at the *old* `ComputedBuffer` object.  This is the same
identity-vs-name split as §2a's scratchpad clone.

### 2d. Relocation / body-rewrite helper (`pass_utils.replace_computed_buffer_body`)

`replace_computed_buffer_body` is the canonical helper for §2c's
pattern.  It is the shared implementation for
`padding.py:154`, `split_multi_ops.py:682`, and three callsites in
`wsr/coarse_tile.py` (3147, 3911, 4291):

```python
# torch_spyre/_inductor/pass_utils.py:1310
def replace_computed_buffer_body(
    op: ComputedBuffer,
    new_data: Loops,
    operations: list[Operation],
    *,
    pass_name: str,
    reason: str | None = None,
) -> ComputedBuffer:
    ...
    new_buf = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=new_data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_buf.operation_name = op.operation_name
    preserve_provenance(op, new_buf, pass_name=pass_name, reason=reason)
    copy_op_metadata(op, new_buf)
    ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)

    op_idx = operations.index(op)
    operations[op_idx] = new_buf
    return new_buf
```

The identity replay is identical to §2c.  What
`replace_computed_buffer_body` adds:
- `preserve_provenance` copies `.origins` and `.origin_node`.
- `copy_op_metadata` (`loop_info.py`) copies `.loop_info` and other
  torch-spyre-specific attributes.
- `get_default_sizes_body.clear_cache(new_buf)` — the whole point of
  reconstructing the object rather than mutating in place.

The docstring at `pass_utils.py:1320` calls out the fields it copies:
`operation_name`, `origins`, `origin_node`, and `_split_size` /
`_original_*`.

## 3. Replacement identity rules

Reading the four buffer-rewrite sites together, the invariants an
in-place `ComputedBuffer` replacement (as opposed to a fresh op
insertion) must satisfy are:

1. **Buffer name preserved.**  `new_buf.name = op.get_name()` —
   consumers keyed off the buffer name never resolve to a missing
   buffer.  Cited: `pass_utils.py:1334`, `insert_restickify.py:243`.

2. **`operation_name` preserved.**  `new_buf.operation_name =
   op.operation_name` — otherwise the copy op leaves
   `operation_name = None` and `get_operation_name()` asserts.
   Cited: `pass_utils.py:1342`, `insert_restickify.py:251`.

3. **Layout preserved.**  `new_buf.layout = op.layout` — layout
   decisions from earlier passes (finalize_layouts, stickification)
   must not silently regress.  Cited: `pass_utils.py:1335`,
   `insert_restickify.py:244`.

4. **Position in `operations` preserved.**  The replacement uses
   `operations[op_idx] = new_buf` (`pass_utils.py:1348`) or
   `operations[op_index] = new_consumer_buffer`
   (`insert_restickify.py:260`), not `operations.remove()` +
   `operations.append()` — the topological position stays fixed so
   downstream `op_position` dicts (§5) stay valid until the next
   rebuild.

5. **`origins` / `origin_node` preserved.**  Done inside
   `preserve_provenance`.  If a rebuild skips this (see §5's list of
   fields `_clone_buffer_op` sets by hand), FX provenance is lost and
   downstream passes that key off `origins` (e.g. `split_multi_ops`,
   `deadcode_elimination`) misbehave.

6. **`loop_info` preserved.**  Done inside `copy_op_metadata`.  This
   is a torch-spyre extension — the coarse-tile passes attach
   `loop_info` as a bare attribute (via `object.__setattr__`) and every
   rebuild must copy it forward or the tile-advance metadata gets
   dropped mid-pipeline.

7. **`get_default_sizes_body` cache cleared.**  Cited:
   `pass_utils.py:1345`, `insert_restickify.py:264`.  The cache is
   instance-keyed on the `ComputedBuffer` object, so a new object
   automatically has no entries; the explicit `clear_cache` call is
   defensive against future changes to the cache implementation.

8. **The users list is *not* mentioned.**  Neither
   `replace_computed_buffer_body` nor
   `insert_restickify_on_node_inputs` touches
   `V.graph.name_to_users`.  This is safe iff the buffer name is
   preserved (invariant 1), because `name_to_users` is keyed by buffer
   name, not by object identity.  A pass that *changes* the buffer
   name during rebuild would silently orphan the users list — no
   in-tree pass does this today, but the invariant is undocumented.

9. **`name_to_op` is NOT rewired to the new object.**  Both rewrite
   sites update `V.graph.name_to_buffer[new_op.get_name()] = new_op`
   (`pass_utils.py` does not; `replace_computed_buffer_body`'s callers
   do — e.g. `split_multi_ops.py:689`,
   `wsr/coarse_tile.py:3154/3299/3918`), but no callsite updates
   `V.graph.name_to_op[new_op.operation_name] = new_op`.  After a
   rebuild, `name_to_op[operation_name]` still points to the *stale*
   `ComputedBuffer` object, while `operations[op_idx]` points to the
   fresh one.  See §5 for whether any consumer actually reads
   `name_to_op` on this axis.

## 4. WrapperHandler nesting

Torch-spyre installs `WrapperHandler` subclasses that patch an
inner_fn's `ops.load(name, index)` calls.  Every install site follows
the same recipe: capture `orig_inner = op.data.inner_fn`, define a
`new_inner_fn` closure that enters a `V.set_ops_handler(...)` block and
calls `orig_inner(*args)`, then `object.__setattr__(op.data,
"inner_fn", new_inner_fn)` and rebuild the `ComputedBuffer` via
`replace_computed_buffer_body`.  Instances in-tree:

| File:Line | Handler class | Purpose |
| --- | --- | --- |
| `torch_spyre/_inductor/insert_restickify.py:76` | `NameSwapHandler` | Rename input buffers after restickify insertion. |
| `torch_spyre/_inductor/dedup_constants.py:45` | `NameSwapHandler` (reused from insert_restickify) | Redirect a consumer of a duplicate constant to the canonical constant. |
| `torch_spyre/_inductor/wsr/coarse_tile.py:2487` | `_NameSwapHandler` (locally-defined, index-rescaling variant) | Redirect consumer loads from full_buf to tile-local copy_buf, rescaling stride coefficients. |
| `torch_spyre/_inductor/wsr/coarse_tile.py:3143` | `_NameSwapHandler` (as above) | Same, installed by `_patch_consumer_to_read_copy`. |
| `torch_spyre/_inductor/wsr/coarse_tile.py:3906` | `NameSwapHandler` (imported from insert_restickify) | Redirect outside consumers from tiled-op scratch to full-sized output, in `_patch_consumers`. |
| `torch_spyre/_inductor/wsr/coarse_tile.py:4180` | `_RetileLoadIndexHandler` | Rewrite retiled load indexes for consumers of retiled buffers. |
| `torch_spyre/_inductor/wsr/coarse_tile.py:4197` | `_NameAndIndexSwapHandler` | Combines name-swap + retile-index-rewrite for `_patch_consumers` when strides differ. |
| `torch_spyre/_inductor/wsr/coarse_tile.py:4287` | `_RetileLoadIndexHandler` (as above) | Installed by `_patch_retiled_load_indexes`. |
| `torch_spyre/_inductor/scratchpad/passes.py:40` | `_NameSwapHandler` | Scratchpad rewrite of loop-hack input buffer names. |
| `torch_spyre/_inductor/scratchpad/graph_editor.py:281` | `_NameSwapHandler` (nested class, duplicate of the passes.py one) | Scratchpad loop-hack. |
| `torch_spyre/_inductor/split_multi_ops.py:63` | `_SplitOpsHandler` (installed at line 674) | Redirect intermediate op loads / rewrite constants during multi-op split. |
| `torch_spyre/_inductor/split_multi_ops.py:92` | `_IntermediateOpHandler` (installed at line 675, wraps `_SplitOpsHandler`) | Materialize intermediate op results from the deque. |

### The nesting risk

The install pattern is *stacking*: `new_inner_fn` captures the
*current* `op.data.inner_fn` as `orig_inner`, and the next install on
the same op captures the previous `new_inner_fn` as its own
`orig_inner`.  Concretely, `_patch_consumer_to_read_copy` runs Pass 1
(`wsr/coarse_tile.py:3140`), and if the same consumer is later reached
via `_patch_consumers` in Pass 3
(`wsr/coarse_tile.py:3892`), the second install wraps the first.  Each
call adds one `V.set_ops_handler(...)` context to the trace, and every
retrace re-enters both.

The install sites are aware of the pattern in principle — the
CLAUDE.md-style docstring at `wsr/coarse_tile.py:2487` reads:

> See NameSwapHandler in insert_restickify.py — same pattern (CLAUDE.md
> "Compiler Pass Conventions": wrap inner_fn via a WrapperHandler,
> never reconstruct it from index expressions).

— but no site *checks* whether it is already inside a wrapped
`inner_fn` before adding another layer, and no site combines maps into
a single handler when multiple maps target the same op.

Failure modes observed in-tree that this admits (unpatched, per grep):

- **Redundant retrace cost.**  Every retrace of the inner_fn (e.g.
  from `get_default_sizes_body` or from the scheduler's fusion
  check) walks through every stacked handler.  For a hot op patched
  N times, retrace cost is O(N).

- **Name-map order dependence.**  Two `NameSwapHandler` layers with
  maps `{A: B}` and `{B: C}` compose as expected (`A → B → C`), but
  two layers with maps `{A: B}` and `{A: C}` — where the outer sees
  the inner's already-swapped name — depend entirely on which layer
  is outer.  The `_patch_consumer_to_read_copy` +
  `_patch_consumers` sequence is not obviously safe against this
  case; it appears not to be triggered in practice only because the
  two passes target different buffer names.

- **Stride-rescale double-apply.**  `_NameAndIndexSwapHandler`
  (`wsr/coarse_tile.py:4197`) rewrites the index while the name is
  still the *old* name.  If nested inside an outer name-swap that
  already renamed the same buffer, the index rewrite runs on the
  wrong name and silently misses.  The docstring at
  `wsr/coarse_tile.py:4269–4274` acknowledges the double-application
  hazard for a different reason (ops inserted by later passes are
  skipped by `_should_patch_retiled_load_indexes` to avoid
  double-applying the retile), which is evidence the author has
  thought about the layering — but the check is per-op, not
  per-handler-stack.

### Proposed contract (do not implement)

The five WrapperHandler kinds in coarse_tile.py plus the shared
NameSwapHandler compose into one small algebra:

- **Name swap** (`old → new`) with optional
- **Stride rescale** (`full_strides → tile_strides`, per-swapped-name), or
- **Retile-index rewrite** (`old_stride → new_stride`, per-name).

A single combined handler could carry three per-name dicts and apply
them in a fixed order on every `load(name, index)`:
1. If `name` has a retile-index rewrite entry, apply it first (before
   the name changes).
2. If `name` has a stride-rescale entry, apply it.
3. If `name` has a name-swap entry, swap the name last.

Every install site would merge its map into the op's single existing
combined handler (constructing one if none exists) rather than
stacking a new `V.set_ops_handler(...)` layer.  This makes retrace
cost O(1) in the number of installs and eliminates the layer-order
question entirely.  Deferred: this contract is descriptive; the
combined-handler refactor is out of scope here.

## 5. Object identity as a long-lived key

The critique this contract is grounded in: `op.operation_name` is
minted as `op{len(self.operations)}` at the moment
`register_operation` runs (`torch/_inductor/graph.py:942`), and
torch-spyre physically removes operations from
`graph.operations` in at least these sites:

| File:Line | Removed op | Removing pass |
| --- | --- | --- |
| `insert_restickify.py:220` | `restick_buff` (repositioning, then re-insert) | insert_restickify |
| `insert_restickify.py:596/599` | `buf_tmp`, `buf_copyback` (repositioning) | insert_post_mutation_restickify |
| `enforce_indirect_access_layout.py:385/397` | `buf_tmp`, `buf_copyback` (repositioning) | enforce_indirect_access_layout |
| `propagate_layouts.py:1605` | `op` (permanent removal) | propagate_layouts |
| `dedup_constants.py:97` | `dup` (permanent — duplicate constant) | dedup_and_promote_constants |
| `deadcode_elimination.py:97` | dead ops (permanent) | deadcode_elimination |
| `split_multi_ops.py:481, 576` | `buf`, `new_buf` (repositioning) | split_multi_ops |
| `padding.py:297` | `new_op` | padding |
| `scratchpad/graph_editor.py:226` | `new_com_buf` (repositioning) | scratchpad clone |
| `wsr/coarse_tile.py:2295` | `full_buf` (repositioning) | coarse_tile `_allocate_full_buffer` |
| `wsr/coarse_tile.py:3758` | `scalar_op` (repositioning to before fill_buf) | coarse_tile `_insert_fill_op` |

The **permanent** removals (dead-code, dedup, propagate_layouts) never
call `register_operation` again with a corrected suffix — the removed
positional slot is never reclaimed.  After a permanent removal, the
next call to `register_operation` still uses
`op{len(self.operations)}`; if the removed op had suffix `N` and there
are still `N+1` operations after the removal, the *next* registered op
gets suffix `N+1`, but if the removal happened before any further
registration then `len(operations)` decreased by 1 and the *next*
registered op gets suffix `N`, colliding textually with the just-
removed name.

However — `dedup_constants.py:100` explicitly does
`V.graph.name_to_op.pop(op_name, None)` on removal, so the mapping is
cleared; and `register_operation`'s `assert op.operation_name is None`
check catches only *re-registration of the same object*, not name
collisions between distinct objects.  The name-collision hazard is
therefore latent: no in-tree code registers an op after a permanent
removal on the same graph in a way that would actually collide, but
the invariant that would prevent it (never reuse a name suffix within
one GraphLowering) is not enforced.

### Call sites that key off `operation_name`

Classified by intent:

**Operation identity (dedup within an operations list — correct use):**

- `wsr/coarse_tile.py:1216` — `_validate_contiguous` maps names to
  positions to check ops form a contiguous slice.  Correct because
  the map is built from `enumerate(operations)` at the same instant
  it is consumed (`wsr/coarse_tile.py:1556`).
- `wsr/coarse_tile.py:1556` — `op_to_position = {op.get_operation_name(): i for i, op in enumerate(operations)}` in `_coarse_tile_common`.  Local dict; short-lived.
- `wsr/coarse_tile.py:3214` — `op_position = {op.get_operation_name(): i for i, op in enumerate(operations)}` in `_plan_read_copies`.  Local dict; short-lived.
- `wsr/coarse_tile.py:3250` — sorts `op_deps` by `op_position[pair[0].get_operation_name()]` — consumes the previous line's dict.
- `wsr/coarse_tile.py:3287–3290` — records `insert_before_op_name` /
  `sizing_op_name` / `consumer_op_names` in a `ReadCopyEntry` for later
  execution.  This is the one identity-through-time use: the plan is
  built here and consumed later inside the same `_coarse_tile_common`
  invocation.  Safe because no op is removed or renamed between plan
  and apply, but the invariant is undocumented.
- `wsr/coarse_tile.py:3317` — `op.get_operation_name(): op` — a
  name-to-op dict rebuilt locally right before use.
- `wsr/coarse_tile.py:4245–4247` — `_replace_group_op` uses
  `old_name = old_op.get_operation_name()` and matches either by
  object identity (`is`) or by name.  The `or` clause is defensive:
  a rebuild via `replace_computed_buffer_body` would produce a new
  object with the same name, and this matcher catches that.
- `deadcode_elimination.py:40, 48, 89` — `live_ops` is a set of
  operation names computed and consumed within one pass invocation.

**Buffer identity, not operation identity:**

- `spyre_kernel.py:679, 908` — `ir_node.get_operation_name()` is used
  as a *label* for the current codegen node, not as a lookup key.
  Debug-adjacent.

**Debug identity:**

- `dump_cost_model.py:68, 577` — cost-model dump.
- `pass_utils.py:1965` — pretty-print header.
- `wsr/propagate_named_dims.py:352, 365, 428, 458, 673` — hint logs.
- `propagate_layouts.py:1765, 1775` — log lines.
- `wsr/coarse_tile.py:322, 1449` — debug logs.

**Cross-object identity (the one dangerous case):**

- `dedup_constants.py:90` — `op_name = dup.get_operation_name()`,
  used at line 100 as `V.graph.name_to_op.pop(op_name, None)`.  This
  is the only site that reads `name_to_op` by an operation_name it
  captured earlier.  Correct because it captures immediately before
  the pop, but demonstrates the pattern is in-use.

**No site reads `V.graph.name_to_op` after a rebuild.**  This is the
key survivability observation: even though §3.9 flagged that a
rebuild leaves `name_to_op` pointing at the stale object, in-tree the
only reader (`dedup_constants`) never consults `name_to_op` for a
rebuilt op — it only pops entries it just registered.  The
`name_to_op`-stale-after-rebuild bug is latent but unreached today.

### Fragility summary

- `operation_name` is safe as a **short-lived, same-invocation** dict
  key.  Every in-tree use falls into this category.
- `operation_name` would be unsafe as a **long-lived** key across
  passes that remove or rebuild ops:
  - Removal without name-suffix reservation could produce a collision
    (unreached today).
  - Rebuild without rewiring `name_to_op` leaves the mapping stale
    (unreached today).
- Object identity (`is`) survives repositioning-via-remove-and-insert
  (§5's repositioning sites all keep the same object) but *not*
  rebuild (§2c, §2d install fresh objects).  `_replace_group_op` at
  `wsr/coarse_tile.py:4245–4247` correctly handles both.

## 6. Open questions

- **Q1.**  Is there any harness-external caller (a downstream vLLM
  hook, an out-of-tree pass) that reads `V.graph.name_to_op` after a
  `replace_computed_buffer_body` rebuild?  If so, the stale-mapping
  bug in §3.9 is reachable; if not, the fix is a two-line change
  inside `replace_computed_buffer_body`.  *Not resolvable from the
  pinned scan alone; downstream integrations would need to be
  audited.* **OPEN QUESTION**.

- **Q2.**  Does `register_operation` ever run after
  `deadcode_elimination` or `dedup_constants` on the same
  `GraphLowering` object?  If yes, the suffix-collision hazard in §5
  is live.  Grepping the pinned scan for `register_operation` calls
  in passes that run *after* these two would answer this — deferred
  to the reentrancy audit (Phase 4). **OPEN QUESTION**.

- **Q3.**  The five WrapperHandler kinds in `wsr/coarse_tile.py` plus
  the two shared ones in `insert_restickify.py` / `dedup_constants.py`
  never install more than one handler stack per op in practice — is
  there a static assertion or test that would catch it if a future
  pass did?  Grep finds neither.  If the refactor in §4 is deferred,
  a minimum-viable check would be `assert not
  hasattr(op.data, "_ts_wrapped")` at each install site.
  **OPEN QUESTION** — no such guard exists today.

- **Q4.**  `insert_restickify.py:242` reconstructs the consumer
  ComputedBuffer specifically to invalidate
  `get_default_sizes_body`'s instance-keyed cache.  Would `object.__setattr__(op, "data", new_data)` + `clear_cache(op)` do the same job with one fewer allocation, or is there some other pass keyed off object identity that requires the fresh object?  §5's audit turned up no reader keyed off `ComputedBuffer` object identity across passes, but the answer depends on how the scheduler and codegen internals treat identity.  **OPEN QUESTION**.

- **Q5.**  The synthetic coarse-tile helpers (§2b) never call
  `register_operation`, so `name_to_op[copy_name]` is always missing.
  Any pass that later grep-walks `name_to_op` for an op it expected to
  see (rather than walking `operations` directly) would miss the
  synthetic copies.  No in-tree reader exhibits this pattern today.
  Confirm downstream (vLLM connector, scheduling) does not either.
  **OPEN QUESTION**.
