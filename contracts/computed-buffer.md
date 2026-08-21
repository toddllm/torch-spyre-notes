---
- **id:** computed-buffer
- **category:** contracts
- **created:** 2026-08-20
- **revision_manifest:** 2026-08-21 rewrite: corrected the frozen-dataclass premise (ComputedBuffer is `frozen=False` at v2.13); reframed around why passes still reconstruct even though `.data` can be reassigned; recounted synthetic-construction and `name_to_buffer` sites from grep output; replaced `/tmp/ts-pinned-scan/...` paths with `torch-spyre@fea0c4b:...` citations and used the pinned pytorch permalink for upstream.
- **confidence:** medium — direct-construction and callsite counts are grep-verified; the "why reconstruct?" enumeration mixes observable causes (memo invalidation, layout re-flow) with cause-hypotheses that would need a test to falsify.
- **status:** revised
---

# ComputedBuffer — mutability, why passes still rebuild, identity rules, and handler nesting

Torch-spyre citations use the form `torch-spyre@fea0c4b:<path>:<line>` and resolve against SHA fea0c4be901e1383b1f700dbad8887128b0fcb27 on `github.com/torch-spyre/torch-spyre` (private). PyTorch citations link to the pinned v2.13.0 baseline at cf30153c4c131c8164ee7798e5022d810682e2cb on the public pytorch/pytorch repo.

## 1. What `ComputedBuffer` is, and why the mutability matters

`ComputedBuffer` is a `Buffer` + `Operation` composite whose payload is a `Loops` subclass (`Pointwise` or `Reduction`). At the pinned v2.13.0 baseline it is declared:

```python
# torch/_inductor/ir.py:5201-5202
@ir_dataclass(frozen=False)
class ComputedBuffer(OperationBuffer):
```

Full URL: https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/ir.py#L5201

The `frozen=False` argument is load-bearing: `ir_dataclass` defaults to `frozen=True` (that is why `Operation` at `ir.py:855` must also spell out `@ir_dataclass(frozen=False)` — pytorch@cf30153:torch/_inductor/ir.py:855). ComputedBuffer, Loops (`ir.py:4975`), Reduction (`ir.py:5112`), and the surrounding Pointwise/Scatter family are all mutable-decl at v2.13. **`buf.data`, `buf.layout`, `buf.name`, `buf.operation_name` can all be reassigned in place with a normal attribute write.**

Torch-spyre proves this in-tree:

```python
# torch-spyre@fea0c4b:torch_spyre/_inductor/scratchpad/graph_editor.py:277
old_loop.data = new_loop
```

`_replace_loop_input` swaps the entire `Pointwise`/`Reduction` payload of an existing `ComputedBuffer`-shaped `Operation` with a plain attribute write, then invalidates a torch-spyre memo (`invalidate_op_read_writes(old_loop)` at line 279).

A single comment inside `replace_computed_buffer_body` still calls ComputedBuffer "a frozen dataclass" (torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1320); that comment is stale — the operation the helper performs would work as an in-place mutation of `op.data` at v2.13.

Given that mutation is allowed, the interesting question is **not** "how do we work around a frozen dataclass" but:

> Given ComputedBuffer is mutable at v2.13, why do five torch-spyre callsites reconstruct a fresh ComputedBuffer instead of mutating in place?

The remainder of this contract answers that, then documents the invariants those reconstruction sites happen to preserve.

## 2. Why some passes reconstruct instead of mutating

Five in-tree callers go through `replace_computed_buffer_body`, plus one `ComputedBuffer(...)` construction inline in `insert_restickify_on_node_inputs`. Grouped by the actual reason each one rebuilds rather than mutates:

### 2.1 Memo freshness (the strongest reason)

Two caches key on the ComputedBuffer instance:

- **Upstream `ComputedBuffer.get_default_sizes_body`** — an instance-keyed cache (`clear_cache(buf)` is a real method on it, as used at `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1345` and `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:264`). Cache entries are keyed by the object's identity. Constructing a new object automatically has no entries; the explicit `clear_cache(new_buf)` on the fresh object is defensive.
- **Torch-spyre's `op_read_writes` memo** — stored under the private key `_ts_cached_read_writes` on `op.__dict__` (`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:112-116`). Upstream `ComputedBuffer.get_read_writes` (pytorch@cf30153:torch/_inductor/ir.py:5281) has no cache — but the torch-spyre helper does, and the memo lives on the instance.

The `insert_restickify` docstring is explicit about the motive:

```python
# torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:240-241
# Reconstruct ComputedBuffer as a fresh object so the instance-keyed cache
# on get_default_sizes_body can be cleanly invalidated below.
```

The in-place alternative — `object.__setattr__(op.data, "inner_fn", new_inner_fn); clear_cache(op); invalidate_op_read_writes(op)` — is functionally equivalent at v2.13 but adds one non-obvious risk: any *other* memo installed elsewhere (a downstream vLLM connector cache, an out-of-tree pass) that keys on the same instance survives the mutation invisibly. Reconstructing the object invalidates every instance-keyed memo at once, whether or not the current pass knows about them. This is a real correctness advantage of reconstruction over mutation, but it is not free — see §4.

`_replace_loop_input` (`graph_editor.py:277`) is the counter-example: it *does* mutate in place, and it manually calls `invalidate_op_read_writes(old_loop)` on the next line. It works because scratchpad's set of memos is small and locally auditable; that assumption does not extend to arbitrary passes.

### 2.2 Layout re-flow

`padding.py:154` and `split_multi_ops.py:682` both feed a `new_data` whose `Loops.ranges` differ from `op.data.ranges`. When `op.layout` is a `FlexibleLayout`, Inductor's stride resolution runs off `data.ranges`; a fresh `ComputedBuffer` gets a fresh FlexibleLayout resolution pass (or reuses `op.layout` verbatim, which is the pattern actually observed in these two callsites — `new_buf.layout = op.layout`). Neither site relies on the "fresh layout" pathway today, but the option is preserved by reconstruction and would require an extra manual `finalize_layouts`-style call under mutation.

### 2.3 Object-identity as a break-in-topology signal

Every downstream pass that maintains a `id(op) -> metadata` dict (there are none in-tree today; see §5 for the audit) would see mutation as a no-op and reconstruction as a real replacement. This is speculative — no in-tree consumer keys off `id(op)` — but the pattern preserves that option.

### 2.4 What is **not** a reason at v2.13

- **"ComputedBuffer is frozen."** False at v2.13, per §1.
- **"`.data` can't be reassigned."** False — see `graph_editor.py:277`.
- **"Rebuilding forces users list rewiring."** No — `V.graph.name_to_users` is keyed by *buffer name*, not object identity, and the buffer name is preserved verbatim across every rebuild (§3, invariant 1).

The honest summary: **memo invalidation is the load-bearing reason.** Layout re-flow is a latent option. Identity-as-signal is speculative. The frozen-dataclass framing was wrong.

## 3. Synthetic construction sites — verified count

Grep for `ComputedBuffer(` (constructor call, excluding one docstring reference at `wsr/coarse_tile.py:217`) returns **9 sites** in `torch-spyre@fea0c4b:torch_spyre/_inductor/`:

| # | File:Line | Site | Registers via `register_operation`? | Purpose |
| --- | --- | --- | --- | --- |
| 1 | `torch-spyre@fea0c4b:torch_spyre/_inductor/lowering.py:1286` | `_build_mutation_lowering` | yes | `spyre.copy_forced` / `spyre.opaque_copy_` lowering; only site on the normal lowering path. |
| 2 | `torch-spyre@fea0c4b:torch_spyre/_inductor/scratchpad/graph_editor.py:172` | `_clone_buffer_op` | yes (line 182) | Clones a buffer to break a false anti-dependency in scratchpad planning; then physically repositions in `operations` at line 226. |
| 3 | `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:242` | `insert_restickify_on_node_inputs` | no — hand-assigns `operation_name` at line 251 | Rebuild-in-place to invalidate `get_default_sizes_body` cache. |
| 4 | `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1333` | `replace_computed_buffer_body` (shared helper) | no — hand-assigns at line 1342 | Called by 5 downstream sites (see below). |
| 5 | `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:2330` | `copy_buf` (write copy-out for a tiled op) | no — hand-assigns at line 2336 | Splices into `operations` without `register_operation`. |
| 6 | `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:2921` | `copy_buf` (read copy-in shared across tiled consumers) | no — hand-assigns at line 2923 | Same pattern. |
| 7 | `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3421` | `combine_buf` (reduction combine op) | no — hand-assigns at line 3427 | Same pattern. |
| 8 | `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3520` | `copy_buf` (reduction accum_tile → accum_full) | no — hand-assigns at line 3526 | Same pattern. |
| 9 | `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3738` | `fill_buf` (fill identity for reduction accum) | no — hand-assigns at line 3744 | Same pattern. |

**Count: 9 constructor calls, 5 callers of `replace_computed_buffer_body`.**

Callers of `replace_computed_buffer_body` (grep-verified — 5, not the previously-claimed 3):

1. `torch-spyre@fea0c4b:torch_spyre/_inductor/padding.py:154`
2. `torch-spyre@fea0c4b:torch_spyre/_inductor/split_multi_ops.py:682`
3. `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3147`
4. `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3911`
5. `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4291`

`name_to_buffer[...] = ...` sites that install a synthetic ComputedBuffer (grep-verified — 8 in `wsr/coarse_tile.py`, of which 5 install a synthetic coarse_tile buffer and 3 install a `replace_computed_buffer_body` result):

| Line | Kind |
| --- | --- |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:2457` | Synthetic `copy_buf` (from site #5) |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3062` | Synthetic `copy_buf` (from site #6) |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3154` | `new_op` from `replace_computed_buffer_body` (`_patch_consumer_to_read_copy`) |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3471` | Synthetic `combine_buf` (from site #7) |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3558` | Synthetic `copy_buf` (from site #8) |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3754` | Synthetic `fill_buf` (from site #9) |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3918` | `new_consumer` from `_patch_consumers` |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4299` | `new_op` from `_patch_retiled_load_indexes` |

No coarse-tile helper ever calls `V.graph.register_operation` or writes into `V.graph.name_to_op`. A synthetic op inserted by these helpers is visible via `graph.operations.index(op)` but *not* via `name_to_op`.

## 4. Replacement identity rules

Reading the six rebuild sites together (five callers of `replace_computed_buffer_body` plus the inline rebuild in `insert_restickify_on_node_inputs`), the invariants an in-place `ComputedBuffer` replacement must satisfy:

1. **Buffer name preserved.** `new_buf.name = op.get_name()` — consumers keyed off the buffer name never resolve to a missing buffer. Cited: `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1334`, `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:243`.

2. **`operation_name` preserved.** `new_buf.operation_name = op.operation_name` — otherwise the rebuilt op leaves `operation_name = None` and `get_operation_name()` asserts. Cited: `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1342`, `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:251`.

3. **Layout preserved.** `new_buf.layout = op.layout` — layout decisions from earlier passes (`finalize_layouts`, stickification) must not silently regress. Cited: `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1335`, `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:244`.

4. **Position in `operations` preserved.** The replacement uses `operations[op_idx] = new_buf` (`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1348`) or `operations[op_index] = new_consumer_buffer` (`torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:260`), not `remove()` + `append()` — the topological position stays fixed so downstream `op_position` dicts stay valid until the next rebuild.

5. **`origins` / `origin_node` preserved.** Done inside `preserve_provenance`. If a rebuild skips this, FX provenance is lost and downstream passes that key off `origins` misbehave.

6. **`loop_info` and other torch-spyre bare-attribute metadata preserved.** Done inside `copy_op_metadata` (loop_info.py). Coarse-tile passes attach `loop_info` as a bare attribute (via `object.__setattr__` where needed) and every rebuild must copy it forward.

7. **`get_default_sizes_body` cache cleared.** Cited: `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1345`, `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:264`. Instance-keyed; a new object automatically has no entries, but the explicit call is defensive.

8. **Users list is *not* touched.** Neither `replace_computed_buffer_body` nor `insert_restickify_on_node_inputs` writes into `V.graph.name_to_users`. Safe iff invariant 1 holds — `name_to_users` is keyed by buffer name, not by object identity. A pass that changed the buffer name during rebuild would silently orphan the users list; no in-tree pass does this today, but the invariant is undocumented.

9. **`name_to_op` is NOT rewired.** Neither rewrite site updates `V.graph.name_to_op[new_op.operation_name] = new_op`. After a rebuild, `name_to_op[operation_name]` still points at the *stale* `ComputedBuffer` object, while `operations[op_idx]` points at the fresh one. §6 surveys whether any in-tree consumer reads `name_to_op` on this axis (answer: no in-tree reader; latent hazard for out-of-tree callers).

## 5. WrapperHandler nesting

Torch-spyre installs `WrapperHandler` subclasses that patch an inner_fn's `ops.load(name, index)` calls. Every install site follows the same recipe: capture `orig_inner = op.data.inner_fn`, define a `new_inner_fn` closure that enters a `V.set_ops_handler(...)` block and calls `orig_inner(*args)`, then `object.__setattr__(op.data, "inner_fn", new_inner_fn)` and rebuild the `ComputedBuffer` via `replace_computed_buffer_body`. Instances in-tree:

| File:Line | Handler class | Purpose |
| --- | --- | --- |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:76` | `NameSwapHandler` | Rename input buffers after restickify insertion. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/dedup_constants.py:45` | `NameSwapHandler` (reused from insert_restickify) | Redirect a consumer of a duplicate constant to the canonical constant. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:2487` | `_NameSwapHandler` (locally-defined, index-rescaling variant) | Redirect consumer loads from full_buf to tile-local copy_buf, rescaling stride coefficients. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3143` | `_NameSwapHandler` (as above) | Same, installed by `_patch_consumer_to_read_copy`. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3906` | `NameSwapHandler` (imported from insert_restickify) | Redirect outside consumers from tiled-op scratch to full-sized output, in `_patch_consumers`. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4180` | `_RetileLoadIndexHandler` | Rewrite retiled load indexes for consumers of retiled buffers. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4197` | `_NameAndIndexSwapHandler` | Combines name-swap + retile-index-rewrite for `_patch_consumers` when strides differ. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4287` | `_RetileLoadIndexHandler` (as above) | Installed by `_patch_retiled_load_indexes`. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/scratchpad/passes.py:40` | `_NameSwapHandler` | Scratchpad rewrite of loop-hack input buffer names. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/scratchpad/graph_editor.py:281` | `_NameSwapHandler` (nested class, duplicate of the passes.py one) | Scratchpad loop-hack. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/split_multi_ops.py:63` | `_SplitOpsHandler` (installed at line 674) | Redirect intermediate op loads / rewrite constants during multi-op split. |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/split_multi_ops.py:92` | `_IntermediateOpHandler` (installed at line 675, wraps `_SplitOpsHandler`) | Materialize intermediate op results from the deque. |

### The nesting risk

The install pattern is *stacking*: `new_inner_fn` captures the *current* `op.data.inner_fn` as `orig_inner`, and the next install on the same op captures the previous `new_inner_fn` as its own `orig_inner`. Concretely, `_patch_consumer_to_read_copy` (`wsr/coarse_tile.py:3140`) runs Pass 1, and if the same consumer is later reached via `_patch_consumers` in Pass 3 (`wsr/coarse_tile.py:3892`), the second install wraps the first. Each call adds one `V.set_ops_handler(...)` context to the trace, and every retrace re-enters both.

No install site *checks* whether it is already inside a wrapped `inner_fn` before adding another layer, and no site combines maps into a single handler when multiple maps target the same op.

Failure modes observed in-tree (unpatched, per grep):

- **Redundant retrace cost.** Every retrace of the inner_fn (e.g. from `get_default_sizes_body` or from the scheduler's fusion check) walks through every stacked handler. For a hot op patched N times, retrace cost is O(N).
- **Name-map order dependence.** Two `NameSwapHandler` layers with maps `{A: B}` and `{B: C}` compose as expected (`A → B → C`), but two layers with maps `{A: B}` and `{A: C}` depend entirely on which layer is outer. Not obviously safe against `_patch_consumer_to_read_copy` + `_patch_consumers` targeting the same buffer; appears not to be triggered in practice only because the two passes target different buffer names.
- **Stride-rescale double-apply.** `_NameAndIndexSwapHandler` (`torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4197`) rewrites the index while the name is still the *old* name. If nested inside an outer name-swap that already renamed the same buffer, the index rewrite runs on the wrong name and silently misses.

### Proposed contract (do not implement)

The handler kinds compose into one small algebra: **Name swap** (`old → new`) with optional **Stride rescale** (per-swapped-name), or **Retile-index rewrite** (per-name). A single combined handler could carry three per-name dicts and apply them in a fixed order on every `load(name, index)`. Every install site would merge its map into the op's single existing combined handler rather than stacking a new `V.set_ops_handler(...)` layer. This makes retrace cost O(1) in the number of installs and eliminates the layer-order question entirely. Deferred: this contract is descriptive; the combined-handler refactor is out of scope.

## 6. Object identity as a long-lived key

`op.operation_name` is minted as `op{len(self.operations)}` at the moment `register_operation` runs (pytorch@cf30153:torch/_inductor/graph.py:942 — https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/graph.py#L942). Torch-spyre physically removes operations from `graph.operations` in these sites:

| File:Line | Removed op | Removing pass |
| --- | --- | --- |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:220` | `restick_buff` (repositioning, then re-insert) | insert_restickify |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:596` | `buf_tmp` (repositioning) | insert_post_mutation_restickify |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:599` | `buf_copyback` (repositioning) | insert_post_mutation_restickify |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/enforce_indirect_access_layout.py:385` | `buf_tmp` (repositioning) | enforce_indirect_access_layout |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/enforce_indirect_access_layout.py:397` | `buf_copyback` (repositioning) | enforce_indirect_access_layout |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/propagate_layouts.py:1605` | `op` (permanent removal) | propagate_layouts |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/dedup_constants.py:97` | `dup` (permanent — duplicate constant) | dedup_and_promote_constants |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/deadcode_elimination.py:97` | dead ops (permanent) | deadcode_elimination |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/split_multi_ops.py:481` | `buf` (repositioning) | split_multi_ops |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/split_multi_ops.py:576` | `new_buf` (repositioning) | split_multi_ops |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/padding.py:297` | `new_op` | padding |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/scratchpad/graph_editor.py:226` | `new_com_buf` (repositioning) | scratchpad clone |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:2295` | `full_buf` (repositioning) | coarse_tile `_allocate_full_buffer` |
| `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3758` | `scalar_op` (repositioning to before fill_buf) | coarse_tile `_insert_fill_op` |

The **permanent** removals (dead-code, dedup, propagate_layouts) never call `register_operation` again with a corrected suffix. `register_operation`'s `assert op.operation_name is None` check catches only *re-registration of the same object*, not name collisions between distinct objects. The name-collision hazard is therefore latent: no in-tree code registers an op after a permanent removal on the same graph in a way that would actually collide, but the invariant that would prevent it (never reuse a name suffix within one GraphLowering) is not enforced.

### Call sites that key off `operation_name`

**Operation identity (dedup within an operations list — correct use):**
- `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:1216, 1556, 3214, 3250, 3317` — local dicts built and consumed within the same invocation.
- `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3287-3290` — records op names in a `ReadCopyEntry` for later execution *within the same pass*; safe because no op is removed or renamed between plan and apply, but the invariant is undocumented.
- `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4245-4247` — `_replace_group_op` matches by object identity (`is`) or by name; the `or` clause is defensive against rebuild.
- `torch-spyre@fea0c4b:torch_spyre/_inductor/deadcode_elimination.py:40, 48, 89` — `live_ops` set is computed and consumed within one pass invocation.

**Cross-object identity (the one dangerous case):**
- `torch-spyre@fea0c4b:torch_spyre/_inductor/dedup_constants.py:90, 100` — captures `dup.get_operation_name()`, then `V.graph.name_to_op.pop(op_name, None)`. Correct because it pops immediately, but demonstrates the pattern is in-use.

**Debug identity only:**
- `torch-spyre@fea0c4b:torch_spyre/_inductor/spyre_kernel.py:679, 908` (labels)
- `torch-spyre@fea0c4b:torch_spyre/_inductor/dump_cost_model.py:68, 577`
- `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1965`
- `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/propagate_named_dims.py:352, 365, 428, 458, 673`
- `torch-spyre@fea0c4b:torch_spyre/_inductor/propagate_layouts.py:1765, 1775`
- `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:322, 1449`

**No in-tree site reads `V.graph.name_to_op` after a rebuild.** Even though invariant 9 flagged that a rebuild leaves `name_to_op` stale, the only in-tree reader (`dedup_constants`) never consults `name_to_op` for a rebuilt op. The `name_to_op`-stale-after-rebuild bug is latent but unreached today.

### Fragility summary

- `operation_name` is safe as a **short-lived, same-invocation** dict key. Every in-tree use falls into this category.
- `operation_name` would be unsafe as a **long-lived** key across passes that remove or rebuild ops:
  - Removal without name-suffix reservation could produce a collision (unreached today).
  - Rebuild without rewiring `name_to_op` leaves the mapping stale (unreached today).
- Object identity (`is`) survives repositioning-via-remove-and-insert but *not* rebuild. `_replace_group_op` at `torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4245-4247` correctly handles both.

## 7. Open questions

- **Q1.** Is there any harness-external caller (a downstream vLLM hook, an out-of-tree pass) that reads `V.graph.name_to_op` after a `replace_computed_buffer_body` rebuild? If so, the stale-mapping bug in invariant 9 is reachable; if not, the fix is a two-line change inside `replace_computed_buffer_body`. *Not resolvable from the pinned scan alone.* **OPEN**.

- **Q2.** Does `register_operation` ever run after `deadcode_elimination` or `dedup_constants` on the same `GraphLowering` object? If yes, the suffix-collision hazard in §6 is live. **OPEN — deferred to reentrancy audit**.

- **Q3.** No install site guards against multiply-wrapped `inner_fn`. A minimum-viable check would be `assert not hasattr(op.data, "_ts_wrapped")` at each install site. **OPEN — no such guard exists today**.

- **Q4.** Given `ComputedBuffer` is `frozen=False` at v2.13, would `object.__setattr__(op, "data", new_data); clear_cache(op); invalidate_op_read_writes(op)` replace `replace_computed_buffer_body` with one fewer allocation? The audit in §6 turned up no in-tree reader keyed off `ComputedBuffer` object identity across passes. The remaining question is whether any downstream (out-of-tree vLLM connector, scheduler, codegen internals) treats object identity as a stability signal. **OPEN** — the reconstruction-first policy is defensible on memo-safety grounds even if mutation works locally.

- **Q5.** The synthetic coarse-tile helpers (§3, sites 5-9) never call `register_operation`, so `name_to_op[copy_name]` is always missing. Any pass that grep-walks `name_to_op` for an op it expected to see (rather than walking `operations` directly) would miss the synthetic copies. No in-tree reader exhibits this pattern today. Confirm downstream does not either. **OPEN**.

- **Q6.** The stale docstring at `torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1320` describes ComputedBuffer as "a frozen dataclass" — false at v2.13. If the reconstruction-vs-mutation decision is being reargued (Q4), that docstring should be corrected in the same change so future readers do not inherit the wrong premise. **OPEN — trivial fix, not landed here**.
