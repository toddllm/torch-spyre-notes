# Pre-scheduling pass contract matrix

Torch-Spyre defines several classes derived from Inductor's
`CustomGraphPass` / `CustomSchedulerPass` machinery.  The one that
carries the substantive pre-scheduler work is `CustomPreSchedulingPasses`
(`torch_spyre/_inductor/passes.py:415`).  Its pipeline is a hard-coded
ordered list of callables built in `__init__` (`passes.py:444`).  A
sibling pair of pipelines run at other extension points:
`CustomPreGradPasses`, `CustomPrePasses`, `CustomPostPasses` (all
`_SpyreGraphPassPipeline`), `CustomPreFusionPasses`, and
`CustomPostFusionPasses` (both `_SpyreNodePassPipeline`).

This matrix enumerates every pass function referenced from one of those
pipeline `__init__`s and records the contract it operates under.  The
Requires / Reads / Mutates / Creates-or-removes-ops columns are filled
from each pass's docstring and body; `?` indicates a value that cannot
be settled by static inspection alone and is captured in the open
questions at the bottom.

Pinned commit: `fea0c4be901e1383b1f700dbad8887128b0fcb27`. Torch-spyre
citations use the form `torch-spyre@fea0c4b:<path>:<line>` and resolve
against SHA fea0c4be901e1383b1f700dbad8887128b0fcb27 on
`github.com/torch-spyre/torch-spyre` (private).

## CustomPreGradPasses  (`passes.py:211`)

Empty pipeline in the pinned tree -- reserved extension point.  Nothing
to enumerate.

## CustomPrePasses  (`passes.py:221`)

Runs on the post-grad FX graph, early.

| Pass                         | Requires                        | Reads                          | Mutates                                | Creates/removes ops              | Invalidates                        | Layout phase |
| ---------------------------- | ------------------------------- | ------------------------------ | -------------------------------------- | -------------------------------- | ---------------------------------- | ------------ |
| `collect_spyre_hints` (`propagate_hints.py:138`) | none                             | `graph.nodes`, `node.meta["custom"]`, hint globals | `node.meta["custom"]["spyre_hints"]` on FX nodes | none                             | (records hints only)              | FX pre-grad  |

## CustomPostPasses  (`passes.py:231`)

Runs on the post-grad FX graph, late.  Undoes upstream's addmm
re-fusion so the mm flow stays on the Spyre lowerings.

| Pass                         | Requires                        | Reads                          | Mutates                                | Creates/removes ops              | Invalidates                        | Layout phase |
| ---------------------------- | ------------------------------- | ------------------------------ | -------------------------------------- | -------------------------------- | ---------------------------------- | ------------ |
| `decompose_addmm` (`temp_passes.py`) | FX post-grad                    | FX nodes                       | Rewrites `aten.addmm` into `mm` + `add` + scalar muls | replaces `aten.addmm` nodes | breaks any pattern match that assumed addmm survives | FX post-grad |
| `mm_to_bmm_pass.apply` (`temp_passes.py`) | after `decompose_addmm`         | FX `mm` nodes                  | Rewrites `mm` into `bmm` with unit batch dim | replaces `mm` with `bmm`         | (none observed)                    | FX post-grad |
| `mark_direct_unit_bmm_pass` (`temp_passes.py`) | after `mm_to_bmm_pass`         | FX `bmm` nodes                 | Marks unit-batch bmm nodes with a hint | metadata only                    | (none)                             | FX post-grad |
| `bmm_unflatten_pass.apply` (`temp_passes.py`) | after mark pass                | FX `bmm` nodes                 | Unflattens `bmm(reshape(x,[B*M,K]), ...)` | rewires FX edges                 | invalidates reshape/view topology  | FX post-grad |

## CustomPreSchedulingPasses.passes  (`passes.py:444-485`)

Runs on `GraphLowering` operations immediately before the Scheduler is
constructed.  Operations are in topological order.  Each pass takes the
`GraphLowering` and mutates `graph.operations` in place.

| # | Pass | Requires | Reads | Mutates | Creates/removes ops | Invalidates | Layout phase |
| - | ---- | -------- | ----- | ------- | ------------------- | ----------- | ------------ |
| 1 | `deadcode_elimination` (`deadcode_elimination.py:71`) | topo-ordered operations | `graph.operations`, per-op `get_read_writes()`, `get_operation_name`, graph outputs | `graph.operations`, `graph.removed_buffers` | removes dead ops in-place | any dep list built earlier against removed ops | pre-stickify |
| 2 | `propagate_named_dims` (`wsr/propagate_named_dims.py:508`) | `name_tensor_dims()` annotations OR in-graph `named_dims` hints | `graph.operations`, `_named_tensor_dims`, hint metadata | stamps `op._dim_prop_info`, `op.named_dims`, `op.named_reduction_dims`; clears `_named_tensor_dims` / `_enabled` in `finally` | none | (populates metadata later passes read) | pre-stickify |
| 3 | `validate_named_dims` (`wsr/propagate_named_dims.py:536`) | after (2) ran; `op._dim_prop_info` present | per-op `_dim_prop_info`, `get_op_hints()` scopes | none (assertion pass) | none | asserts, raises on mismatch | pre-stickify |
| 4 | `assign_dim_hints` (`wsr/propagate_named_dims.py:685`) | after (2)                              | `_named_dims`, per-op `_dim_prop_info`, spyre_hint scopes on FX nodes | writes `op.dim_hints`; deletes `op._dim_prop_info`; calls `reset()` in `finally` | none | consumes `_dim_prop_info` (deletes it) | pre-stickify |
| 5 | `_maybe_reorder_unhinted_interlopers` (`passes.py:321`) -> `reorder_unhinted_interlopers` (`wsr/coarse_tile_hints.py:166`) | after (4); gated by `config.ignore_wsr_hints` | `graph.operations`, `op.dim_hints`, op types | reorders `graph.operations` in-place (`list.pop`/`insert`) | reorders only | invalidates any index into `graph.operations` captured earlier | pre-stickify |
| 6 | `_maybe_coarse_tile_hints` (`passes.py:333`) -> `hints_to_coarse_tile_groups` + `validate_coarse_tile_groups` + `coarse_tile_pre_stickify` (`wsr/coarse_tile.py:1456`) | after (5); gated by `config.ignore_wsr_hints` | `graph.operations`, `op.dim_hints` | inserts read copy-ins, reduction machinery, write copy-outs; stamps `op.loop_info.loop_group_id`, `op.loop_info.loop_count` | inserts new buffer/copy ops into `graph.operations` | invalidates any prior `op_order` snapshots | pre-stickify (this is the last WSR phase before stickification) |
| 7 | `split_multi_ops` (`split_multi_ops.py:774`) | `V.graph` set; pre-stickify | `graph.name_to_users`, FX env, `op.data.inner_fn` (via traced handler) | creates intermediate `ComputedBuffer`s and `SpyreConstantFallback` ops; patches original `inner_fn` via `_SplitOpsHandler` | inserts new ops; modifies existing `inner_fn` | invalidates any snapshot of `graph.operations`; can create `SpyreConstantFallback` that dedup will see later | pre-stickify |
| 8 | `propagate_spyre_tensor_layouts` (`propagate_layouts.py:1697`) | topo-ordered ops; `V.get_real_inputs()` available | `graph.graph_input_names`, `graph.graph_inputs`, per-op deps and dtypes | sets `op.layouts` (candidate STLs), `op.restick_cost_fn`, `input_buf.layout` on graph inputs; records `forced_mutation_alts` | none                                | (populates layout candidates the optimizer consumes) | stickification |
| 9 | `validate_ops` (`split_multi_ops.py:713`) | after (8); STLs on inputs | `op.get_read_writes()`, per-buffer `.layouts` (`SpyreTensorLayout`) | none (assertion pass) | none | raises on `ElementArrangement` mismatch | stickification |
| 10 | `optimize_restickify_locations` (`optimize_restickify.py:731`) | after (8); ops carry `layouts`+`restick_cost_fn` | `op.layouts`, `op.restick_cost_fn`, prior ops' `committed_stl` | writes `op.committed_stl` on every op (greedy or beam) | none | fixes each op's committed layout (`layouts` becomes vestigial) | stickification |
| 11 | `finalize_layouts` (`insert_restickify.py:288`) | after (10); each op has `committed_stl` | `op.committed_stl`, `graph.graph_inputs` `InputBuffer.committed_stl` | wraps `committed_stl` -> `FixedTiledLayout`, sets `op.layout`; clears `layouts` / `restick_cost_fn` / `committed_stl`; builds `graph.restickify_plan` | none | consumes and clears optimizer scratch attrs | stickification |
| 12 | `insert_restickify` (`insert_restickify.py:267`) | `graph.restickify_plan` populated by (11) | `graph.restickify_plan`, `graph.operations` | inserts restickify `ComputedBuffer` ops immediately before their consumers | inserts new ops (index-based `list.insert`) | invalidates any op-index snapshot | stickification |
| 13 | `enforce_indirect_access_layout` (`enforce_indirect_access_layout.py:585`) | after (12); every op has committed `FixedTiledLayout` | per-op indirect-access requirements (`_get_indirect_access_dim_order_requirements`), producer layout | either rewrites a producer's `SpyreTensorLayout` in place OR inserts a `spyre.restickify` copy node | may insert new copy ops           | can change producer layouts (single-consumer, non-mutation, non-graph-output only) | stickification |
| 14 | `insert_post_mutation_restickify` (`insert_restickify.py:494`) | after (12/13); graph inputs used as mutation targets | `graph.operations`, `graph.graph_inputs`, mutation ops (`MutationLayoutSHOULDREMOVE`) | inserts pre-mutation restickify + copy-back triple (steps 1..4 of the docstring) | inserts new ops | invalidates op-index snapshots | stickification |
| 15 | `insert_bmm_padding` (`padding.py:163`) | after stickify passes; `BATCH_MATMUL_OP` ops present | matmul input identification via `identify_matmul_inputs`, dtype tables | inserts padded `y` buffer immediately before each BMM; leaves `x` untouched | inserts new pad buffers | reduction_ranges unchanged; K widens at SDSC codegen time (invariant preserved) | stickification (post) |
| 16 | `dedup_and_promote_constants` (`dedup_constants.py:109`) | after (12-15); `SpyreConstantFallback` may exist | `graph.operations`, per-constant `(value, dtype, device)` key | rewires consumer `ComputedBuffer.inner_fn`s to canonical constant name; moves surviving constants to the head of `graph.operations`; adds duplicate output names to `graph.removed_buffers` | removes duplicate constants | any prior index into `graph.operations` (order changes); consumer inner_fns are patched by name | pre-scheduling (post-stickify) |
| 17 | `_maybe_coarse_tile_span_overflow` (`passes.py:355`) -> `span_overflow_groups` + `validate_coarse_tile_groups` + `coarse_tile_post_stickify` (`wsr/coarse_tile.py:1485`) | after (11); every op has `FixedTiledLayout.device_layout` | `graph.operations`, `op.loop_info.loop_group_id` (for offset), `dim_hint_assignments` | writes `op.dim_hints` from planning step; stamps `op.loop_info.loop_group_id`, `loop_count`; inserts write copy-outs and reduction machinery (NO read copy-ins) | inserts new buffer/copy ops       | invalidates op-order snapshots; loop_group_id offset must be strictly greater than any hint-driven group | post-stickify |
| 18 | `span_reduction` (`work_division.py:1621`) | after stickify passes; `MAX_SPAN_BYTES` obeys `_validate_max_cores()` | `op.data` type (Pointwise/Reduction), read/write memdeps | writes `op.op_it_space_splits` (span-reduction pass output) | none | commits per-op split; consumed by (19)/(20) | division |
| 19 | `_distribute_work` (`passes.py:399`) -> `cost_model_matmul_division` (`work_division.py:1756`) + `work_distribution` (`work_division.py:1634`) | after (18) | `op.op_it_space_splits`, memdeps, hardware cost model | rewrites `op.op_it_space_splits` for matmul/bmm; then divides remaining ops | none | invariant: every op divided by exactly one of (18), cost-model, or distribution | division |
| 20 | `_maybe_scratchpad_planning` (`passes.py:406`) -> `scratchpad_planning` (`scratchpad/allocator.py:2254`) | after division; gated by `config.lx_planning` | `graph.operations` (topo), tensor layouts and buffer sizes | writes `layout.allocation["lx"] = ...` on eligible buffers; falls back to greedy on `SolveError` | none | LX allocations are consumed later by codegen; HBM pool planning must skip LX-claimed buffers | LX allocation |

## CustomPreFusionPasses  (`passes.py:253`)

Runs on the list of LoopLevelIR scheduler nodes immediately before
Inductor's fusion pass.  Nodes are in topological order (contract on
input and output).

| Pass | Requires | Reads | Mutates | Creates/removes ops | Invalidates | Layout phase |
| ---- | -------- | ----- | ------- | ------------------- | ----------- | ------------ |
| `propagate_mutation_layouts` (`propagate_layouts.py:2007`) | after CustomPreSchedulingPasses; SchedulerNode graph built | per-node `n.node.data`, `n.node.layout`, `n.read_writes` | sets `n.node.layout` to a `FixedTiledLayout` on remaining `MutationLayoutSHOULDREMOVE` ops | none | consumes MutationLayoutSHOULDREMOVE where possible | second layout phase |
| `align_lx_producer_loop_order` (`scheduler.py:371`) | before `build_loop_scheduler_nodes` (must see plain `SchedulerNode`s) | scheduler node graph, LX-producer loop orders | reorders loops on LX-producing nodes | none | invariant: LX producer/consumer loop orders align | scheduler |
| `build_loop_scheduler_nodes` (`scheduler.py:306`) | after `align_lx_producer_loop_order`; ops with `loop_group_id` present | scheduler node graph, `loop_group_id` metadata | wraps nodes into `CountedLoopSchedulerNode` groups | replaces top-level nodes | downstream fusion sees `CountedLoopSchedulerNode` boundaries as unfusable | scheduler |

## CustomPostFusionPasses  (`passes.py:280`)

Runs on the LoopLevelIR nodes immediately after Inductor's fusion pass.

| Pass | Requires | Reads | Mutates | Creates/removes ops | Invalidates | Layout phase |
| ---- | -------- | ----- | ------- | ------------------- | ----------- | ------------ |
| `demote_incoherent_lx_buffers` (`scheduler.py:450`) | after fusion; final loop orders | LX producer/consumer loop orders | demotes non-coherent LX buffers back to HBM | metadata mutation | invariant: HBM pool planning must see demoted buffers as unclaimed intermediates | post-fusion |
| `spyre_fuse_nodes` (`fusion.py:41`) | after demote pass | scheduler node graph | fuses adjacent Spyre nodes into a single SDSC bundle | replaces adjacent nodes with a `FusedSchedulerNode` | HBM pool planning depends on this producing bundle-level nodes | post-fusion |
| `hbm_pool_planning` (`hbm_pool_planning.py:151`) | after `spyre_fuse_nodes`; nodes are final SDSC bundles; LX planning already ran | bundle-level nodes, per-buffer read/write sets, layout `.allocation` | writes `layout.allocation["hbm_pool"] = INTERMEDIATES_SEGMENT + offset` | none | invariant: skips buffers already claimed by LX planning | post-fusion |

## Layout-phase legend

| Phase           | Meaning                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| FX pre-grad     | Runs on the pre-grad FX graph                                              |
| FX post-grad    | Runs on the post-grad FX graph (before Inductor scheduler is built)        |
| pre-stickify    | Runs on the GraphLowering before Spyre tensor layouts are committed        |
| stickification  | Runs during the layout-commit phase (propagate/optimize/finalize/insert)   |
| post-stickify   | Runs after `finalize_layouts`; every op has a `FixedTiledLayout`           |
| division        | Runs after stickification; assigns per-op iteration-space splits           |
| LX allocation   | Runs after division; assigns on-chip scratchpad addresses                  |
| scheduler       | Runs on `BaseSchedulerNode`s, before Inductor's fusion pass                |
| post-fusion     | Runs on `BaseSchedulerNode`s, after Inductor's fusion pass                 |

## Open questions

These are the columns marked `?` in the matrix above -- they cannot be
settled without either running the pass on a real graph, or reading a
much larger portion of the source than fits into an audit turn.

1. **What set of upstream Inductor pattern matchers depend on `aten.addmm`
   surviving into post-grad?** `decompose_addmm` unconditionally rewrites
   it; if any post-grad matcher assumes `addmm` still exists, that
   matcher would silently miss.  Static grep on `torch._inductor` under
   `torch/2.13` would confirm.
2. **Does `_maybe_reorder_unhinted_interlopers` guarantee no cycles when
   both a move-before and a move-after are legal?** The docstring
   defines the two-cursor algorithm but does not prove termination on
   pathological interleavings.
3. **Does `enforce_indirect_access_layout` preserve topological order
   when it rewrites a producer layout that is shared with mutation
   ops?** The pass declares it only rewrites in place for single-
   consumer non-mutation cases; the corner cases (multi-consumer where
   only one is an indirect-access op) fall through to `_insert_relayout_copy`,
   but the docstring does not enumerate every branch.
4. **Is the invariant "every op divided by exactly one of span_reduction /
   cost_model_matmul_division / work_distribution" enforced by an
   assertion, or only by mutual exclusion in the callers?** The three
   pass entries assert nothing beyond `preassigned_ops` membership.  An
   `_iter_computed_buffers` skipping a corner case would silently divide
   the same op twice.
5. **What happens if `scratchpad_planning`'s greedy fallback also fails?**
   The `try/except SolveError` catches the analytic solver failure but
   does not re-catch a failure from the `ScratchpadAllocator(GreedyLayoutSolver,
   ...)` retry.  A greedy-solver failure would propagate as an uncaught
   `SolveError`.
6. **Does `hbm_pool_planning` treat `SpyreEmptyFallback` full buffers
   uniformly with `ComputedBuffer` intermediates for live-range
   computation?** The docstring calls this out as a bespoke path
   (ExternKernel invisible to the read/write sets) but does not spell
   out whether their live ranges are computed against the same bundle
   boundary.

## Provenance

- Ordered pipeline in `torch_spyre/_inductor/passes.py:444-485`.
- Wrappers (`_maybe_*`, `_distribute_work`) live at `passes.py:308-412`.
- Each pass function's own site is cited in the tables above.
- Torch-spyre citations use the form `torch-spyre@fea0c4b:<path>:<line>`
  and resolve against SHA fea0c4be901e1383b1f700dbad8887128b0fcb27 on
  `github.com/torch-spyre/torch-spyre` (private).
