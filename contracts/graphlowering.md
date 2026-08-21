---
name: graphlowering
pinned_sha: fea0c4be901e1383b1f700dbad8887128b0fcb27
pytorch_supported: v2.13.0
pytorch_main: c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62
status: draft
---

# GraphLowering: internal-state contract with upstream Inductor

## 1. What `GraphLowering` is (upstream)

`torch._inductor.graph.GraphLowering` is the Inductor IR container built
during `compile_fx`. It subclasses `torch.fx.Interpreter` and owns the
buffers, operations, name tables, size vars, and output list that later
codegen consumes. Only the shape of its state is upstream API; the
individual dict fields are private.

```
356  class GraphLowering(torch.fx.Interpreter):
357      graph_outputs: list[ir.IRNode]
...
415          self.graph_inputs: dict[str, TensorBox | TorchBindObject | sympy.Expr] = {}
416          self.graph_inputs_original: dict[str, InputBuffer] = {}
...
464          self.removed_buffers: OrderedSet[str] = OrderedSet()
...
469          self.inplaced_to_remove: OrderedSet[str] = OrderedSet()
...
485          self.name_to_buffer: dict[str, ir.Buffer] = {}
486          self.name_to_users: defaultdict[str, list[ir.IRNode]] = defaultdict(list)
487          self.name_to_op: dict[str, ir.Operation] = {}
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/graph.py#L356-L487, v2.12.0 fallback)

`V.graph` is the per-thread accessor for the currently-installed
`GraphLowering`. `torch/_inductor/virtualized.py` describes the pattern
explicitly:

```
16  There are a few distinct usage patterns for virtualized global variables:
...
24  2. Per-compilation global state.  Examples: ``V.fake_mode``, ``V.graph``.
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/virtualized.py#L16-L24, v2.12.0 fallback)

```
91  threadlocal = local()
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/virtualized.py#L91, v2.12.0 fallback)

Two consequences follow, and both matter later:

- `V.graph` is thread-local, so two threads each doing a `compile_fx`
  each see their own `GraphLowering`. State that lives *on* the
  `GraphLowering` instance is not shared across threads.
- State that lives on the *class* (`GraphLowering.<attr>`), on
  `torch._inductor.lowering.lowerings` (a module-level dict), or on
  `enable_spyre_lowerings.<attr>` (a function attribute) IS process-wide
  and IS shared across threads.

## 2. Torch-Spyre reads of `GraphLowering` internals

Every read below goes through `V.graph.<attr>` (per-thread) and treats
the `GraphLowering` as an authoritative name-directory. Torch-spyre
citations use the form `torch-spyre@fea0c4b:<path>:<line>` and resolve
against SHA fea0c4be901e1383b1f700dbad8887128b0fcb27 on
`github.com/torch-spyre/torch-spyre` (private).

Aggregate: 151 `V.graph.<attr>` reference sites (both reads and writes)
across the pinned tree.

```
$ grep -rn "V\.graph\." torch_spyre/ | wc -l  # in a torch-spyre checkout at fea0c4b
151
```

### 2.1 Buffer / operation lookup by name

The core read primitive is `V.graph.get_buffer(name)` /
`V.graph.try_get_buffer(name)` / direct dict access on
`V.graph.name_to_buffer`.

Representative sites:

```
torch_spyre/_inductor/hbm_pool_planning.py:112:    buf = V.graph.get_buffer(name)
torch_spyre/_inductor/hbm_pool_planning.py:236:        buf = V.graph.get_buffer(name)
torch_spyre/_inductor/hbm_pool_planning.py:257:        buf = V.graph.get_buffer(name)
torch_spyre/_inductor/hbm_pool_planning.py:461:            buf = V.graph.get_buffer(name)
torch_spyre/_inductor/insert_restickify.py:115:    graph_lowering = V.graph
torch_spyre/_inductor/optimize_restickify.py:727:        op = V.graph.get_buffer(name)
torch_spyre/_inductor/pass_utils.py:97:            buf = V.graph.get_buffer(arg.name)
torch_spyre/_inductor/pass_utils.py:1667:    buf_op = V.graph.get_buffer(buf_name)
torch_spyre/_inductor/propagate_layouts.py:123:            buf = V.graph.get_buffer(arg.name)
torch_spyre/_inductor/propagate_layouts.py:1461:        buf = V.graph.get_buffer(name) if name else None
torch_spyre/_inductor/propagate_layouts.py:1751:                target_buf = V.graph.get_buffer(target_name) if target_name else None
torch_spyre/_inductor/propagate_layouts.py:1933:                    isinstance(V.graph.get_buffer(r.name), SpyreConstantFallback)
torch_spyre/_inductor/scheduler.py:64:    buffer = V.graph.try_get_buffer(name)
torch_spyre/_inductor/scheduler.py:360:    buffer = V.graph.try_get_buffer(name)
torch_spyre/_inductor/scheduler.py:552:        buffer = V.graph.try_get_buffer(source_name)
torch_spyre/_inductor/spyre_kernel.py:1007:            buf = V.graph.get_buffer(name)
torch_spyre/_inductor/spyre_kernel.py:1021:        buf = V.graph.get_buffer(name)
torch_spyre/_inductor/spyre_kernel.py:1051:        buf = V.graph.get_buffer(real_dst_name)
torch_spyre/_inductor/spyre_kernel.py:1177:        buf = V.graph.get_buffer(name)
torch_spyre/_inductor/wsr/coarse_tile.py:2141:        buf = V.graph.get_buffer(d.name)
torch_spyre/_inductor/wsr/coarse_tile.py:2687:    full_buf = V.graph.get_buffer(dep.name)
torch_spyre/_inductor/wsr/coarse_tile.py:3109:    full_buf = V.graph.get_buffer(dep.name)
torch_spyre/_inductor/wsr/coarse_tile.py:3566:        combine_buf = V.graph.name_to_buffer.get(combine_name)
torch_spyre/_inductor/wsr/propagate_named_dims.py:79:    return V.graph.get_buffer(dep.name)
torch_spyre/_inductor/wsr/span_overflow_hint_analysis.py:448:            buf = V.graph.get_buffer(dep.name)
torch_spyre/_inductor/dump_cost_model.py:119:        buf = V.graph.get_buffer(name)
torch_spyre/_inductor/split_multi_ops.py:735:            buf = V.graph.get_buffer(inp.name)
torch_spyre/_inductor/split_multi_ops.py:421:            dtype_map[vid] = V.graph.get_buffer(buf_name).get_layout().dtype
```

### 2.2 Graph I/O and output-name reads

```
torch_spyre/_inductor/deadcode_elimination.py:33:    live_bufs: set[str] = set(V.graph.get_output_names())
torch_spyre/_inductor/dedup_constants.py:63:    if D in V.graph.get_output_names():
torch_spyre/_inductor/hbm_pool_planning.py:195:    graph_inputs: set[str] = set(V.graph.graph_inputs.keys())
torch_spyre/_inductor/hbm_pool_planning.py:196:    graph_outputs: set[str] = set(V.graph.get_output_names())
torch_spyre/_inductor/optimize_restickify.py:378:    for name in V.graph.graph_input_names:
torch_spyre/_inductor/optimize_restickify.py:379:        tb = V.graph.graph_inputs[name]
torch_spyre/_inductor/propagate_layouts.py:1447:    graph_input = V.graph.graph_inputs.get(name)
torch_spyre/_inductor/propagate_layouts.py:1523:    graph_inputs = set(V.graph.graph_input_names)
torch_spyre/_inductor/propagate_layouts.py:1524:    graph_outputs = set(V.graph.get_output_names())
torch_spyre/_inductor/propagate_layouts.py:1902:                        graph_input = V.graph.graph_inputs.get(target_name)
torch_spyre/_inductor/wrapper.py:114:        if old_name not in V.graph.get_output_names() and delete_old:
torch_spyre/_inductor/wsr/coarse_tile.py:644:                        target_is_graph_input = mut_target_name in V.graph.graph_inputs
torch_spyre/_inductor/wsr/coarse_tile.py:2161:        return set(V.graph.get_output_names())
torch_spyre/_inductor/wsr/propagate_named_dims.py:86:    tb = V.graph.graph_inputs.get(dep.name)
torch_spyre/_inductor/wsr/propagate_named_dims.py:102:    tb = V.graph.graph_inputs.get(dep.name)
```

### 2.3 Sizevars / shape-env reads

`V.graph.sizevars` is treated as the single hint / simplifier / shape-env
oracle by every pass that reasons about symbolic shapes:

```
torch_spyre/_inductor/pass_utils.py:161:        return V.graph.sizevars.optimization_hint(expr)
torch_spyre/_inductor/pass_utils.py:183:    vr = V.graph.sizevars.shape_env.bound_sympy(expr)
torch_spyre/_inductor/pass_utils.py:198:    vr = V.graph.sizevars.shape_env.bound_sympy(expr)
torch_spyre/_inductor/pass_utils.py:307:            hint = V.graph.sizevars.optimization_hint(s)
torch_spyre/_inductor/pass_utils.py:341:    return V.graph.sizevars.optimization_hint(expr)
torch_spyre/_inductor/pass_utils.py:355:    shape_env = V.graph.sizevars.shape_env
torch_spyre/_inductor/scheduler.py:601:        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)
torch_spyre/_inductor/spyre_kernel.py:219:                s: V.graph.sizevars.guarding_hint_or_throw(s) for s in v.free_symbols
torch_spyre/_inductor/spyre_kernel.py:226:            return repr(V.graph.sizevars.guarding_hint_or_throw(v))
torch_spyre/_inductor/spyre_kernel.py:1025:        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
torch_spyre/_inductor/spyre_kernel.py:1068:        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
torch_spyre/_inductor/spyre_kernel.py:1186:        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
torch_spyre/_inductor/views.py:132:        return V.graph.sizevars.optimization_hint(expr)
torch_spyre/_inductor/codegen/superdsc.py:1482:        return V.graph.sizevars.guarding_hint_or_throw(expr)
```

### 2.4 Operations list / topological-order reads

```
torch_spyre/_inductor/lowering.py:1492:    for op in V.graph.operations[first_new_op:]:
torch_spyre/_inductor/lowering.py:1516:    first_new_op = len(V.graph.operations)
torch_spyre/_inductor/propagate_layouts.py:1733:    # Operations are in topological order (guaranteed by GraphLowering).
```

### 2.5 FX-graph, current-node, and scheduler reads

```
torch_spyre/_inductor/insert_restickify.py:116:    fx_graph = graph_lowering.graph
torch_spyre/_inductor/lowering.py:325:    fx_graph = V.graph.graph
torch_spyre/_inductor/lowering.py:1424:        torch.ops.spyre.constant, V.graph.current_node.target._overloadname
torch_spyre/_inductor/lowering.py:1434:        torch.ops.spyre.empty, V.graph.current_node.target._overloadname
torch_spyre/_inductor/spyre_kernel.py:1048:        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
torch_spyre/_inductor/spyre_kernel.py:1061:        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
torch_spyre/_inductor/spyre_kernel.py:1188:        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
torch_spyre/_inductor/spyre_kernel.py:1329:        wrapper = V.graph.wrapper_code
torch_spyre/_inductor/spyre_kernel.py:1352:        wrapper = V.graph.wrapper_code
torch_spyre/_inductor/scheduler.py:830:        wrapper = V.graph.wrapper_code
```

## 3. Torch-Spyre writes to `GraphLowering` internals

Writes fall into two families: (a) the *documented* mutation surfaces
that upstream expects passes to touch (`register_buffer`,
`register_operation`, `removed_buffers`, `inplaced_to_remove`,
`name_to_buffer`), and (b) *undocumented* attribute injection where
Torch-Spyre stashes per-graph analysis state on the `GraphLowering`
instance itself.

### 3.1 Sanctioned upstream mutation: `register_buffer` / `register_operation`

Called from every Spyre custom IR node constructor:

```
torch_spyre/_inductor/ir.py:401:        self.name = V.graph.register_buffer(self)
torch_spyre/_inductor/ir.py:402:        V.graph.register_operation(self)
torch_spyre/_inductor/ir.py:451:        self.name = V.graph.register_buffer(self)
torch_spyre/_inductor/ir.py:452:        V.graph.register_operation(self)
torch_spyre/_inductor/ir.py:515:        self.name = V.graph.register_buffer(self)
torch_spyre/_inductor/ir.py:516:        V.graph.register_operation(self)
torch_spyre/_inductor/ir.py:574:        self.name = V.graph.register_buffer(self)
torch_spyre/_inductor/ir.py:575:        V.graph.register_operation(self)
torch_spyre/_inductor/ir.py:627:        self.name = V.graph.register_buffer(self)
torch_spyre/_inductor/ir.py:628:        V.graph.register_operation(self)
torch_spyre/_inductor/ir.py:678:        self.name = V.graph.register_buffer(self)
torch_spyre/_inductor/ir.py:679:        V.graph.register_operation(self)
torch_spyre/_inductor/lowering.py:1291:    buffer.name = V.graph.register_buffer(buffer)
torch_spyre/_inductor/lowering.py:1292:    V.graph.register_operation(buffer)
```

Upstream contract for these two entry points:

```
1053  def register_operation(self, op: ir.Operation) -> str:
...
1058          self.name_to_op[name] = op
...
1062  def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str:
...
1065          self.name_to_buffer[name] = buffer
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/graph.py#L1053-L1082, v2.12.0 fallback)

### 3.2 Direct `name_to_buffer` writes (bypass `register_buffer`)

Passes that replace or splice ops write into `name_to_buffer` directly.
This is where the *nine-invariant replacement contract* the
`computed-buffer.md` sibling covers lives; the writes below are the
sites that maintain (or need to maintain) it.

```
torch_spyre/_inductor/insert_restickify.py:261:    V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer
torch_spyre/_inductor/split_multi_ops.py:689:    V.graph.name_to_buffer[new_op.get_name()] = new_op
torch_spyre/_inductor/wsr/coarse_tile.py:2457:    V.graph.name_to_buffer[copy_name] = copy_buf
torch_spyre/_inductor/wsr/coarse_tile.py:3062:    V.graph.name_to_buffer[copy_name] = copy_buf
torch_spyre/_inductor/wsr/coarse_tile.py:3154:    V.graph.name_to_buffer[new_op.get_name()] = new_op
torch_spyre/_inductor/wsr/coarse_tile.py:3471:    V.graph.name_to_buffer[combine_name] = combine_buf
torch_spyre/_inductor/wsr/coarse_tile.py:3558:    V.graph.name_to_buffer[copy_name] = copy_buf
torch_spyre/_inductor/wsr/coarse_tile.py:3754:    V.graph.name_to_buffer[fill_name] = fill_buf
torch_spyre/_inductor/wsr/coarse_tile.py:3918:    V.graph.name_to_buffer[new_consumer.get_name()] = operations[
torch_spyre/_inductor/wsr/coarse_tile.py:4299:    V.graph.name_to_buffer[new_op.get_name()] = new_op
```

### 3.3 `removed_buffers` / `inplaced_to_remove` writes

`removed_buffers` is the upstream convention for "this name is dead;
the wrapper must not allocate it". Torch-Spyre writes it from four
places:

```
torch_spyre/_inductor/dedup_constants.py:98:    V.graph.removed_buffers.add(D)
torch_spyre/_inductor/dedup_constants.py:99:    V.graph.name_to_buffer.pop(D, None)
torch_spyre/_inductor/dedup_constants.py:100:    V.graph.name_to_op.pop(op_name, None)
torch_spyre/_inductor/dedup_constants.py:103:    extra_users = V.graph.name_to_users.pop(D, [])
torch_spyre/_inductor/dedup_constants.py:105:        V.graph.name_to_users.setdefault(C, []).extend(extra_users)
torch_spyre/_inductor/hbm_pool_planning.py:481:                V.graph.removed_buffers.add(name)
torch_spyre/_inductor/propagate_layouts.py:1604:            V.graph.removed_buffers.add(write.name)
torch_spyre/_inductor/spyre_kernel.py:1050:            V.graph.removed_buffers.add(name)
torch_spyre/_inductor/spyre_kernel.py:1072:            V.graph.removed_buffers.add(name)
torch_spyre/_inductor/spyre_kernel.py:1191:            V.graph.removed_buffers.add(name)
torch_spyre/_inductor/scheduler.py:716:        V.graph.removed_buffers |= kernel.removed_buffers
torch_spyre/_inductor/scheduler.py:717:        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
torch_spyre/_inductor/scheduler.py:754:        V.graph.removed_buffers |= kernel.removed_buffers
torch_spyre/_inductor/scheduler.py:755:        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
```

The `dedup_constants` block is the only Torch-Spyre site that mutates
`name_to_op` and `name_to_users` directly; the merge convention there
is worth noting for the concurrency argument (§5): every pop/setdefault
is unsynchronized, but it's on a per-thread `GraphLowering`, so intra-
thread ordering with earlier passes is the only requirement.

### 3.4 `graph_outputs` mutation

Coarse-tile rewrites can substitute the output-list entries in place:

```
torch_spyre/_inductor/wsr/coarse_tile.py:4303:    """Replace references to old_name in V.graph.graph_outputs with new_buf."""
torch_spyre/_inductor/wsr/coarse_tile.py:4305:        outputs = V.graph.graph_outputs
```

### 3.5 Undocumented attribute injection on the `GraphLowering` instance

None of the following attributes exist upstream. Torch-Spyre stashes
per-graph state directly on the `GraphLowering` object; every consumer
reads it back via `getattr(V.graph, ..., default)` or a bare
`V.graph.<attr>`.

```
torch_spyre/_inductor/insert_restickify.py:457:    V.graph.restickify_plan = plan
torch_spyre/_inductor/hbm_pool_planning.py:192:        V.graph.hbm_pool_sizes = {}
torch_spyre/_inductor/hbm_pool_planning.py:372:    V.graph.hbm_pool_sizes = {}
torch_spyre/_inductor/hbm_pool_planning.py:514:        V.graph.hbm_pool_sizes[bundle_name] = pool_extent
torch_spyre/_inductor/scheduler.py:668:        emitted = V.graph.__dict__.setdefault("_emitted_layout_targets", set())
torch_spyre/_inductor/views.py:170:        V.graph._repeat_info = dict(repeat_info)
torch_spyre/_inductor/views.py:172:        V.graph._repeat_info.update(repeat_info)
torch_spyre/_inductor/views.py:678:        V.graph._repeat_info.clear()
```

These are safe against upstream churn only because they collide with no
existing `GraphLowering` attribute at the pinned upstream tip. There is
no positive test that this stays true on a future `v2.14` bump — see
Open Questions.

### 3.6 `V.graph.sizevars` monkey-patch on wrapper construction

`SpyrePythonWrapperCodegen.__init__` rebinds a *method* on the live
`sizevars` instance:

```
31  class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
32      def __init__(self):
33          super().__init__()
34          V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
35              V.graph.sizevars, SizeVarAllocator
36          )
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/wrapper.py:31-36`)

This is *not* a class-level patch: the bound-method assignment lands on
the per-`GraphLowering` `SizeVarAllocator` instance, so it does not
leak between compiles. That is worth noting because it stands out from
the class-level patches in `patches.py` (§4.2).

### 3.7 `GraphLowering._update_scheduler` class-level monkey-patch

`enable_spyre_context` also swaps the *class* method
`GraphLowering._update_scheduler`, and *does not hold any lock while
doing so*. See §4.2.

## 4. The lowering-registry lock

### 4.1 Declaration and scope

```
59  # A module-level lock + nesting counter to make the CM reentrant/thread-safe
60  _lowerings_lock = threading.RLock()
61  _lowerings_nesting = 0
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/lowering.py:59-61`)

The lock is entered once at the top of `enable_spyre_lowerings()`, held
across the entire mutation phase (unregister + register + save aten
overloads), and released *before* the yield. It is re-acquired on
`finally` for the symmetric restore phase.

```
172  # Context manager that enables spyre specific lowerings in addition to PyTorch in-tree lowerings
173  @contextmanager
174  def enable_spyre_lowerings():
...
181      global _lowerings_nesting
182      with _lowerings_lock:
183          first_enter = (_lowerings_nesting == 0)  # fmt: skip
184          _lowerings_nesting += 1
185
186          if first_enter:
187              enable_spyre_lowerings._removed_fallbacks = {}
188              enable_spyre_lowerings._removed_fallbacks = unregister_lowerings(
189                  fallback_ops, lowering.lowerings, allow_missing=True
190              )
...
195              enable_spyre_lowerings._added_fallbacks = register_fallback_over_decomp(
196                  fallback_ops
197              )
198              saved_intree_lowerings = {}
199              for spyre_lowering_op, spyre_lowering_impl in spyre_lowerings.items():
200                  if spyre_lowering_op in lowering.lowerings:
201                      saved_intree_lowerings[spyre_lowering_op] = lowering.lowerings[
202                          spyre_lowering_op
203                      ]
204                  lowering.lowerings[spyre_lowering_op] = spyre_lowering_impl
...
238              enable_spyre_lowerings._saved_aten_lowerings = saved
239              enable_spyre_lowerings._saved_lowerings = saved_intree_lowerings
240
241          try:
242              yield
243          finally:
244              _lowerings_nesting -= 1
245              last_exit = (_lowerings_nesting == 0)  # fmt: skip
246              if last_exit:
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/lowering.py:172-246`)

Two observations from this text:

1. The `with _lowerings_lock:` block wraps both the enter and (via the
   `try/finally` structure) the exit phases, but **not the yield**.
   The lock is released once the mutation of `lowering.lowerings` /
   `lowering.fallbacks` completes; user code inside
   `with enable_spyre_lowerings():` runs with the lock *not* held.
2. The nesting counter (`_lowerings_nesting`) makes the CM reentrant
   for a single thread; the RLock makes reentrant acquire cheap. On the
   *last* exit (`_lowerings_nesting == 0` after decrement), the CM
   restores the process-wide state.

### 4.2 Class-level monkey-patches outside the lock

`enable_spyre_context` (the CM that wraps `enable_spyre_lowerings`)
performs two additional process-wide mutations, and neither is inside
`_lowerings_lock`:

```
109      old_update_scheduler = GraphLowering._update_scheduler
...
113      def _spyre_update_scheduler(self: GraphLowering) -> None:
...
128          old_update_scheduler(self)
129
130      GraphLowering._update_scheduler = _spyre_update_scheduler  # type: ignore[method-assign]
...
144      old_scheduler_node_has_side_effects = SchedulerNode.has_side_effects
...
159      SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects  # type: ignore[method-assign]
...
161      with (
162          spyre_data_types(),
163          enable_spyre_lowerings(),
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/patches.py:109-163`)

The class rebinds happen *before* `enable_spyre_lowerings()` is even
entered, so `_lowerings_lock` does not protect them at all. Restore
happens on the `finally` branch:

```
170          try:
171              yield
172          finally:
173              joint_graph.pass_patterns[:] = origin_pass
174              Loops.has_large_inner_fn = old_loop
175              GraphLowering._update_scheduler = old_update_scheduler  # type: ignore[method-assign]
176              SchedulerNode.has_side_effects = old_scheduler_node_has_side_effects  # type: ignore[method-assign]
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/patches.py:170-176`)

### 4.3 What the lock actually protects

Inside the critical section, `enable_spyre_lowerings` mutates:

- `torch._inductor.lowering.lowerings` — the upstream module-level
  op → callable registry. Every write goes through `pop`, direct
  `__setitem__`, and (indirectly, via `make_fallback`)
  `lowering.fallbacks` (`torch._inductor.lowering.fallbacks`, an
  OrderedSet).
- `enable_spyre_lowerings._removed_fallbacks`,
  `enable_spyre_lowerings._added_fallbacks`,
  `enable_spyre_lowerings._saved_lowerings`,
  `enable_spyre_lowerings._saved_aten_lowerings` — the function-attribute
  bookkeeping that the exit branch reads to restore.

The lock does *not* protect:

- `GraphLowering._update_scheduler` (§4.2) — assigned outside the CM.
- `SchedulerNode.has_side_effects` (§4.2) — assigned outside the CM.
- `torch._prims_common._computation_dtype_map` — mutated in
  `spyre_data_types` (see `torch-spyre@fea0c4b:torch_spyre/_inductor/patches.py:27-37`).
- `torch._inductor.fx_passes.joint_graph.pass_patterns` — popped/restored
  in `enable_spyre_context` at lines 105-107 / 173.
- `torch._inductor.ir.Loops.has_large_inner_fn` — replaced at lines
  100-101, restored at 174.
- The undocumented `V.graph.<attr>` injections in §3.5 (those live on a
  per-thread `GraphLowering`, so the lock would be the wrong tool).

### 4.4 An unused sibling lock

`torch_spyre/_inductor/decompositions.py:70` declares
`_decompositions_lock = threading.RLock()` but the file contains no
`with _decompositions_lock:` — the lock is never acquired. This is not
a `GraphLowering` mutation, but it is a symmetric case worth flagging:
`register_spyre_decompositions` mutates a module-level
`spyre_decompositions` dict with no synchronization.

```
$ grep -n "_decompositions_lock" torch_spyre/_inductor/decompositions.py  # in a torch-spyre checkout at fea0c4b
70:_decompositions_lock = threading.RLock()
```

## 5. Concurrency scenarios

The threat model: `torch.compile` is called from multiple threads in
the same process. Each call ends up in `_spyre_wrapper` at
`torch_spyre/_inductor/__init__.py:156` and enters `enable_spyre_context`.

### 5.1 Two concurrent Spyre compiles

*Sequence*: threads T1, T2 both enter `enable_spyre_context`.

- **Class rebind race (patches.py:109-159)**: T1 reads
  `old_update_scheduler = GraphLowering._update_scheduler`; before T1
  assigns its wrapper, T2 also reads the same original. Then T1
  assigns its wrapper; T2 assigns *its* wrapper — which closes over
  the original, not over T1's. On T1's `finally`, T1 restores its
  captured original — which is the true original, so T2's wrapper is
  torn out from under T2 while T2 is still compiling. Symmetric case
  with T2 finishing first. The read-modify-write on
  `GraphLowering._update_scheduler` is entirely unsynchronized (§4.2).
  `_lowerings_lock` is not held around it (§4.3). This scenario is
  argued from the source alone; a demonstration test lives (or will
  live) at OPEN QUESTION in §6.
- **Config patch stacking**: `torch._inductor.config.patch(new_config)`
  is context-nested by both threads. The upstream `config.patch` uses
  a per-thread stack in modern Inductor (see
  `torch._inductor.config` handling — OPEN QUESTION: verify against
  pinned upstream), so this is likely fine.
- **`joint_graph.pass_patterns[:]` / `Loops.has_large_inner_fn` race**:
  same shape as the class-rebind race — a mutable module-level list
  and a class attribute are swapped and later restored to *the caller's
  captured original*. Interleaved compiles corrupt the restore.
- **Lowering registry**: `enable_spyre_lowerings` DOES take the lock,
  so the mutation of `lowering.lowerings` is serialized. But: the
  RLock is released before the yield. If T1 is inside the yield (still
  compiling) and T2 enters, T2 waits on the lock only long enough to
  see `_lowerings_nesting == 1` (already nonzero from T1), increment
  it to `2`, and skip the `first_enter` block entirely — because
  `_lowerings_nesting` is a *process-wide global*, not a per-thread
  counter. That is the intended reentrant behavior for one thread, but
  it is unsound across threads: T2's yield now runs against T1's
  lowerings without doing its own registration, and T1's final exit
  will restore the process-wide state while T2 is still inside its
  yield.

### 5.2 Spyre + CPU-Inductor compile concurrent

Same fleet of process-wide state as §5.1, plus:

- **Class rebind visible to CPU compile**: while
  `GraphLowering._update_scheduler` is the Spyre wrapper, any CPU
  `compile_fx` running on another thread that reaches
  `GraphLowering._update_scheduler(self)` runs the Spyre wrapper. The
  wrapper guards on `_spyre_pre_scheduling_complete` and calls
  `_pre_scheduling_pass(self)` unconditionally on the CPU graph —
  which is a Spyre-only pipeline (deadcode elimination, WSR passes,
  restickify planning). Whether those passes crash on a CPU
  `GraphLowering` or silently corrupt it is not proven here but the
  code path is reachable.
- **`SchedulerNode.has_side_effects` visible to CPU compile**: the
  wrapper's early-return branch on `_coarse_tile_force_live` or
  `MutationLayoutSHOULDREMOVE` is benign for CPU graphs (attribute
  won't be present; `isinstance` check protects the second branch),
  so this one *is* likely safe — but "likely" from source-reading
  alone; see the Phase 4 metamorphic test slot in §5.4.
- **Lowering registry visible to CPU compile**: while
  `enable_spyre_lowerings` is active, `lowering.lowerings` on the
  module has been rewritten. A concurrent CPU `compile_fx` reads
  from the same module; it will get Spyre lowerings for the ops
  Torch-Spyre swapped (aten.clamp, aten.clamp_min, aten.clamp_max,
  and the Spyre lowering table's keys), which is a functional
  regression.

### 5.3 Nested Spyre compile

If a Spyre compile triggers, via decomposition or subgraph handling,
another Spyre compile on the same thread:

- `_lowerings_lock` is an `RLock`, so re-entry acquires it cheaply
  (§4.1).
- `_lowerings_nesting` is incremented; the inner `first_enter` branch
  is `False`, so no re-registration happens. On inner exit, the
  counter drops back to 1, `last_exit` is `False`, no restore runs.
  On outer exit, the counter reaches 0 and the single restore fires.
  This is well-formed *for a single thread*.
- The class rebinds in `enable_spyre_context` (§4.2) have **no nesting
  counter**. Nested `enable_spyre_context` calls each read the current
  `GraphLowering._update_scheduler` (which is the outer wrapper) and
  install their own wrapper that closes over it. On inner exit, the
  outer wrapper is restored — correct. On outer exit, the *original*
  upstream method is restored — correct. So the class-rebind path is
  actually safe for nested single-thread use; it is the multi-thread
  interleaving in §5.1 that breaks.

### 5.4 Cross-reference: Phase 4 metamorphic tests

The Phase 4 audit slot will house metamorphic tests that exercise the
races argued above. When those land, this section is updated to link
them (do not duplicate the argument here). Test outlines the coordinator
should schedule:

- `test_concurrent_spyre_compiles_do_not_swap_scheduler_hook` — two
  threads each entering `enable_spyre_context` around a trivial graph;
  assert `GraphLowering._update_scheduler is original` after both exit.
- `test_cpu_compile_during_spyre_compile_does_not_see_spyre_lowerings` —
  arrange one Spyre thread inside the yield, verify a concurrent CPU
  `torch.compile` on a plain aten op does not pick up a Spyre lowering.
- `test_nested_spyre_compile_restores_once` — pure single-thread nesting
  correctness for `enable_spyre_lowerings` (should already pass).

## 6. Open questions

- **Byte-for-byte upstream match.** Line citations for `graph.py` and
  `virtualized.py` come from v2.12.0
  (SHA `0d62256a2b23365f8e1604297eb23a6545102aa8`), because the pinned
  target is `v2.13.0` and no v2.13 tag is checked out locally. Attribute names (`name_to_buffer`, `name_to_op`,
  `name_to_users`, `removed_buffers`, `inplaced_to_remove`,
  `graph_inputs`, `graph_outputs`, `sizevars`) match by symbol at v2.13
  main (`c3ebaaba`) as far as identifier grep goes; line numbers
  differ. Verify against a v2.13.0 checkout before freezing.
- **`torch._inductor.config.patch` thread-safety.** §5.1 assumes
  `config.patch` uses a per-thread stack under the hood. This has not
  been re-verified against the pinned upstream; if `config.patch`
  writes shared module state, the concurrent-compile scenario grows a
  fourth failure mode.
- **Whether `_pre_scheduling_pass(self)` on a non-Spyre `GraphLowering`
  crashes or corrupts.** §5.2 identifies the code path but does not
  claim which. A minimal reproducer under Phase 4 would resolve this.
- **`_decompositions_lock` in `decompositions.py:70`.** Declared but
  never taken. Is that an oversight (the CM was intended to be
  locked) or a leftover? Not strictly in scope for this contract; noted
  because it is symmetric to `_lowerings_lock`.
- **Restickify plan and HBM-pool-sizes lifetime.** `V.graph.restickify_plan`
  (`insert_restickify.py:457`) and `V.graph.hbm_pool_sizes`
  (`hbm_pool_planning.py:192,372,514`) are set on the per-thread
  `GraphLowering` and read by later passes. Each `compile_fx` invocation
  gets a fresh `GraphLowering` (upstream constructs one per invocation
  — see `graph.py:415-487`), so cross-compile carry-over is not
  possible in principle. OPEN QUESTION: confirm no cache path
  (`AOTAutograd`, FX-graph cache) reuses a `GraphLowering` across
  invocations at v2.13.
- **`_emitted_layout_targets` reset expectation.** The scheduler
  comment at `scheduler.py:660-668` says "starts empty for each graph
  without any explicit reset", relying on freshness of the
  `GraphLowering`. This is the same assumption as the point above and
  shares the same open question.
