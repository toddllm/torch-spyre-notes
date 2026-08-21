---
name: dependency-extraction
pinned_sha: fea0c4be901e1383b1f700dbad8887128b0fcb27
pytorch_supported: v2.13.0
pytorch_main: c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62
status: draft
---

# Dependency extraction: what mutates the answer of `op.get_read_writes()`

## 1. Header — what `get_read_writes()` computes, and where

`ComputedBuffer.get_read_writes()` is the entry point every Torch-Spyre
compiler pass uses to obtain the set of reads (`MemoryDep`/`StarDep`) and
writes an IR op performs. Upstream, it lives on `ComputedBuffer` in
`torch/_inductor/ir.py`. The pinned Torch-Spyre repo targets
`v2.13.0`; the closest available upstream reference is `v2.12.0`
(SHA `0d62256a2b23365f8e1604297eb23a6545102aa8`), which matches the
shape of the code the pinned scan calls into, and citations below
use it for the upstream side. OPEN QUESTION: verify byte-for-byte
against a checked-out v2.13.0 tag before freezing.

The upstream definition, quoting the pinned-supported layout (torch
v2.12 fallback):

```
4886    def get_read_writes(self) -> dependencies.ReadWrites:
4887        if not isinstance(self.data, (Reduction, Scan, Sort, Pointwise)):
4888            return dependencies.ReadWrites(
4889                reads=OrderedSet(),
4890                writes=OrderedSet(),
4891                index_exprs=OrderedSet(),
4892            )
4893
4894        with patch.object(FlexibleLayout, "allow_indexing", True):
4895            if self.data.get_reduction_type():
4896                return extract_read_writes(
4897                    self.get_store_function(),
4898                    self.data.get_pointwise_size(),
4899                    self.data.get_reduction_size(),
4900                )
4901            else:
4902                return extract_read_writes(
4903                    self.get_store_function(),
4904                    self.data.get_size(),
4905                )
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ir.py#L4886-L4905)

`get_store_function()` wires layout + data:

```
4947    def get_store_function(self) -> Callable[..., None]:
4948        indexer = self.get_layout().as_fixed().make_indexer()
4949        if isinstance(self.data, (Reduction, Scan, Sort)):
4950            return partial(self.data.store_reduction, self.name, indexer)
4951        else:
4952            assert isinstance(self.data, Pointwise), type(self.data)
4953            return partial(self.data.store_output, self.name, indexer)
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ir.py#L4947-L4953)

And `extract_read_writes` re-traces the store function under a fresh
`RecordLoadStore` ops handler every time:

```
659 def extract_read_writes(
660     fn: Callable[..., Any],
661     *argsizes: Sequence[sympy.Expr],
662     normalize: bool = False,
...
679         rw = RecordLoadStore(var_ranges, normalize=normalize)
680         with V.set_ops_handler(rw):
681             fn(*args, *hidden_args)
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/dependencies.py#L659-L682)

Two facts follow directly from this shape:

- The result is **not memoized upstream** — every call re-runs sympy
  dependency extraction. Torch-Spyre already acknowledges this in the
  memo comment: "`ComputedBuffer.get_read_writes` re-runs sympy
  dependency extraction on every call and is not cached upstream"
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:105-106`).
- The set of fields that can *change* the result is exactly the set of
  fields read (directly or transitively) inside `get_read_writes` and
  `extract_read_writes`. Section 2 enumerates them.

## 2. Fields on `ComputedBuffer` whose mutation changes `get_read_writes()`

### 2.1 `self.data` (a `Loops` — `Reduction` | `Scan` | `Sort` | `Pointwise`)

`get_read_writes` dispatches on `isinstance(self.data, ...)`
(ir.py:4887), and both branches call `self.get_store_function()` which
delegates to `self.data.store_output` / `self.data.store_reduction`
(ir.py:4949-4953).

Wholesale replacement of `self.data` therefore changes everything the
next `get_read_writes()` returns. The frozen-dataclass constraint on
`ComputedBuffer` is why torch-spyre uses `replace_computed_buffer_body`
(section 3) rather than in-place assignment for `data` swaps.

### 2.2 `self.data.inner_fn`

`store_output` (Pointwise) and `store_reduction` (Reduction/Scan/Sort)
both call `self.inner_fn(vars, ...)` — this is the trace that
`extract_read_writes` records as reads. Cite:

```
1145    def store_output(
1146        self,
1147        output_name: str | None,
1148        indexer: Callable[[Sequence[Expr]], Never],
1149        vars: Sequence[Expr],
1150    ) -> None:
1151        loader = self.make_loader()
1152        return ops.store(output_name or "unnamed", indexer(vars), loader(vars))
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ir.py#L1145-L1152)

For `Pointwise`, `make_loader()` returns `self.inner_fn` directly
(ir.py:1127-1132), so any change to `data.inner_fn` — even swapping
buffer names it loads — changes the recorded reads.

For `Reduction`:

```
1300    def store_reduction(
1301        self,
1302        output_name: str | None,
...
1311            self.inner_fn(vars, reduction_vars),
1312        )
1313        ops.store_reduction(output_name or "unnamed", indexer(vars), value)
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ir.py#L1300-L1313)

### 2.3 `self.data.ranges` and `self.data.reduction_ranges`

`get_read_writes` passes `self.data.get_pointwise_size()` and (for
reductions) `self.data.get_reduction_size()` as argsizes to
`extract_read_writes` (ir.py:4898-4904). Those return `self.ranges` and
`self.reduction_ranges` respectively:

```
1018    def get_size(self) -> Sequence[Expr]:
1019        return self.ranges
1020
1021    def get_pointwise_size(self) -> Sequence[Expr]:
1022        return self.ranges
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ir.py#L1018-L1022)

```
1294    def get_reduction_size(self) -> Sequence[Expr]:
1295        return self.reduction_ranges
```
(https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ir.py#L1294-L1295)

Those sizes drive `index_vars_squeeze`, which materializes the loop
variable sets and calls `SqueezeView.squeezer` — a squeezed dim vanishes
from `dep.var_names` entirely (loop_info's own docstring at
`torch-spyre@fea0c4b:torch_spyre/_inductor/loop_info.py:246-260`
elaborates on this). So changing `ranges`/`reduction_ranges` is not
just a "size" change — it can also change *which symbols* appear in
each dep's `index`.

### 2.4 `self.data.reduction_type` (Reduction only)

`get_read_writes` branches on `self.data.get_reduction_type()`
(ir.py:4895), which returns `self.reduction_type` for Reductions
(ir.py:1297-1298). A `None` return sends the call down the pointwise
path with only `ranges`; a non-`None` return also feeds
`reduction_ranges` in. Mutating `reduction_type` from a non-`None`
value to `None` (or the reverse) therefore changes both the argsize
vector and which store function is invoked (`store_output` vs
`store_reduction`, ir.py:4949-4953).

### 2.5 `self.layout` (only insofar as it feeds `get_store_function`)

`get_store_function` calls `self.get_layout().as_fixed().make_indexer()`
and hands the resulting `indexer` to `store_output`/`store_reduction`.
The store function then invokes `ops.store(output_name, indexer(vars), ...)`
(ir.py:1152) — i.e. the write dep's index expression is
`indexer(vars)`. A layout change that produces a different `make_indexer`
result changes the recorded *write* dep's index. Reads are not
affected by `self.layout` (they come purely from `inner_fn`'s
`ops.load` calls). This asymmetry matters for the invalidation contract
below.

### 2.6 `self.name`

`get_store_function` bakes `self.name` into the partial as the store's
output-name argument. Renaming a buffer changes the string in the write
dep. No torch-spyre pass in the pinned scan renames a `ComputedBuffer`
in place, but `replace_computed_buffer_body` explicitly *preserves*
the name via `name=op.get_name()`
(`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1334`).

### 2.7 `Scatter.output_indexer` (Scatter is a Pointwise subclass)

Scatter's `store_output` uses `self.output_indexer(vars)` in place of
the layout indexer (upstream ir.py:1184-1191). Torch-Spyre does not
mutate `output_indexer` in the pinned scan (`grep output_indexer` over
`torch_spyre/_inductor/` at `fea0c4b` yields no in-place assignment), but any future pass that did would change the write
dep's index expression. Include it in the invalidation contract for
completeness.

### 2.8 Buffer names referenced by `WrapperHandler`-installed inner_fns

When Torch-Spyre wraps `inner_fn` via a `WrapperHandler`
(`NameSwapHandler`, `_NameSwapHandler`, `_NameAndIndexSwapHandler`,
`_RetileLoadIndexHandler`, `_SplitOpsHandler`, `_IntermediateOpHandler`),
the handler's `_name_map` (a mutable dict passed via default-arg
closure to the new `inner_fn`) *is* dependency-relevant state. Mutating
the map after `inner_fn` is installed silently changes what buffer
names the next `extract_read_writes` trace records — no re-assignment
of `data.inner_fn` occurs, so any per-op or per-buffer version stamp
that keys only on `id(inner_fn)` would miss it.

In practice every torch-spyre site builds a fresh `name_map` per
mutation and immediately re-installs a new `inner_fn` closure
(see section 3), so the dict is never mutated after installation. Still,
the invalidation contract must forbid post-install `name_map` mutation
as an invariant, not rely on it happening not to be done.

## 3. Torch-Spyre mutation sites of those fields

Every site in the pinned scan (`fea0c4be9…`) that touches a field
identified above. Grouped by mutation shape.

### 3.1 `object.__setattr__(op.data, "inner_fn", new_inner_fn)` — six sites

`ComputedBuffer.data` is a frozen dataclass, so torch-spyre uses
`object.__setattr__` to punch through `__setattr__`'s frozen guard.
Every one of these installs a wrapper-closing-over-old-`inner_fn`
pattern — the canonical "wrap, never rebuild" convention from
`CLAUDE.md`.

- `insert_restickify.insert_restickify_on_node_inputs` — patches
  `op.data.inner_fn` to route reads through `NameSwapHandler` after
  restickify insertion; then rebuilds `op` via a fresh `ComputedBuffer`
  constructor. `object.__setattr__(op.data, "inner_fn", new_inner_fn)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/insert_restickify.py:238`).
  Immediately followed by `ComputedBuffer.get_default_sizes_body.clear_cache(new_consumer_buffer)`
  (line 264).

- `dedup_constants._patch_inner_fn` — wraps `consumer.data.inner_fn` with
  a `NameSwapHandler` mapping duplicate constant names to the canonical
  one:
  `object.__setattr__(consumer.data, "inner_fn", _new_inner)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/dedup_constants.py:48`).
  Then `ComputedBuffer.get_default_sizes_body.clear_cache(consumer)`
  (line 49) — but the consumer op instance itself is *not* replaced;
  the memoized `op_read_writes` cache from `pass_utils` is *not*
  invalidated here. See section 4 for why this is a bug attractor.

- `padding._insert_bmm_padding` — wraps reduction's `inner_fn` to
  redirect the `y` load to a padded buffer:
  `object.__setattr__(reduction, "inner_fn", new_inner_fn)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/padding.py:151`).
  Immediately followed by `replace_computed_buffer_body` (line 154-160).

- `split_multi_ops._replace_original_op_body` — installs a
  `_SplitOpsHandler`+`_IntermediateOpHandler` chain, then calls
  `dataclasses.replace(op.data, inner_fn=new_inner_fn)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/split_multi_ops.py:678`).
  Followed by `replace_computed_buffer_body` (line 682-688). Note this
  site uses `dataclasses.replace` on `op.data`, not
  `object.__setattr__` — so it produces a **new `Loops` object** rather
  than mutating the existing one. Any per-`data` version stamp keyed
  on `id(op.data)` naturally invalidates; per-`op` stamps still need
  explicit bumping.

- `wsr/coarse_tile._patch_consumer_to_read_copy` — patches consumer's
  `inner_fn` with a `_NameSwapHandler` that rescales indices from
  full-buffer strides to tile-local strides:
  `object.__setattr__(consumer.data, "inner_fn", new_inner_fn)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3146`).
  Followed by `replace_computed_buffer_body` (line 3147-3153) and an
  explicit re-read (line 3168) that consumes the fresh
  `new_op.get_read_writes()`.

- `wsr/coarse_tile._patch_outside_consumers_to_full_buffer` — same
  pattern for outside consumers redirected to the full-sized buffer:
  `object.__setattr__(consumer.data, "inner_fn", new_inner_fn)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:3910`).

- `wsr/coarse_tile._patch_retiled_load_indexes` — installs a
  `_RetileLoadIndexHandler` that scales load-index coefficients:
  `object.__setattr__(op.data, "inner_fn", new_inner_fn)`
  (`torch-spyre@fea0c4b:torch_spyre/_inductor/wsr/coarse_tile.py:4290`).
  Followed by `replace_computed_buffer_body` (line 4291-4299).

### 3.2 `old_loop.data = new_loop` — scratchpad graph editor

The scratchpad `_replace_loop_input` rebuilds the entire `Loops` object
with a name-swapped `inner_fn`, then assigns the new object into
`old_loop.data` directly (no `object.__setattr__` needed because
`old_loop` — the ComputedBuffer-like wrapper here — is not the frozen
dataclass on this code path):

```
269    def _replace_loop_input(
270        self, old_loop: Operation, old_name: str, new_name: str
271    ) -> None:
272        """Replace one buffer load in a pointwise or reduction loop."""
273        assert isinstance(old_loop.data, Pointwise | Reduction)
274        new_loop = self._create_loop_hack_inner_fn(
275            old_loop.data, name_map={old_name: new_name}
276        )
277        old_loop.data = new_loop
278        # The dependency set changed; force the next query to retrace the loop.
279        invalidate_op_read_writes(old_loop)
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/scratchpad/graph_editor.py:269-279`)

This is currently the **only site** in the pinned scan that calls
`invalidate_op_read_writes` — every other inner_fn-mutation site relies
on `replace_computed_buffer_body` to build a fresh `ComputedBuffer`
whose `__dict__` is empty and thus contains no `_ts_cached_read_writes`
key (section 4.1).

### 3.3 `ComputedBuffer` replacement via `replace_computed_buffer_body`

Rather than mutating fields, this helper constructs a brand-new
`ComputedBuffer` and splices it into `graph.operations`:

```
1310 def replace_computed_buffer_body(
1311     op: ComputedBuffer,
1312     new_data: Loops,
1313     operations: list[Operation],
1314     *,
1315     pass_name: str,
1316     reason: str | None = None,
1317 ) -> ComputedBuffer:
...
1333     new_buf = ComputedBuffer(
1334         name=op.get_name(),
1335         layout=op.layout,
1336         data=new_data,
1337         _split_size=op._split_size,
1338         _original_inner_fn=op._original_inner_fn,
1339         _original_ranges=op._original_ranges,
1340         _original_reduction_ranges=op._original_reduction_ranges,
1341     )
1342     new_buf.operation_name = op.operation_name
1343     preserve_provenance(op, new_buf, pass_name=pass_name, reason=reason)
1344     copy_op_metadata(op, new_buf)
1345     ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)
1346
1347     op_idx = operations.index(op)
1348     operations[op_idx] = new_buf
1349     return new_buf
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:1310-1349`)

Called from six sites: `split_multi_ops.py:682`, `padding.py:154`,
`wsr/coarse_tile.py:3147, 3911, 4291, 3884` (import site). The freshness
of `new_buf` — a distinct object — is what implicitly invalidates
`pass_utils.op_read_writes`'s `__dict__`-keyed memo: the memo lives on
the old op, and no code reads back through the old handle after the
splice. That is a correctness-by-object-identity argument, not
enforced by any assertion, and is fragile the moment any pass holds a
reference to the old op across the splice.

### 3.4 `self.layout = new_layout` — many sites

Direct-assignment layout mutations (`_post_init_setattr` is *not* used
here because `layout` is on the non-frozen `Buffer` base). A partial
list, filtered to sites in `_inductor/`:

- `propagate_layouts.py:1589` — `producer.layout = copy_op.layout`
- `propagate_layouts.py:1726` — `tb.data.data.layout = new_layout`
- `propagate_layouts.py:2027, 2034, 2038` — `n.node.layout = ...`
- `insert_restickify.py:171, 192` — `restick_buff.layout =
  restick_arg_info["target_layout"]`
- `insert_restickify.py:313, 331, 360, 369, 402, 574, 584` — various
  `FixedTiledLayout` / `MutationLayoutSHOULDREMOVE` assignments
- `enforce_indirect_access_layout.py:209, 377, 381, 396`
- `pass_utils.py:1529, 1547` — pad buffer/const buffer layout finalization
- `wsr/coarse_tile.py:2292, 3721` — `full_buf.layout = layout`,
  `scalar_op.layout = FixedTiledLayout(...)`
- `spyre_kernel.py:798` — `tensor.layout = layout`

Per section 2.5, these change **write dep index** (via
`layout.as_fixed().make_indexer()`) but not reads. Any invalidation
contract that assumes "layout mutations don't matter" is wrong; the
pinned code today gets away with it only because no cached
`get_read_writes` result is queried post-layout-mutation before either
(a) `replace_computed_buffer_body` runs on the op, or (b) the op has
never entered `op_read_writes`'s memo yet. This is another
correctness-by-ordering that a contract should make explicit.

### 3.5 `WrapperHandler` installation sites (independent of field mutation)

For completeness, every place torch-spyre wraps `V.ops` via a
`WrapperHandler` around an `inner_fn`. Behaves *only* while the wrapped
`inner_fn` is on the call stack; not directly a mutation of a
`ComputedBuffer` field, but affects the trace `extract_read_writes`
records:

- `insert_restickify.NameSwapHandler` — `insert_restickify.py:76-91`
  (installed at line 235 inside the new `inner_fn` closure)
- `dedup_constants._patch_inner_fn` — `dedup_constants.py:44-45`
  (imports `NameSwapHandler` from `insert_restickify`)
- `wsr/coarse_tile._NameSwapHandler` — `wsr/coarse_tile.py:2487-2530`
  (installed at coarse_tile.py:3143)
- `scratchpad/graph_editor._NameSwapHandler` — `scratchpad/graph_editor.py:281-298`
- `scratchpad/passes._NameSwapHandler` — `scratchpad/passes.py:40-46`
- `split_multi_ops._SplitOpsHandler` — `split_multi_ops.py:63-89`
- `split_multi_ops._IntermediateOpHandler` — `split_multi_ops.py:92+`
  (installed at split_multi_ops.py:675)
- `scratchpad/utils._GetLoadStoreIndices` — `scratchpad/utils.py:524`

The `WrapperHandler` upstream contract itself (`_default` delegating to
`self._inner`, with `load` overridable) is
https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/_inductor/ops_handler.py#L1057-L1062.

## 4. Object identity vs generation — cache-key design tradeoffs

Torch-Spyre today uses `op.__dict__["_ts_cached_read_writes"]` — an
instance-keyed cache on the mutable `Buffer`/`OperationBuffer` base:

```
102 def op_read_writes(op: Operation) -> ReadWrites:
...
111     """
112     rw = op.__dict__.get("_ts_cached_read_writes")
113     if rw is None:
114         rw = op.get_read_writes()
115         op.__dict__["_ts_cached_read_writes"] = rw
116     return rw
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:102-116`)

Invalidation is manual, one call site:

```
130 def invalidate_op_read_writes(op: Operation) -> None:
131     """Drop any memoized :func:`op_read_writes` result for ``op``.
...
140     op.__dict__.pop("_ts_cached_read_writes", None)
```
(`torch-spyre@fea0c4b:torch_spyre/_inductor/pass_utils.py:130-140`)

Three axes to consider before any recommendation:

### 4.1 Per-op instance memo (today's approach)

- **Correctness by object identity**: freshly-constructed
  `ComputedBuffer` from `replace_computed_buffer_body` has an empty
  `__dict__`, so the memo is trivially fresh. This is *the* reason the
  five inner_fn-mutation sites in section 3.1 don't need to call
  `invalidate_op_read_writes` — they replace the op.
- **Fragility**: any pass that holds a reference to the *old* op
  handle after a splice reads a stale memo. `_ts_cached_read_writes`
  key is deliberately private ("a private key only this helper reads"
  — pass_utils.py:108-110), which contains the blast radius to
  `op_read_writes` callers, but does not eliminate it.
- **Dedup_constants gap**: `dedup_constants._patch_inner_fn` mutates
  `consumer.data.inner_fn` **without** calling
  `replace_computed_buffer_body` and **without**
  `invalidate_op_read_writes(consumer)`. It only clears
  `ComputedBuffer.get_default_sizes_body`'s cache (line 49). Any
  subsequent `op_read_writes(consumer)` returns a stale answer that
  still names the old (duplicate) constant. Whether this manifests as
  a bug depends on when `op_read_writes` is first called on that
  consumer — pre-dedup callers seed the memo. **OPEN QUESTION**:
  audit whether `op_read_writes` is ever called on a
  `dedup_constants` consumer between dedup and the next pass that
  rebuilds the op.

### 4.2 Per-op generation counter (`_spyre_analysis_version: int`)

Bump on every field mutation:

- Cache key becomes `(id(op), op._spyre_analysis_version)` or the
  generation is stored inside the memo tuple. A cheap `int` compare
  gates reuse.
- Requires an explicit `bump_analysis_version(op)` call at every
  mutation site — same number of touch-points as
  `invalidate_op_read_writes` today, no fewer.
- Advantage: makes the invariant *auditable* — a linter can require
  every `object.__setattr__(op.data, "inner_fn", ...)` and every
  `op.layout = ...` in `_inductor/` to be followed by a version bump,
  regardless of whether the immediately-next pass consumes
  `get_read_writes`.
- Disadvantage: still requires cooperation from every mutation site,
  and does nothing for scratchpad passes that reassign
  `old_loop.data = new_loop` (the memo key includes `id(op)`, not
  `id(op.data)`).

### 4.3 Graph-wide epoch counter (`V.graph._spyre_analysis_epoch: int`)

Every torch-spyre-owned pass entry point bumps the epoch; the memo
records `(epoch, ReadWrites)`. A stale memo is dropped on epoch
mismatch.

- Simplest invariant to state and enforce.
- Coarsest invalidation: any mutation anywhere in the graph
  invalidates every op's memo, defeating the "hundreds of calls"
  amortization the current memo is built for
  (`pass_utils.py:107-108`: "The scratchpad pass calls it hundreds of
  times, so we cache it under a private key").
- Would need supplementing with a per-op fast path — at which point
  it collapses back into option 4.2.

### 4.4 Recommendation posture (contract-level, not implementation)

- Adopt 4.2 (per-op generation counter) as the **normative** cache
  key. It survives `object.__setattr__(op.data, ...)` (bump `op`, not
  `op.data`), the scratchpad `old_loop.data = new_loop` path (same),
  and layout mutations.
- Provide the "instance-freshness" of `replace_computed_buffer_body`
  as an *implementation detail* — the new op's version starts at 0,
  which trivially misses the old op's memo. No behavior change from
  today, but the contract no longer relies on it.
- Enforce via a linter (see `.claude/skills/inductor-overview/SKILL.md`
  reference in section 3.5) that every mutation site listed in section
  3 is followed by a version bump.

## 5. Invalidation contract proposal

Not implementation — signatures and invariants only.

```python
# In torch_spyre/_inductor/pass_utils.py (or a new dependency_cache.py).

def cache_read_writes(op: Operation) -> ReadWrites:
    """Return op.get_read_writes(), memoized on (id(op), op._spyre_analysis_version).

    Callable from any pass that treats the op's dependency set as
    read-only for the duration of its own work. Callers that mutate any
    field enumerated in dependency-extraction.md §2 must call
    invalidate_read_writes(op) *before* the next cache_read_writes(op)
    call, or bump op._spyre_analysis_version themselves.

    Preserves today's private-key semantics: the memo lives under a
    single private slot on op.__dict__; no non-cache caller reads it.
    """


def invalidate_read_writes(op: Operation) -> None:
    """Drop any memoized cache_read_writes(op) result.

    Idempotent. Safe to call on an op that was never cached. Equivalent
    to a version bump followed by memo drop.
    """


def bump_analysis_version(op: Operation) -> None:
    """Bump op._spyre_analysis_version by one and drop the memo.

    Prefer this over invalidate_read_writes at any mutation site — it
    documents that a real IR field changed, not just "our cache went
    stale".
    """
```

**Property-setter interposition** (optional, tighter): install a
`__setattr__` interposer on `ComputedBuffer` in a torch-spyre-owned
subclass or via monkey-patch that auto-bumps the version whenever any
field listed in §2 is written. Sketch:

```python
_DEPENDENCY_FIELDS = frozenset({
    "data", "layout", "name",
    # Loops-child fields, matched by descending into op.data:
    "inner_fn", "ranges", "reduction_ranges", "reduction_type",
    "output_indexer",
})


def _bump_on_write(obj, name, value):
    object.__setattr__(obj, name, value)
    if name in _DEPENDENCY_FIELDS:
        parent = getattr(obj, "_spyre_owner_op", obj)
        parent.__dict__.pop("_ts_cached_read_writes", None)
        parent._spyre_analysis_version = (
            getattr(parent, "_spyre_analysis_version", 0) + 1
        )
```

The interposer is **defensive**, not primary: sites that go through
`object.__setattr__` (section 3.1) bypass it. Its value is in catching
future sites written with the naive `op.data.inner_fn = ...`, which
would otherwise be silently caught only by an explicit
`invalidate_read_writes` call — the failure mode this contract is
written to prevent.

**Invariant checklist for every mutation site** (linter rule):

1. Every `object.__setattr__(op.data, "inner_fn", ...)` must be
   followed by either (a) `replace_computed_buffer_body(op, ...)`, or
   (b) `bump_analysis_version(op)`.
2. Every `op.layout = ...` on a `ComputedBuffer` that is already in
   `graph.operations` must be followed by `bump_analysis_version(op)`
   *unless* the pass proves no downstream `cache_read_writes(op)` runs
   before the op is rebuilt. Prefer bumping unconditionally.
3. Every `old_loop.data = new_loop` (scratchpad path) must call
   `invalidate_read_writes(old_loop)` — matches
   `graph_editor.py:279` today.
4. `WrapperHandler`-installation via a fresh `inner_fn` closure is
   equivalent to inner_fn mutation for invariant purposes: bump
   applies at closure-installation time. Post-installation
   `name_map`/`_infos` dict mutation is forbidden — the contract
   assumes those dicts are constructed once and read-only thereafter.

## 6. Open questions

- **v2.13.0 vs v2.12.0 upstream diff**: all upstream citations in this
  contract used the locally-cached v2.12.0 torch. Before promoting
  status past `draft`, re-verify each line-number citation against a
  checked-out v2.13.0 tag (or the pinned pytorch main SHA
  `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`). Fields on
  `ComputedBuffer` and `Loops` are stable across 2.12→2.13 as far as
  the pinned scan's imports show, but line numbers will drift.

- **`dedup_constants._patch_inner_fn` stale-memo audit**: does any
  pass call `op_read_writes(consumer)` on a dedup consumer between
  `_patch_inner_fn` (which does *not* invalidate the memo) and the
  next full ComputedBuffer rebuild? If yes, this is a real bug, not
  just a contract gap.

- **Scatter `output_indexer` mutation policy**: no in-place assignment
  in the pinned scan, but the contract should still forbid it or
  require a version bump. Confirm no runtime-generated code path
  (Phase 5 audit) does this via `_post_init_setattr`.

- **Interaction with `get_default_sizes_body`'s own cache**: every
  torch-spyre mutation site clears
  `ComputedBuffer.get_default_sizes_body`'s cache alongside its
  `inner_fn` mutation. Should the invalidation contract fold that
  clear into `bump_analysis_version`, or keep them as independent
  invariants? Argument for folding: they never diverge in the pinned
  scan. Argument against: `get_default_sizes_body` depends on
  `data.ranges` and `data.reduction_ranges` but not directly on
  `data.inner_fn`'s recorded reads — the two caches genuinely answer
  different questions.

- **Scratchpad `_replace_loop_input` uses `old_loop.data = new_loop`
  directly (no `object.__setattr__`)**: implies the op on this path is
  *not* the frozen `ComputedBuffer` type. Confirm the exact op type
  (`SpyreKernelNode`? `SchedulerNode`? something scratchpad-local)
  before the interposer sketch in section 5 is turned into code.
