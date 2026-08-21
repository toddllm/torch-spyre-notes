# Cross-torch-spyre private-API surface inventory

- **Category:** upstream-fragility
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** proven (static; counts and locations are the direct output of the Phase 1 `private_api` scanner over the pinned torch-spyre tree, cross-checked against pinned source)
- **Status:** open
- **Scope:** every torch-spyre `.py` file under `torch_spyre/`, not just `_inductor/patches.py`

## Summary

The Phase 1 scanner (`scans/private_api.py`) walked 101 torch-spyre
Python files and found **756 hits** at the boundary between
torch-spyre and PyTorch private surface (torch names beginning with a
`_`: `torch._inductor.*`, `torch._dynamo.*`, `torch._C._…`,
`torch.utils._…`, plus `# type: ignore[…]` suppressions co-located
with those attribute reads). The full machine-readable output is at
[scans/results/private_api.json](../../scans/results/private_api.json).

The surface is dominated by a small set of files. The top-10 hotspot
files carry **479 of the 756 hits (63%)**:

| Rank | File                                              | Hits  |
|------|---------------------------------------------------|------:|
| 1    | `torch_spyre/_inductor/lowering.py`               |  149  |
| 2    | `torch_spyre/_inductor/wsr/coarse_tile.py`        |   88  |
| 3    | `torch_spyre/_inductor/ir.py`                     |   51  |
| 4    | `torch_spyre/_inductor/patches.py`                |   35  |
| 5    | `torch_spyre/_inductor/scheduler.py`              |   33  |
| 6    | `torch_spyre/_inductor/spyre_kernel.py`           |   32  |
| 7    | `torch_spyre/_inductor/pass_utils.py`             |   30  |
| 8    | `torch_spyre/_inductor/wsr/propagate_named_dims.py` |   25  |
| 9    | `torch_spyre/_inductor/split_multi_ops.py`        |   21  |
| 10   | `torch_spyre/_inductor/hbm_pool_planning.py`      |   17  |

`patches.py` — the file the Phase 1 ledger already covered — is only
**4th on this list**. The three files above it (`lowering.py`,
`wsr/coarse_tile.py`, `ir.py`) together carry **288 hits (38% of the
total)** and are not surveyed by the patches ledger at all.

Breakdown by `kind` across all 756 hits:

| kind            | Count | What it is                                                                                              |
|-----------------|------:|---------------------------------------------------------------------------------------------------------|
| `attr`          |  387  | Attribute chain reads (`torch._inductor.ir.ExternKernel`, `V.graph.get_buffer`, …)                      |
| `import`        |  170  | `import torch._inductor.X` / `from torch._inductor.X import …`                                          |
| `type-ignore`   |   74  | `# type: ignore[…]` suppressions co-located with a private-name access (see mypy blind-spot section)    |
| `registry`      |   34  | Reads/writes of upstream registry dicts (`lowering.lowerings`, `inplaceable_ops`, `V.graph.name_to_buffer`, `pass_patterns`) |
| `getattr`       |   30  | `getattr(x, "_private", default)` — softer form of the same coupling                                    |
| `subclass`      |   28  | `class Spyre… (torch._inductor.<Private>Class): …` — inheritance from a private base                    |
| `__setattr__`   |   17  | `object.__setattr__(node, "…", …)` — dataclass field mutation past a frozen/private setter              |
| `dunder`        |   13  | `__closure__` / `__code__` / `__dict__` reads on upstream callables (used by `pass_utils.py` to inspect FX passes) |
| `setattr`       |    3  | `setattr(…, …)` — used sparingly for adding private markers on torch objects                            |

Two axes matter for fragility:

1. **`subclass`, `registry`, and `__setattr__` are the load-bearing
   couplings.** They mutate or extend upstream types by name, so a
   rename or a signature change breaks the site at import time or
   silently at first use, not at `mypy` time. `type-ignore` counts
   are misleading — a class inheriting from a moved base fails
   *before* type-checking runs.
2. **`import` counts scale with breadth of upstream modules touched.**
   Torch-spyre imports **37 distinct private torch modules**. The
   most-imported are `torch._inductor.ir` (37), `torch._inductor.virtualized`
   (26), `torch._inductor.graph` (20), and `torch._inductor.dependencies` (16).
   Every one of those is an upstream `_`-prefixed module with no
   compatibility contract.

The distinct-symbol count across all attribute chains is **247** — 247
different private names torch-spyre reads or writes.

## Hotspot walkthrough

Anchors are `file:line` against
`torch-spyre@fea0c4be901e1383b1f700dbad8887128b0fcb27`.

### 1. `torch_spyre/_inductor/lowering.py` (149 hits)

Overwhelmingly attribute reads (133 `attr`). The top private symbols:

- `torch._inductor.ir.Pointwise.create` (11) — direct calls to
  Inductor's `Pointwise.create` classmethod.
- `torch._inductor.lowering.lowerings` (9) — reads/writes of the
  process-global lowerings dict (this file also does 5 `registry`
  mutations of this dict inside `enable_spyre_lowerings`; see
  `03-lowering-registry-lock.md` for the mutation flow).
- `torch._inductor.ir.TensorBox.create` (9) and
  `torch._inductor.lowering.ops_wrapper` (8).

Registry mutations are at `lowering.py:201`, `:204`, `:230`, `:253`,
`:262` and target `lowering.lowerings`.

### 2. `torch_spyre/_inductor/wsr/coarse_tile.py` (88 hits)

Second-heaviest hotspot. The mix here is different from `lowering.py`:

- **23 `type-ignore[attr-defined]`** suppressions — this single file
  carries roughly a quarter of all `attr-defined` suppressions in
  the tree.
- 5 `__setattr__` calls at `:1284`, `:1380`, `:3146`, `:3910`,
  `:4290` — mutating `ranges`, `reduction_ranges`, and `inner_fn` on
  upstream Inductor IR nodes.
- 8 `registry` reads/writes of `V.graph.name_to_buffer` — this file
  reaches into Inductor's graph state to look up and (indirectly)
  register buffers.
- 3 `subclass` sites at `:2487`, `:4180`, `:4197`, all extending
  `torch._inductor.ops_handler.WrapperHandler`.

### 3. `torch_spyre/_inductor/ir.py` (51 hits)

- 8 `subclass` sites: 6 subclasses of
  `torch._inductor.ir.ExternKernel` (`:374`, `:405`, `:455`, `:519`,
  `:578`, `:631`), one of `Reduction` (`:45`), one of
  `FixedLayout` (`:86`).
- 5 `registry` sites, all constructing `OrderedSet` instances
  (`torch.utils._ordered_set.OrderedSet`) — these depend on the
  `torch.utils._ordered_set` module continuing to exist and export
  `OrderedSet`.
- 6 attribute reads of `torch._ops.OpOverload` (the private op
  overload class).

### 4. `torch_spyre/_inductor/patches.py` (35 hits)

Covered in detail by
[01-patches-ledger.md](01-patches-ledger.md); relevant here only for
scale — 35 hits, versus 149 in `lowering.py` and 288 across the top
three non-`patches.py` files. **Reading `patches.py` alone under-counts
torch-spyre's exposure by ~13×.**

### 5. `torch_spyre/_inductor/scheduler.py` (33 hits)

- 2 `subclass` sites: `FusedSchedulerNode` at `:80` and
  `BaseScheduling` at `:596`.
- 4 `set_kernel_handler` writes to `V.set_kernel_handler` (the
  virtualized kernel-handler slot).
- 4 `OrderedSet` constructor uses.

### 6. `torch_spyre/_inductor/spyre_kernel.py` (32 hits)

- 4 reads of `V.graph.get_buffer` (Inductor's private buffer table).
- 3 reads of `V.graph.sizevars.precomputed_replacements` — an
  extremely internal Inductor field.
- 3 uses of `V.graph.scheduler.mutation_real_name.get` — walking the
  scheduler's mutation-remapping table.
- 3 writes to `V.graph.removed_buffers.add` — informing Inductor
  which buffers Spyre has removed.
- 1 `subclass` of `torch._inductor.ops_handler.DefaultHandler` at `:416`.

### 7. `torch_spyre/_inductor/pass_utils.py` (30 hits)

The only file that reads Python-level dunders on upstream FX passes:

- 4 `__closure__` reads (getting captured names out of upstream
  passes).
- 3 `__dict__` reads.
- 2 `__code__` reads.

These couple pass_utils to the *implementation shape* of upstream FX
passes, not just their public callable signature — a refactor that
inlines a closure or moves state to `self.` breaks these reads
silently.

### 8. `torch_spyre/_inductor/wsr/propagate_named_dims.py` (25 hits)

- 10 `type-ignore[attr-defined]` suppressions.
- 7 `getattr(node, "_dim_prop_info", …)` reads — this file threads a
  Spyre-specific `_dim_prop_info` attribute through upstream FX nodes
  via `getattr` with defaults; the attribute is not defined by upstream
  and the reads are `attr-defined`-suppressed.

### 9. `torch_spyre/_inductor/split_multi_ops.py` (21 hits)

- 2 `subclass` sites extending
  `torch._inductor.ops_handler.WrapperHandler` (`:63`, `:92`).
- 3 `__setattr__` writes to `origins` on upstream IR nodes (`:567`,
  `:607`, `:681`).
- 3 reads/writes of `V.set_ops_handler` — swapping Inductor's
  op-handler mid-lowering.

### 10. `torch_spyre/_inductor/hbm_pool_planning.py` (17 hits)

- 5 reads of `V.graph.hbm_pool_sizes` — a Spyre-specific pool-sizes
  field that is registered onto `V.graph` at runtime, not defined by
  upstream. The `registry` hit at `:514` writes it.

## Registry, subclass, and __setattr__ inventory

The scanner also classifies the more consequential mutation kinds
outside `patches.py`. These are the couplings that break silently
under an upstream rename.

### 28 `subclass` sites (torch-spyre subclasses of private upstream classes)

Grouped by base class:

| Upstream private base                                     | Sites | Torch-spyre files                                                                                                                                                                          |
|-----------------------------------------------------------|------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `torch._inductor.ir.ExternKernel`                         |     6 | `_inductor/ir.py:374,405,455,519,578,631`                                                                                                                                                  |
| `torch._inductor.ops_handler.WrapperHandler`              |     9 | `_inductor/insert_restickify.py:76`, `_inductor/scratchpad/graph_editor.py:281`, `_inductor/scratchpad/passes.py:40`, `_inductor/scratchpad/utils.py:524`, `_inductor/split_multi_ops.py:63,92`, `_inductor/wsr/coarse_tile.py:2487,4180,4197` |
| `torch._inductor.ops_handler.DefaultHandler`              |     1 | `_inductor/spyre_kernel.py:416`                                                                                                                                                            |
| `torch._inductor.ir.Reduction`                            |     1 | `_inductor/ir.py:45`                                                                                                                                                                       |
| `torch._inductor.ir.FixedLayout`                          |     1 | `_inductor/ir.py:86`                                                                                                                                                                       |
| `torch._inductor.choices.InductorChoices`                 |     1 | `_inductor/choices.py:22`                                                                                                                                                                  |
| `torch._inductor.custom_graph_pass.CustomGraphPass`       |     1 | `_inductor/passes.py:160`                                                                                                                                                                  |
| `torch._inductor.custom_graph_pass.CustomSchedulerPass`   |     1 | `_inductor/passes.py:182`                                                                                                                                                                  |
| `torch._inductor.scheduler.FusedSchedulerNode`            |     1 | `_inductor/scheduler.py:80`                                                                                                                                                                |
| `torch._inductor.scheduler.BaseScheduling`                |     1 | `_inductor/scheduler.py:596`                                                                                                                                                               |
| `torch._inductor.codegen.wrapper.PythonWrapperCodegen`    |     1 | `_inductor/wrapper.py:31`                                                                                                                                                                  |
| `torch._dynamo.device_interface.DeviceInterface`          |     1 | `device/interface.py:32`                                                                                                                                                                   |
| `torch._dynamo.device_interface.DeviceInterface.Worker`   |     1 | `device/interface.py:50`                                                                                                                                                                   |
| `torch._inductor.codegen.common.DeviceOpOverrides`        |     1 | `device/op_overrides.py:20`                                                                                                                                                                |
| `torch._inductor.async_compile.AsyncCompile`              |     1 | `execution/async_compile.py:85`                                                                                                                                                            |

(Total: 28 sites across 15 distinct private base classes.)

### 34 `registry` mutations (writes into upstream registry state)

Distributed across:

- `torch._inductor.lowering.lowerings` — 5 sites in
  `_inductor/lowering.py:201,204,230,253,262` (see
  [03-lowering-registry-lock.md](03-lowering-registry-lock.md)).
- `torch._inductor.virtualized.V.graph.name_to_buffer` — 9 sites
  across `_inductor/insert_restickify.py`, `_inductor/split_multi_ops.py`,
  `_inductor/wsr/coarse_tile.py`.
- `torch._inductor.fx_passes.joint_graph.pass_patterns` —
  `_inductor/patches.py:171` (positional pop; see 01).
- `torch._inductor.fx_passes.post_grad.pass_patterns` —
  `_inductor/patches.py:186` (positional `[2]` index; see 01).
- `torch._inductor.fx_passes.reinplace.inplaceable_ops` —
  `_inductor/customops.py:395` (module-level write; see
  [04-monkey-patches-outside-patches-py.md](04-monkey-patches-outside-patches-py.md)).
- `torch._inductor.virtualized.V.graph.hbm_pool_sizes` —
  `_inductor/hbm_pool_planning.py:514`.
- `torch._inductor.codegen.common.Kernel` — `_inductor/spyre_kernel.py:423,543`.
- `torch.utils._ordered_set.OrderedSet` — 10 sites across
  `_inductor/ir.py` and `_inductor/scheduler.py` (constructor use of
  a private-utility class; not a mutation of state upstream owns,
  but same rename-fragility).

### 17 `__setattr__` sites (bypassing frozen-attribute protection)

- `object.__setattr__(node, "inner_fn", …)` — 6 sites: `_inductor/dedup_constants.py:48`, `_inductor/insert_restickify.py:238`, `_inductor/padding.py:151`, `_inductor/wsr/coarse_tile.py:3146,3910,4290`. This writes to `Loops.inner_fn` on an already-constructed IR node — upstream marking `Loops` as frozen would break these silently.
- `object.__setattr__(node, "origins", …)` — 4 sites: `_inductor/lowering.py:334`, `_inductor/split_multi_ops.py:567,607,681`.
- `object.__setattr__(node, "origin_node", …)` — 4 sites: `_inductor/pass_utils.py:1539`, `_inductor/provenance.py:294,317,351`.
- `object.__setattr__(node, "ranges", …)` / `"reduction_ranges"` — `_inductor/wsr/coarse_tile.py:1284,1380`.
- Dynamic name write in `_inductor/provenance.py:252`.

Every one of these calls the *unbound* `object.__setattr__` to force a
write past a possible descriptor or frozen-dataclass check — a form of
opting out of upstream's own invariant discipline.

## Static-checking blind spots at the upstream boundary

Torch-spyre's mypy configuration is deliberately permissive at the
private-torch boundary, and this is where the auditor should
calibrate expectations: **the type-checker will not warn about the
overwhelming majority of the fragility surface enumerated above.**

Anchor: `pyproject.toml` at
`torch-spyre@fea0c4b`.

### `[tool.mypy]` — repo-wide

`pyproject.toml:100–102`:

```toml
[tool.mypy]
ignore_missing_imports = true
exclude = ["tests/inductor", "captured"]
```

- `ignore_missing_imports = true` — mypy silently accepts any
  `import torch._inductor.X` regardless of whether the module
  actually exports the symbols torch-spyre expects; a missing
  import is not an error.
- `exclude = ["tests/inductor", "captured"]` — the entire
  `tests/inductor/` directory (which is where several of the
  Inductor-specific unit tests live) is exempt from type-checking.

### `[[tool.mypy.overrides]]` — the `_inductor` blanket suppression

`pyproject.toml:104–106`:

```toml
[[tool.mypy.overrides]]
module = ["torch_spyre._inductor.*"]
disable_error_code = ["attr-defined"]
```

**Scope:** every module under `torch_spyre/_inductor/` — 76 files.

**Effect:** mypy will not raise `attr-defined` in any of them. The
scanner counts **37 explicit `type: ignore[attr-defined]` sites**
(see the `type-ignore` kind row above), but the blanket override
means the *implicit* `attr-defined` suppression is repo-wide across
`_inductor`. Any private attribute that changes name or shape in
upstream torch will pass mypy silently for the entire subtree that
holds 6 of the 10 hotspot files (`lowering.py`, `wsr/coarse_tile.py`,
`ir.py`, `patches.py`, `scheduler.py`, `spyre_kernel.py`,
`pass_utils.py`, `wsr/propagate_named_dims.py`, `split_multi_ops.py`,
`hbm_pool_planning.py`).

### No pyright / pyre / pytype configuration

No `pyrightconfig.json`, `pyproject`-level `[tool.pyright]`,
`[tool.pyre]`, `[tool.pytype]`, `mypy.ini`, `.mypy.ini`, or
`setup.cfg` type-check config was found in the pinned tree (checked
`find fea0c4b -maxdepth 3` for those names — no matches). Type
checking is mypy-only, on the configuration above.

### Distribution of the 74 `type: ignore` co-located hits

The 74 `type-ignore` scanner hits (co-located with private-name
attribute reads) break down as:

| Ignore code            | Count | Meaning                                                                                              |
|------------------------|------:|------------------------------------------------------------------------------------------------------|
| `attr-defined`         |    37 | Reading an attribute mypy cannot see on the target type.                                             |
| `empty-body`           |     9 | Custom-op stub functions in `_inductor/customops.py` with `pass`/no body (schema declared via decorator). |
| `method-assign`        |     5 | Assigning to a method slot on an upstream class (four in `_inductor/patches.py` and one in `model_utils.py`). |
| `assignment`           |     4 | Assigning through a computed target mypy considers incompatible.                                     |
| `union-attr`           |     3 | Attribute access on a `Optional[X]` that hasn't been narrowed.                                       |
| `override`             |     2 | A subclass method signature that doesn't line up with the private-base signature (`ir.py:58`, `scheduler.py:104`). |
| Bare `# type: ignore`  |    14 | Un-coded blanket suppressions: 7 in `_inductor/scratchpad/*` (`graph_editor.py:44,132`, `simulated_annealing.py:381`, `permutation_layout.py:304,460,476,1323`) plus 7 in `ops/eager.py:453,463,482,498,511,518,532`. |

The 5 `method-assign` suppressions are the most consequential — they
each mark a monkey-patch of an upstream method slot as a known
signature mismatch. `_inductor/patches.py:130,159,173,174` are the
`GraphLowering._update_scheduler` and `SchedulerNode.has_side_effects`
swaps documented in [01-patches-ledger.md](01-patches-ledger.md).
`model_utils.py:365` is the `nn.Module.to` override documented in
[04-monkey-patches-outside-patches-py.md](04-monkey-patches-outside-patches-py.md).

### What the type-checker will and will not catch

Combining the two configuration facts above, the static-checker
posture is:

- ✗ Any `torch._inductor.foo.bar` read that starts failing after a
  torch upgrade — silent under the `_inductor` blanket suppression.
- ✗ A rename of `torch._inductor.ir.ExternKernel` — the 6
  subclasses in `_inductor/ir.py` fail at import, not at type-check.
- ✗ `object.__setattr__(node, "inner_fn", …)` after upstream marks
  `Loops` frozen — no static signal (unbound `object.__setattr__`
  bypasses descriptor protocols and is invisible to mypy anyway).
- ✗ A signature drift of a swapped method (`_update_scheduler`,
  `has_side_effects`, `nn.Module.to`) — suppressed by
  `method-assign` marker.
- ✓ A missing attribute on a stdlib or fully-typed torch public
  symbol outside `_inductor/` — mypy will still complain (the
  override is scoped to `torch_spyre._inductor.*`).

## Machine-readable inventory

The full 756-row hit table is at
[scans/results/private_api.json](../../scans/results/private_api.json).
Each row carries `file`, `line`, `kind` (one of the 9 above),
`private_name` (the dotted path), `snippet` (three lines of source
context), and `context` (a short label). Re-running the scanner is:

```
python3 scans/private_api.py
```

which reads `/tmp/ts-pinned-scan/fea0c4b/torch_spyre/**/*.py` at the
pinned SHA and rewrites `scans/results/private_api.json`.

## What this changes about the audit story

Prior state: the Phase 1 ledger implied `patches.py` was the
main torch-spyre↔upstream coupling site.

Post-inventory state: `patches.py` is one of ~10 hotspot files and
carries under 5% of the total private-API surface. Any future
"one-file compat sweep" that touches only `patches.py` is a
completeness fiction. The `_inductor/lowering.py` +
`_inductor/wsr/coarse_tile.py` + `_inductor/ir.py` triad is the load
that the type-checker cannot see, because of the two-line blanket
mypy override at `pyproject.toml:104-106`.
