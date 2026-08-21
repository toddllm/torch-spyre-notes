# `dedup_and_promote_constants` guard-vs-drop misalignment: not observed under lowering paths examined, direct custom-op emission untested

- **Id:** COR-01
- **Category:** correctness
- **Created:** 2026-08-20
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** plausible
- **Status:** not-observed

## Summary

`_redirect_consumers` returns early when the duplicate constant's name
`D` is a graph output, but the outer caller unconditionally invokes
`_drop_constant` on the very next line. If the early-return path were
ever reached, `_drop_constant` would still remove `dup` from
`operations`, add `D` to `V.graph.removed_buffers`, and pop `D` from
`V.graph.name_to_buffer` — corrupting a live graph output.

A previous version of this finding claimed an "impossibility proof"
resting on the assertion that `torch.ops.spyre.constant` is internal
and user code cannot call it. At the pinned SHA that assertion is
false: `spyre::constant` is registered as a **public**
`torch.library.custom_op` at `torch_spyre/_inductor/customops.py:659`,
so a user model — or a decomposition target — could legitimately place
`torch.ops.spyre.constant.default(...)` in a graph-output position,
and its registered lowering `lower_constant` produces exactly the
`SpyreConstantFallback` that dedup step 2 processes. The status is
downgraded to **not-observed**: none of the lowering paths and passes
enumerated in the audit at `fea0c4b` produce this state, but direct
user-level `torch.ops.spyre.constant.default(...)` at the graph output
has not been adversarially tested and is the natural repro candidate.

## Files and symbols

- torch-spyre: `torch_spyre/_inductor/customops.py` — `spyre_constant` public custom op registration (lines 659–668, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/customops.py#L659-L668>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `_redirect_consumers` (lines 52–79, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L52-L79>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `_drop_constant` (lines 82–106, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L82-L106>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `dedup_and_promote_constants` step 2 (lines 131–138, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L131-L138>)
- torch-spyre: `torch_spyre/_inductor/ir.py` — `SpyreConstantFallback.__init__` (lines 374–402, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/ir.py#L374-L402>)
- torch-spyre: `torch_spyre/_inductor/lowering.py` — `lower_constant` (lines 1421–1426, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/lowering.py#L1421-L1426>)
- torch-spyre: `torch_spyre/_inductor/lowering.py` — `lower_full` (lines 1385–1418, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/lowering.py#L1385-L1418>)
- torch-spyre: `torch_spyre/_inductor/split_multi_ops.py` — constant-materialization branch (lines 534–564, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/split_multi_ops.py#L534-L564>)
- torch-spyre: `torch_spyre/_inductor/wsr/coarse_tile.py` — accumulator identity fill (line 3716, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/wsr/coarse_tile.py#L3716>)
- upstream main: `torch/_inductor/graph.py` — `GraphLowering._get_output_names` (lines 3167–3181, permalink: <https://github.com/pytorch/pytorch/blob/c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62/torch/_inductor/graph.py#L3167-L3181>)

## Observed behavior

Inside the step-2 dedup loop, for each duplicate `dup` in a
value-group, the caller does:

```python
for dup in group[1:]:
    _redirect_consumers(operations, dup, canonical)
    _drop_constant(operations, dup, canonical)
```

`_redirect_consumers` guards a specific edge case:

```python
D = dup.get_name()
C = canonical.get_name()
name_map = {D: C}

# Do not dedup a constant that is itself a graph output.
if D in V.graph.get_output_names():
    logger.debug("dedup_and_promote_constants: skipping output constant %s", D)
    return
```

The guard bails out of the consumer-rewrite phase. The very next line
in the caller then runs `_drop_constant(operations, dup, canonical)`
regardless of whether the guard fired. `_drop_constant`'s body:

```python
merge_provenance(
    [canonical, dup],
    canonical,
    pass_name="dedup_and_promote_constants",
    reason="duplicate constant",
)
operations.remove(dup)
V.graph.removed_buffers.add(D)
V.graph.name_to_buffer.pop(D, None)
V.graph.name_to_op.pop(op_name, None)
# Merge the duplicate's users into the canonical's user list so that passes
# which iterate name_to_users (e.g. scratchpad planning) see the full set.
extra_users = V.graph.name_to_users.pop(D, [])
if extra_users:
    V.graph.name_to_users.setdefault(C, []).extend(extra_users)
```

`operations.remove(dup)` deletes the producer op. `removed_buffers.add(D)`
tells downstream code the buffer is dead. `name_to_buffer.pop(D, None)`
removes the buffer lookup entry. Meanwhile `V.graph.graph_outputs`
(the list of IR nodes returned by the compiled function) is not
touched — it still holds a reference to `dup`. `get_output_names()`
therefore continues to report `D` in its output-name list, but that
output now refers to a buffer with no live producer and no
`name_to_buffer` entry, and marked for removal. The compiled program's
return contract would be broken.

## Upstream behavior

- **v2.13.0 (supported baseline):** `GraphLowering.graph_outputs` is
  set from the FX `output` node during `run_node`; it holds direct
  references to the `IRNode` objects returned by lowered ops.
  Upstream never rewrites `graph_outputs` after
  `GraphLowering.run` finishes and would not know that a downstream
  torch-spyre pass has retired one of those references.
- **main:** unchanged for the surfaces this finding depends on;
  `_get_output_names` is at
  `torch/_inductor/graph.py:3167-3181` and iterates `graph_outputs`,
  calling `node.get_name()` for each non-`None`/non-shape entry. No
  cross-check against `removed_buffers` is performed.

## Hidden assumption or duplicated knowledge

The guard-vs-drop pair encodes an implicit contract:

> "The path `_redirect_consumers` early-returns from is unreachable at
> this call site, so `_drop_constant` does not need a symmetric guard."

Nowhere in the file is that contract stated, and nowhere is it
asserted. The invariant it depends on is deeper:

> "A `SpyreConstantFallback` op is never assigned to
> `V.graph.graph_outputs` at the point `dedup_and_promote_constants`
> runs (pre-scheduling, after `insert_bmm_padding`)."

That invariant is enforced only by inspection of the construction
sites, and — critically — by nothing preventing a user model from
directly emitting `torch.ops.spyre.constant.default(...)` as a graph
output. The public custom-op registration means a torch-spyre user
who imports the ops namespace can produce exactly that shape.

## Evidence

### Public custom-op registration (new)

At `torch_spyre/_inductor/customops.py:659`:

```python
@torch.library.custom_op("spyre::constant", mutates_args=(), device_types="spyre")
def spyre_constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    # This custom operator marks scalar constant in the FX graph.
    ...
    return fill_value
```

`torch.library.custom_op` registers a public op in the `spyre::`
namespace, reachable from Python as `torch.ops.spyre.constant.default`.
No visibility gate protects it: any user program that imports
`torch_spyre` (which registers the op at module load) can call
`torch.ops.spyre.constant.default(v, dtype, torch.device("spyre"))`
inside a `@torch.compile`d function. The prior claim that "user code
cannot call it" is refuted by this registration.

### Construction sites of `SpyreConstantFallback`

The producer type — `SpyreConstantFallback` — is instantiated in
exactly four places at `fea0c4b`:

**1. `lower_constant` in `torch_spyre/_inductor/lowering.py:1421-1426`**
— the registered `type_promotion_kind=None` lowering for
`torch.ops.spyre.constant.default`:

```python
@register_spyre_lowering(torch.ops.spyre.constant.default, type_promotion_kind=None)
def lower_constant(value, dtype, device):
    op_overload = getattr(
        torch.ops.spyre.constant, V.graph.current_node.target._overloadname
    )
    return ir.TensorBox.create(SpyreConstantFallback(op_overload, value, dtype, device))
```

If a user emits `torch.ops.spyre.constant.default(v, dtype, device)`
directly at the graph output, this lowering returns a `TensorBox`
wrapping a `SpyreConstantFallback` — which is exactly the shape
`graph_outputs` would then hold. The `Pointwise` wrapper in
`lower_full` does NOT apply here: `lower_full` and `lower_constant`
are distinct lowerings for distinct FX nodes.

**2. `lower_full` in `torch_spyre/_inductor/lowering.py:1385-1418`**
— the registered lowering for `torch.ops.aten.full.default`:

```python
scalar = ir.TensorBox.create(
    SpyreConstantFallback(
        torch.ops.spyre.constant.default, float(fill_value), dtype, device
    )
)
scalar_loader = scalar.make_loader()

def inner_fn(index):
    return scalar_loader([])

return Pointwise.create(
    device=device,
    dtype=dtype,
    inner_fn=inner_fn,
    ranges=list(size),
)
```

The `SpyreConstantFallback` produced here is a 0-d scalar. The value
`lower_full` returns is a `Pointwise` (a `ComputedBuffer`) whose
`inner_fn` loads from that scalar. When a user model has
`return torch.full(size, v)` at the graph output, what lands in
`graph_outputs` is the `Pointwise` `ComputedBuffer`, not the
underlying `SpyreConstantFallback`. This path does not produce the
bug shape.

**3. `split_multi_ops` constant branch in
`torch_spyre/_inductor/split_multi_ops.py:534-564`** — creates
`torch.ops.spyre.constant.default` FX nodes inside
`inserting_before(orig_node)`. Each new node is inserted before the
node that will consume it; the new nodes are then wired into the
consumer's environment as inputs. Never as an output.

**4. `torch_spyre/_inductor/wsr/coarse_tile.py:3716`** — creates a
`SpyreConstantFallback` directly (not via FX) and wires it as the
fill source for an accumulator's `Pointwise` fill. The
`SpyreConstantFallback` is consumed by `fill_data`, not returned to
the graph.

### What is *not* covered by the audit

Site (1) is only exercised by the audit as a lowering target for
inserts made by sites (3) and internally by other passes. Nothing
static enumerates whether a user model directly emits
`torch.ops.spyre.constant.default(...)` at an output-producing position.
Because the custom-op registration is public, that path is not
prevented by construction — only by the observation that no upstream
aten op decomposes to `spyre::constant`, and no test in the audited
tree exercises the "user calls `torch.ops.spyre.constant.default` as
the graph return" shape.

## Reproducer or proof

**Not observed under lowering paths examined; direct custom-op
emission is the natural repro candidate.**

Under the four construction sites enumerated above, no path lands a
`SpyreConstantFallback` in `V.graph.graph_outputs`:

- `lower_full` (site 2) inserts a `Pointwise` wrapper between the
  constant and the graph output.
- `split_multi_ops` (site 3) and `coarse_tile` (site 4) create
  constants for internal consumers and do not rewrite `graph_outputs`.
- No upstream aten op is known to decompose to `spyre::constant`.

The path that has *not* been ruled out is a user program (or a future
decomposition, or a future pass) that emits `torch.ops.spyre.constant`
in an output-producing FX position. Because the op is a public
`torch.library.custom_op` and its lowering `lower_constant` returns a
`TensorBox(SpyreConstantFallback(...))` unwrapped, the resulting IR
node's `.get_name()` would be exactly the name inserted into
`graph_outputs` for that output slot. If a second output of the same
function emits the same constant (same value/dtype/device), dedup
step 2 would run `_redirect_consumers` on the duplicate, the guard
would fire, `_drop_constant` would run anyway, and the compiled
program's return contract would refer to a retired buffer.

### Reproduction candidate

```python
import torch
import torch_spyre  # registers spyre::constant

def f(x):
    # Two calls with identical (value, dtype, device) — dedup should
    # merge them. Returning both directly places SpyreConstantFallback
    # in graph_outputs for both slots at lowering time.
    c1 = torch.ops.spyre.constant.default(
        1.0, torch.float16, torch.device("spyre")
    )
    c2 = torch.ops.spyre.constant.default(
        1.0, torch.float16, torch.device("spyre")
    )
    return c1, c2, x + c1

compiled = torch.compile(f, backend="spyre")
x = torch.zeros((4,), dtype=torch.float16, device="spyre")
out = compiled(x)
```

Expected observation if the bug fires: after
`dedup_and_promote_constants`, one of `{c1, c2}` is dropped from
`V.graph.name_to_buffer` while still referenced by `graph_outputs`;
codegen either fails or allocates against a retired buffer name.

**Caveats before treating this as a live bug:**

1. It is not yet known whether `torch.compile`'s Dynamo layer folds
   `torch.ops.spyre.constant.default(1.0, ...)` at trace time (the op
   returns a scalar-valued number, not a tensor; that shape may not
   even survive Dynamo).
2. The custom-op's return type is `torch.types.Number`, not a
   tensor — Inductor's output-name handling for scalar returns may
   route around `SpyreConstantFallback` entirely and place a
   `SymFloat`-shaped output.
3. `lower_constant`'s current call sites all pass through
   `V.graph.current_node.target._overloadname`; a direct user call
   with the `.default` overload may hit a different code path.

Any of (1)–(3) could turn this into a genuine impossibility again.
The point of the reclassification is that none of (1)–(3) has been
verified at the pinned SHA, and the previously stated proof rested
on a false premise ("user code cannot call the op").

## Compile-time impact

None observed on the audited paths. If the bug fires under the
reproduction candidate, compile fails or emits a corrupt program.

## Runtime impact

None observed on the audited paths. If the bug fires and codegen
still succeeds, the returned value refers to a retired buffer.

## Correctness impact

None observed on the audited paths. If the invariant is violated in
a future pass or by user code, the compiled program's return value
would refer to a buffer with no producer op (`operations.remove(dup)`
already ran) and no `name_to_buffer` entry — codegen would fail or
emit an allocation against a retired buffer name.

## Measurement needed (if any)

Run the reproduction candidate under a Spyre-capable environment
(see [`needs-pod/01-constant-graph-output-repro.sh`](../../needs-pod/01-constant-graph-output-repro.sh))
and capture:

1. Whether `torch.compile(f, backend="spyre")` reaches
   `dedup_and_promote_constants` with two `SpyreConstantFallback`
   entries in `graph_outputs`, or whether an earlier layer (Dynamo
   constant folding, output-type routing) prevents that shape.
2. If (1) is reached: whether the compile fails, produces a working
   program, or produces a program that crashes at runtime on the
   returned buffer.

## Suggested change

Turn the invariant into an assertion at the caller in
`dedup_and_promote_constants`. Two options:

**Option A — hard assertion at entry to step 2:**

```python
for dup in group[1:]:
    assert dup.get_name() not in V.graph.get_output_names(), (
        f"dedup_and_promote_constants: constant {dup.get_name()!r} is a "
        f"graph output; _drop_constant would corrupt the return contract"
    )
    _redirect_consumers(operations, dup, canonical)
    _drop_constant(operations, dup, canonical)
```

This removes the dead `if D in V.graph.get_output_names(): return`
guard in `_redirect_consumers` (dead in the observed paths) and
replaces it with an assertion that fails loudly if the invariant ever
breaks. If the reproduction candidate above shows the invariant *can*
break, replace the assertion with a correct handler (skip both
`_redirect_consumers` and `_drop_constant`, or preserve the buffer
under the canonical name).

**Option B — return a status from `_redirect_consumers` and let the
caller decide:**

```python
def _redirect_consumers(...) -> bool:
    ...
    if D in V.graph.get_output_names():
        return False
    for op in operations: ...
    return True

for dup in group[1:]:
    if _redirect_consumers(operations, dup, canonical):
        _drop_constant(operations, dup, canonical)
```

This preserves the current defensive shape but makes the pairing
between "guard triggered" and "drop skipped" explicit. Option B is
now the preferred fix (rather than Option A) because the audit no
longer proves the guard is dead — it only proves it is unreached on
the paths enumerated.

## Skill / contract update

Add a note to `contracts/` (or create
`contracts/graph-output-invariants.md`) documenting the invariant
this finding depends on:

> `SpyreConstantFallback` never appears in `V.graph.graph_outputs`.
> Any pass that introduces `torch.ops.spyre.constant.default` FX
> nodes must wire them as intermediates, not as elements of the FX
> `output` node's args. Because `spyre::constant` is a public
> `torch.library.custom_op`, this invariant depends on user models
> not calling it directly at output positions — a contract that is
> currently unenforced. Either enforce it (reject the op at
> `graph_outputs`) or make `dedup_and_promote_constants` safely
> handle a graph-output constant.

Also: the general lesson for the audit — "a guard that returns early
inside one of two paired helpers, with no reason given for why the
partner helper is safe to run anyway, is a signature that either the
guard is dead code or the partner helper is subtly wrong." Add this
as a reviewer heuristic under `cases/` if such a file exists, or
elevate it into the operating brief. And: "an impossibility proof
that rests on `X is internal and user code cannot call it` must cite
the visibility gate; a `torch.library.custom_op` registration is
public even if the surrounding module treats the op as private."
