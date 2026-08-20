# `dedup_and_promote_constants` guard-vs-drop misalignment is unreachable

- **Category:** correctness
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** plausible
- **Status:** not-a-bug

## Summary

`_redirect_consumers` returns early when the duplicate constant's name
`D` is a graph output, but the outer caller unconditionally invokes
`_drop_constant` on the very next line. If the early-return path were
ever reached, `_drop_constant` would still remove `dup` from
`operations`, add `D` to `V.graph.removed_buffers`, and pop `D` from
`V.graph.name_to_buffer` — corrupting a live graph output. Static
analysis of every construction site of `SpyreConstantFallback` at
`fea0c4b` shows the state is not reachable today: the type is only
instantiated at four internal call sites (`lower_constant`,
`lower_full`, `split_multi_ops`'s constant branch, and
`coarse_tile.py`'s accumulator identity fill), and each one wires the
resulting node as an intermediate producer feeding a downstream op,
never as the value returned from `V.graph.graph_outputs`. The
invariant is real but nowhere asserted, so this is filed as
`not-a-bug` with a suggested defensive assertion rather than a fix to
`_drop_constant` itself.

## Files and symbols

- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `_redirect_consumers` (lines 52–79, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L52-L79>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `_drop_constant` (lines 82–106, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L82-L106>)
- torch-spyre: `torch_spyre/_inductor/dedup_constants.py` — `dedup_and_promote_constants` step 2 (lines 131–138, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/dedup_constants.py#L131-L138>)
- torch-spyre: `torch_spyre/_inductor/ir.py` — `SpyreConstantFallback.__init__` (lines 374–402, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/ir.py#L374-L402>)
- torch-spyre: `torch_spyre/_inductor/lowering.py` — `lower_constant` (lines 1421–1426, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/lowering.py#L1421-L1426>)
- torch-spyre: `torch_spyre/_inductor/lowering.py` — `lower_full` (lines 1385–1418, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/lowering.py#L1385-L1418>)
- torch-spyre: `torch_spyre/_inductor/split_multi_ops.py` — constant-materialization branch (lines 534–564, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/split_multi_ops.py#L534-L564>)
- torch-spyre: `torch_spyre/_inductor/coarse_tile.py` — accumulator identity fill (line 3549, permalink: <https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/coarse_tile.py#L3549-L3556>)
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
sites: no torch-spyre pass ever rewrites a graph-output FX node into a
`torch.ops.spyre.constant.default` call.

## Evidence

The producer type — `SpyreConstantFallback` — is instantiated in
exactly three places at `fea0c4b`:

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

`torch.ops.spyre.constant` is a torch-spyre-internal op. User models
do not call it. It is not a decomposition target of any public aten
op. It is only introduced by torch-spyre passes (see 2, 3, 4 below).

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
underlying `SpyreConstantFallback`.

**3. `split_multi_ops` constant branch in
`torch_spyre/_inductor/split_multi_ops.py:534-564`** — creates
`torch.ops.spyre.constant.default` FX nodes inside
`inserting_before(orig_node)`. Each new node is inserted before the
node that will consume it; the new nodes are then wired into the
consumer's environment as inputs. Never as an output.

**4. `coarse_tile.py:3549`** — creates a `SpyreConstantFallback`
directly (not via FX) and wires it as the fill source for an
accumulator's `Pointwise` fill. The `SpyreConstantFallback` is
consumed by `fill_data`, not returned to the graph.

Additionally, upstream lowerings that could reach the graph output
never target `SpyreConstantFallback` — the spyre `constant` op is not
in aten and no FX-graph rewrite before `dedup_and_promote_constants`
replaces an existing output-producing node with a `spyre.constant`
call.

## Reproducer or proof

**Impossibility proof (option (c) in the finding template).**

Claim: at the pinned SHA, when `_redirect_consumers` is invoked from
`dedup_and_promote_constants`, `D = dup.get_name()` is never in
`V.graph.get_output_names()`.

Argument:

1. `V.graph.get_output_names()` iterates `V.graph.graph_outputs` and
   returns each non-shape / non-`None` node's `.get_name()`
   (`torch/_inductor/graph.py:3167-3181` at
   `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`).
2. `V.graph.graph_outputs` is populated from the FX `output` node
   during `GraphLowering.run` and is not rewritten by any torch-spyre
   pass between then and `dedup_and_promote_constants` (checked by
   `grep` at `fea0c4b`: `graph_outputs` is assigned only by
   upstream Inductor, never in `torch_spyre/_inductor/*.py`).
3. Therefore the names in `get_output_names()` are exactly the
   `.get_name()` values of the IR nodes that upstream lowering placed
   at the FX `output` position.
4. `dup` is a `SpyreConstantFallback`. For `dup.get_name()` to be one
   of those names, some FX output node must lower to a
   `SpyreConstantFallback`. The `SpyreConstantFallback` type is
   constructed only at the four call sites in the Evidence section.
   None of the four wire the resulting node into `graph_outputs`:
   - `lower_constant` (site 1) is the lowering of a torch-spyre
     internal op that user code cannot call.
   - `lower_full` (site 2) wraps the `SpyreConstantFallback` in a
     `Pointwise`. `graph_outputs` receives the `Pointwise`.
   - `split_multi_ops` (site 3) and `coarse_tile` (site 4) create
     constants for internal consumers and do not rewrite
     `graph_outputs`.
5. Therefore `dup.get_name() ∈ get_output_names()` is unreachable at
   this call site. The guard in `_redirect_consumers` never fires;
   `_drop_constant` therefore never operates on a graph-output
   constant; the corrupted-output failure mode is not exhibited.

**Caveat.** The proof relies on a static enumeration of the four
construction sites and on the absence of any pass between
`GraphLowering.run` and `dedup_and_promote_constants` that rewrites
`graph_outputs`. Neither fact is expressed as an assertion in the
codebase. A future FX pass that legitimately produces a `spyre.constant`
node in an output-producing position, or a future upstream lowering
that added `torch.ops.spyre.constant.default` to a decomposition
graph and let it flow to the graph output, would silently break the
proof.

## Compile-time impact

None. This is a not-a-bug finding.

## Runtime impact

None. This is a not-a-bug finding.

## Correctness impact

None today. If the invariant were violated in a future pass, the
compiled program's return value would refer to a buffer with no
producer op (`operations.remove(dup)` already ran) and no
`name_to_buffer` entry — codegen would fail or emit an allocation
against a retired buffer name.

## Measurement needed (if any)

Not applicable.

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
guard in `_redirect_consumers` (dead by construction) and replaces it
with an assertion that fails loudly if the invariant ever breaks.

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
between "guard triggered" and "drop skipped" explicit.

Option A is preferred because it makes the invariant checkable and
the code path linear: nothing about the current implementation
correctly handles "graph output is a duplicate," and pretending it
does — as the current guard shape suggests — is worse than asserting
it never happens.

## Skill / contract update

Add a note to `contracts/` (or create
`contracts/graph-output-invariants.md`) documenting the invariant
this finding depends on:

> `SpyreConstantFallback` never appears in `V.graph.graph_outputs`.
> Any pass that introduces `torch.ops.spyre.constant.default` FX
> nodes must wire them as intermediates, not as elements of the FX
> `output` node's args. Any change that violates this invariant must
> also update `dedup_and_promote_constants` to safely handle a
> graph-output constant.

Also: the general lesson for the audit — "a guard that returns early
inside one of two paired helpers, with no reason given for why the
partner helper is safe to run anyway, is a signature that either the
guard is dead code or the partner helper is subtly wrong." Add this
as a reviewer heuristic under `cases/` if such a file exists, or
elevate it into the operating brief.
