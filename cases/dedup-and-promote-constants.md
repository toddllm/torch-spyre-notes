# Case study: `dedup_and_promote_constants`

A first-principles walkthrough of one loop-level IR pass that
eliminates redundant runtime work — as opposed to a pass whose job is
to rewrite compiler state safely.

- **File:** `torch_spyre/_inductor/dedup_constants.py`
- **Pass entry point:** `dedup_and_promote_constants`
- **Pinned SHA:** `2b3ca93f12cc2571031c63514b723bc54aa55703`
- **Permalink to the whole file:**
  <https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py>
- **Permalink to the entry point:**
  <https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py#L109-L151>

Companion document:
[`replace-computed-buffer-body.md`](./replace-computed-buffer-body.md)
walks through a helper whose job is to keep compiler-internal
invariants consistent during a mutation. That helper does not change
what the compiled program executes; it changes how the compiler's
own state is maintained. This document walks through a pass whose job
is different: it removes work that the compiled program would
otherwise perform.

---

## 1. What layer of the compiler this touches

A modern deep-learning framework compiles a Python program in stages.
Each stage lowers the program to a representation with fewer
abstractions and more explicit machine detail. A common progression:

1. **Traced graph.** The Python model is traced into a graph of
   framework-level operations (matmul, add, softmax, view, etc.).
2. **Functional graph.** Autograd and functionalization rewrite the
   traced graph so it is side-effect-free and differentiable through
   pure functions.
3. **Loop-level IR.** The functional graph is lowered into loops —
   each tensor-producing operation becomes a description of *how* to
   compute each element as a function of loop indices, plus a
   description of *what* is read from other buffers.
4. **Scheduler.** The loop-level IR is grouped into kernels, fused
   where safe, and ordered for execution.
5. **Backend codegen.** Kernels are emitted as target code — CUDA
   source, C++, a device-specific spec, etc.

An out-of-tree backend that wants to reuse the upstream compiler's
tracing, autograd, decomposition, symbolic-shape handling, and
lowering work generally hooks in at stage 3 or 4. That is where the
compiler has already done the hard work of turning Python into
indexed loop bodies but has not yet committed to any target-specific
decisions about tiling, memory placement, or code emission.

The pass in this document runs at stage 3, immediately before the
scheduler is constructed. It is part of a family of *loop-level IR
transformation passes* that rewrite the graph after upstream lowering
but before upstream scheduling. Unlike the helper walked through in
the companion document, this pass's *purpose* is to change what the
compiled program does at runtime, not merely to keep the compiler's
internal state coherent.

---

## 2. What a loop-level IR node actually contains

At stage 3, the natural unit of the IR is roughly:

- A **buffer** — a named tensor-shaped value, with a layout (shape,
  strides, dtype, device).
- An **operation** — a computation that produces one or more buffers.
- A **computed buffer** — the common case where a single operation
  produces a single buffer whose contents are defined by a loop body.

The loop body inside a computed buffer is not stored as text or as a
static syntax tree. It is stored as a **callable** — an ordinary
Python function whose parameters are the loop indices and whose body
issues abstract operations (`load`, `store`, `add`, `where`, …)
against an ambient handler object. Roughly:

```python
def inner_fn(index):
    i0, i1 = index
    x = ops.load("buf17", 1024 * i0 + i1)
    y = ops.load("buf18", 1024 * i0 + i1)
    return ops.add(x, y)
```

The `ops` object is not fixed. Different passes install different
handlers so that calling the *same* `inner_fn` produces different
results:

- Dependency extraction installs a handler that records which buffers
  are read and at which symbolic indices.
- Code generation installs a handler that emits target instructions.
- A rewrite pass may install a handler that transparently renames
  loads.

This design matters for the pass walked through below in one specific
way: rewriting a consumer to read from a *different* buffer is done by
installing a name-swapping handler over the consumer's existing
`inner_fn`, not by rebuilding the function. That mechanism is
explained in section 4.2.

---

## 3. Why a graph can end up with identical constants

Before looking at the pass itself, it helps to understand why the
condition it exists to fix arises in the first place. Nothing in the
upstream compiler is trying to produce duplicates. They emerge as an
interaction of three independent facts about how lowering works.

**Constants are materialized close to their users.** When a
framework-level operation needs a scalar or small tensor constant —
padding value, mask value, identity element of a reduction, sentinel
for a `where` — that constant is emitted as its own operation in the
loop-level IR. It has a name, a layout, and a producer. It behaves
like any other buffer: it occupies device memory, it has to be
written before it is read, and consumers refer to it by name.

**Lowering is per-node and per-decomposition.** The traced graph
contains many higher-level operations, each of which may lower into
several loop-level operations. Two `aten.constant_pad_nd` calls with
the same fill value, or two `where` calls with the same identity
element, each produce their own constant-materialization operation as
part of their local lowering. Neither lowering step consults the
other — they run in isolation over their own subgraphs.

**Upstream Inductor has a general-purpose common-subexpression
mechanism, but it does not deduplicate at this level.** The compiler
performs many kinds of pattern-based and post-grad transformations,
but the specific case of "two structurally identical constant
materializations produced by independent lowering steps" is not
handled by the upstream mechanisms that reach the out-of-tree
backend.

The consequence is that a model with, say, a dozen padded operations
and a dozen masking operations enters stage 3 with a dozen copies of
the pad-value constant and a dozen copies of the mask-value constant.
Each copy is a distinct buffer with a distinct name and a distinct
materialization operation, and each will otherwise flow all the way
through the scheduler and backend into the compiled program.

---

## 4. What that costs, and what the pass changes

This is the part that makes this pass different from the one in the
companion document. If nothing eliminates the duplicates, the compiled
program pays for them — every one of them — every time it runs.

### 4.1 The runtime cost of leaving duplicates in place

Each duplicate constant that survives to codegen carries three
distinct runtime costs:

**Device memory.** Every duplicate constant is allocated its own
device-side buffer. For a small scalar, this is negligible per copy
but is paid once per copy per model instance. For a larger tensor
constant — a fixed embedding, an attention bias table, a lookup
table — the per-copy cost is non-negligible, and the total scales
with the number of independent lowering sites that produced the same
value.

**Kernel launches and fill work.** Each duplicate constant is
produced by its own operation. That operation becomes its own kernel
launch (or its own inclusion in a fused kernel), issuing enough work
to fill the duplicate's buffer with a value that is bit-identical to
what another buffer already contains. On an accelerator this is
kernel-launch overhead, dispatch overhead, and either DMA traffic or
compute cycles depending on how the constant is produced.

**Cache and bandwidth pressure downstream.** Consumers that read
duplicate constants read from different device addresses that hold
identical bytes. The device's cache hierarchy cannot tell those are
"the same value" and cannot share cache lines between them. Reading
one canonical constant from many consumers is a friendlier access
pattern than reading N identical constants from N different
addresses.

None of these costs would appear as a bug. The compiled program
produces the correct answer. It is simply doing more work than it
needs to.

### 4.2 The transformation the pass performs

The pass replaces the duplicated-materialization pattern with a
single-materialization pattern. Concretely, for each set of
structurally identical constant-producing operations, the pass:

1. Picks one of them as the *canonical* operation.
2. Rewrites every consumer of every non-canonical duplicate to load
   from the canonical buffer name instead.
3. Removes each non-canonical duplicate from the operation list.
4. Retires each duplicate's buffer name via the graph's logical
   deletion mechanism.
5. Moves all surviving constant operations to the head of the
   operation list so they are guaranteed to be materialized before
   any consumer runs.

After the pass, the loop-level IR contains one buffer per distinct
constant value instead of one per lowering site. The scheduler and
backend see fewer operations, emit fewer kernels, allocate less
device memory, and produce a compiled program that does strictly
less work at runtime for the same numerical result.

The remainder of this section walks through the pass one function at
a time.

---

## 5. The pass, step by step

### 5.1 Detecting duplicates: a normalized identity key

<https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py#L28-L37>

```python
def _constant_key(op: SpyreConstantFallback) -> tuple:
    """Normalised (value, dtype, device) identity key for a SpyreConstantFallback."""
    layout = op.layout
    dev = layout.device
    norm_dev = (
        torch.device(dev.type, dev.index)
        if dev.index is not None
        else torch.device(dev.type)
    )
    return (op.constant_args[0], layout.dtype, norm_dev)
```

Two constants are *the same* if they hold the same value, produce
the same dtype, and land on the same device. The key normalizes the
device — `cuda` and `cuda:None` should compare equal — so that
lowering sites which differ only in whether they specify a device
index still group together. The pass will later use this key to
bucket operations; duplicates are exactly those buckets with more
than one operation.

The value is deliberately not a hash of the buffer's *contents* after
materialization. It is a hash of the *description* the compiler was
told to materialize. Two constants with the same description will
produce bit-identical bytes; two constants with different
descriptions may accidentally produce identical bytes at runtime but
are not safe to merge because a later pass could specialize them
differently.

### 5.2 Rewriting consumers: name-swap wrappers over `inner_fn`

<https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py#L40-L49>

```python
def _patch_inner_fn(consumer: ComputedBuffer, name_map: dict[str, str]) -> None:
    """Wrap consumer's inner_fn to redirect duplicate constant reads to the canonical name."""
    orig_inner = consumer.data.inner_fn

    def _new_inner(*args, _map=name_map, _orig=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig(*args)

    object.__setattr__(consumer.data, "inner_fn", _new_inner)
    ComputedBuffer.get_default_sizes_body.clear_cache(consumer)
```

This is where the design fact from section 2 pays off. To make a
consumer read from the canonical buffer instead of the duplicate,
the pass does not rebuild the consumer's `inner_fn`. It wraps it:
the original callable is captured in a closure, and a new callable
is installed that, when called, temporarily replaces the ambient
ops handler with a `NameSwapHandler` and delegates to the original.

The `NameSwapHandler` — defined in the neighboring
`insert_restickify.py` — is a thin wrapper that intercepts `load`
calls and rewrites the buffer name argument through a lookup
table. Every other operation the handler could receive (`store`,
`add`, index computation, reductions) is forwarded unchanged. The
consumer's dependency structure, symbolic indexing, and generated
code all remain correct; only the name being loaded has changed.

The cache-clear on the last line is the same on-instance-cache issue
walked through in the companion document: an analysis that had
already computed default sizes against the old `inner_fn` must not
return that cached answer against the new one.

Consumer discovery happens one level up, in `_redirect_consumers`:

<https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py#L52-L79>

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
            raise AssertionError(...)
```

Two details are worth naming:

- **Graph outputs are exempt.** If the duplicate constant is itself
  a graph output, removing it would change the shape of the compiled
  function's return value. The pass declines to touch those and
  moves on. This is an example of the pass explicitly scoping what
  it is willing to modify.
- **Consumer discovery is by read-write analysis, not by a reverse
  lookup table.** The pass asks each operation for its read/write
  summary and checks whether the duplicate's name appears in the
  reads. This is more expensive than consulting `name_to_users`, but
  it is authoritative — it reflects what the operation actually
  loads under the current `inner_fn` state, including any earlier
  wrapping done in the same pass invocation.

### 5.3 Retiring the duplicate: the multi-invariant cleanup

<https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py#L82-L106>

```python
def _drop_constant(
    operations: list[Operation],
    dup: SpyreConstantFallback,
    canonical: SpyreConstantFallback,
) -> None:
    """Remove a duplicate constant from the graph and update bookkeeping."""
    D = dup.get_name()
    C = canonical.get_name()
    op_name = dup.get_operation_name()
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
    logger.debug("dedup_and_promote_constants: merged %s into canonical %s", D, C)
```

Where `_patch_inner_fn` is the payload of the pass — the thing that
makes the compiled program smaller — `_drop_constant` is the
bookkeeping that keeps the compiler's own state coherent after that
payload has been applied. It is instructive because it enumerates,
in one place, every piece of graph-level state that a
constant-removal touches:

- **Provenance.** `merge_provenance` fuses the duplicate's origin
  history into the canonical, so debug output and source
  attribution reflect that both lowering sites contributed to the
  canonical operation. Without this, later inspection of the
  canonical would report only one of the sites that produced it.
- **Operation list.** `operations.remove(dup)` deletes the
  duplicate from the linear operation sequence. This is what makes
  the scheduler stop seeing the duplicate as a materialization to
  schedule.
- **Logical buffer retirement.** Adding `D` to
  `V.graph.removed_buffers` is the graph's convention for "this
  buffer name is no longer live." Downstream code that iterates
  buffers or emits allocations treats these as dead. This is
  distinct from *unregistering* the buffer, which would allow the
  compiler's identity allocator to reuse the name for a future
  buffer and produce a subtle aliasing bug.
- **Name-to-buffer lookup.** `V.graph.name_to_buffer.pop(D, None)`
  removes the duplicate from the graph-wide lookup table so that
  passes which resolve a buffer name to its current object cannot
  accidentally find a stale reference.
- **Name-to-op lookup.** The same treatment for the operation-name
  namespace.
- **Reverse-user index.** Any user records that were keyed on the
  duplicate's buffer name are moved to the canonical. This matters
  for later passes that reason about which operations depend on a
  given buffer.

The pattern is the same nine-invariant shape as in the companion
document, adapted for *removal* rather than *replacement*. The
companion document argued that centralizing this sort of protocol in
one helper prevents drift across call sites; this pass is currently
an open-coded example, and the same argument applies.

### 5.4 Front-loading survivors: guaranteeing materialization order

<https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/dedup_constants.py#L140-L147>

```python
constants = [op for op in operations if isinstance(op, SpyreConstantFallback)]
if not constants:
    return
non_constants = [
    op for op in operations if not isinstance(op, SpyreConstantFallback)
]
operations[:] = constants + non_constants
```

After deduplication, the surviving constants are moved to the front
of the operation list. This is not required for correctness of the
deduplication — the redirected consumers still name the canonical
buffer, and the operation list order among constants and their
consumers was already topologically valid. It is a scheduling
prepare step: putting all constants first ensures that later scheduling
and codegen work sees a shape where every constant is materialized
before anything else, which simplifies memory planning for the
backend.

The `operations[:] = ...` idiom is deliberate. Assigning to the slice
mutates the list in place, so any other reference to the same list
elsewhere in the graph sees the reordered version. Rebinding the
name `operations` would leave those other references pointing at
the original ordering.

---

## 6. What this pass is and isn't

This pass is a **runtime-work-elimination pass**. Its reason for
existing is that without it, the compiled Spyre executable would
materialize each duplicated constant separately at every invocation:
allocating more device memory than needed, issuing more fill kernels
than needed, and reading identical bytes from more distinct addresses
than needed. The correctness of the program does not depend on this
pass, but the efficiency of the program does.

This is a specific instance of a family of compiler optimizations
that go by several names — common-subexpression elimination, value
numbering, constant deduplication, constant pooling. Every mature
compiler has some version of it. The reason this out-of-tree backend
has to implement its own is section 3: the specific pattern that
produces the duplicates (independent lowering steps materializing
structurally identical constants) is not caught by the mechanisms
upstream provides at the level this backend hooks in.

This pass is **not** an example of the class of problem walked
through in the companion document. That document was about a helper
whose entire purpose is to keep the compiler's own state consistent
during an unrelated mutation; it does not remove any runtime work.
The two documents together describe the two distinct reasons
loop-level IR passes exist in this backend:

1. **Framework-plumbing passes** — like the helper in the companion
   document — exist so that other passes can mutate the IR without
   quietly breaking downstream readers. Their value shows up in
   engineering time and correctness over time, not in what the
   compiled program does.
2. **Optimization passes** — like this one — exist so that the
   compiled program does less work at runtime. Their value shows up
   as fewer kernels, smaller memory footprints, and faster
   execution.

Both kinds of pass live in the same file layout, use the same IR
primitives, and touch the same graph-level state. Reading either
kind in isolation, it is easy to conflate them. Naming the
distinction is the point of these two documents.

---

## 7. What to take away

The pass is short — under a hundred and fifty lines of code — but
each piece corresponds to one specific concern:

1. `_constant_key` defines when two constants are the same.
2. `_patch_inner_fn` and `_redirect_consumers` change which buffer
   consumers read from, without rebuilding their loop bodies.
3. `_drop_constant` retires the duplicate from every graph-level
   index that would otherwise still see it.
4. The entry point orchestrates the three above and front-loads
   surviving constants.

The runtime effect on the compiled program is:

- **Fewer materialization operations** in the operation list, and
  therefore fewer kernels or fused-kernel contributions after
  scheduling.
- **Less device memory allocated** for constants that are
  bit-identical.
- **Better locality** for consumers that used to read from many
  addresses holding the same bytes and now read from one.

The framework-plumbing effect on the compiler's own state — every
lookup table updated, every provenance record merged, every logical
retirement recorded — is what keeps that optimization safe. The
companion document argues that the plumbing part deserves to live in
shared helpers; this pass is one of the sites where that argument
applies most concretely.
