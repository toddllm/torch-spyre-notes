# Case study: `replace_computed_buffer_body`

A first-principles walkthrough of one helper in an out-of-tree PyTorch
Inductor backend, and the invariants it exists to preserve.

- **File:** `torch_spyre/_inductor/pass_utils.py`
- **Function:** `replace_computed_buffer_body`
- **Pinned SHA:** `2b3ca93f12cc2571031c63514b723bc54aa55703`
- **Permalink to the function:**
  <https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/pass_utils.py#L1160-L1199>
- **Permalink to just the function body (without docstring):**
  <https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/pass_utils.py#L1181-L1199>

The goal of this document is to explain why forty lines of code are the
way they are — starting from what a compiler IR is, working down through
what Inductor's IR represents, why one particular IR node is unusually
awkward to modify, and finally what each line of the helper is doing and
which invariant it is protecting.

---

## 1. What layer of the compiler this touches

A modern deep-learning framework compiles a Python program in stages.
Each stage lowers the program to a representation with fewer abstractions
and more explicit machine detail. A common progression:

1. **Traced graph.** The Python model is traced into a graph of
   framework-level operations (matmul, add, softmax, view, etc.).
2. **Functional graph.** Autograd and functionalization rewrite the
   traced graph so it is side-effect-free and differentiable through
   pure functions.
3. **Loop-level IR.** The functional graph is lowered into loops — each
   tensor-producing operation becomes a description of *how* to compute
   each element as a function of loop indices, plus a description of
   *what* is read from other buffers.
4. **Scheduler.** The loop-level IR is grouped into kernels, fused where
   safe, and ordered for execution.
5. **Backend codegen.** Kernels are emitted as target code — CUDA
   source, C++, a device-specific spec, etc.

An out-of-tree backend that wants to reuse the upstream compiler's
tracing, autograd, decomposition, symbolic-shape handling, and lowering
work generally hooks in at stage 3 or 4. That is where the compiler has
already done the hard work of turning Python into indexed loop bodies
but has not yet committed to any target-specific decisions about
tiling, memory placement, or code emission.

The helper in this document runs at stage 3, immediately before the
scheduler is constructed. It is part of a family of *loop-level IR
transformation passes* that rewrite the graph after upstream lowering
but before upstream scheduling.

---

## 2. What a loop-level IR node actually contains

At stage 3, the natural unit of the IR is roughly:

- A **buffer** — a named tensor-shaped value, with a layout (shape,
  strides, dtype, device).
- An **operation** — a computation that produces one or more buffers.
- A **computed buffer** — the common case where a single operation
  produces a single buffer whose contents are defined by a loop body.

The loop body inside a computed buffer is not stored as text or as a
static syntax tree. It is stored as a **callable** — an ordinary Python
function whose parameters are the loop indices and whose body issues
abstract operations (`load`, `store`, `add`, `where`, …) against an
ambient handler object. Roughly:

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

This design has two consequences that shape the rest of this document.

First, **the loop body is executable IR**, not a passive data
structure. Its behavior — including exactly which SymPy symbols appear
in its index expressions — is only observed when it is called. Passes
that want to inspect or rewrite it must either call it or wrap it.

Second, **rebuilding an `inner_fn` from its printed form is unsafe.**
The printed form shows *values* of loop-index symbols at the time it
was printed, not a binding contract. If a pass reconstructs a new
`inner_fn` by constructing fresh SymPy symbols named `i0`, `i1`, they
are not the same objects the caller will pass in the future. The new
function can silently ignore its arguments and emit indices in terms of
symbols that never appear in the enclosing loop. Downstream dependency
extraction then produces indices with the wrong rank or the wrong free
variables. This class of bug is what motivates a family of "wrap, do
not reconstruct" rules in this codebase.

---

## 3. Why "just change the buffer" is not a one-liner

Suppose a pass has decided that a particular computed buffer should now
compute something different — a retiled version, a version that reads
from a different upstream buffer, a version whose body has been fused
with a small neighbor. What has to happen for that change to be
observed correctly by the rest of the compiler?

The computed-buffer object itself is a **frozen dataclass**. Its
`data` field, which holds the loop description, cannot be reassigned in
place. So the pass must construct a *new* computed-buffer object and
splice it into the graph in place of the old one.

But the loop-level IR is not a single graph object with clean edges.
Around each computed buffer there are several pieces of state that
downstream code assumes are coherent:

- **Identity.** The buffer has a name (`buf17`) and the operation that
  produces it has an operation name (`op42`). Elsewhere in the graph,
  other `inner_fn`s issue `ops.load("buf17", …)`. If the replacement
  buffer has a different name, those reads dangle. So the new buffer
  must inherit the old buffer's name *and* the old operation's name.
- **Position in the operation list.** The graph has a linear list of
  operations in a topologically valid order. Replacing an operation in
  place preserves this order; appending a new one to the end and
  leaving the old one in place does not.
- **Layout.** The layout object (shape, strides, dtype, device)
  encodes how downstream code addresses this buffer. Consumers'
  `inner_fn`s were built against that layout. If the replacement
  changes layout without also rewriting consumer indices, the loads
  read the wrong elements — this is the "same name, different strides"
  hazard mentioned in the codebase's own agent guidance.
- **Loop metadata.** Alongside the current loop description, computed
  buffers carry a set of `_original_*` fields — the original loop body
  and its original ranges — used by an internal method that decides
  how to enumerate this buffer's iteration space. If the replacement
  drops these, that method will fall back to inferring ranges from
  the current body, which may have already been retiled or otherwise
  transformed.
- **Provenance.** Each IR node carries origin metadata linking it back
  to the traced-graph node it came from, plus a history of passes that
  have touched it. Debug tooling, error messages, and profile
  attribution rely on this chain. A pass that constructs a new object
  and drops these fields silently breaks source attribution.
- **Backend-specific metadata.** In an out-of-tree backend, additional
  metadata is attached to IR nodes (target-specific hints, tiling
  choices, layout tags). If it is not copied over, later passes see a
  node that looks fresh and re-derive decisions that were already made.
- **Cached analyses.** Several methods on computed-buffer-shaped
  objects are decorated to cache their results on the instance. If the
  same object is mutated in place, the cache is stale. If a new object
  is constructed and one of these methods was already called on the
  *old* object during earlier lowering, then calling it on the *new*
  object may (depending on how the cache is keyed) return the old
  result. The safe rule is to explicitly invalidate the affected
  cache on the replacement.

Any pass that swaps a computed buffer without addressing every one of
these points is subtly wrong. The failure typically does not appear at
the mutation site. It appears one or two passes later, when something
downstream re-reads the state that was left inconsistent.

---

## 4. What the helper does, line by line

The helper is 20 lines of executable code (plus a docstring and one
comment). Each line corresponds to one of the invariants above.

Direct link to the function body:
<https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/pass_utils.py#L1181-L1199>

Signature (lines 1160–1167):

```python
def replace_computed_buffer_body(
    op: ComputedBuffer,
    new_data: Loops,
    operations: list[Operation],
    *,
    pass_name: str,
    reason: str | None = None,
) -> ComputedBuffer:
```

- `op` is the existing computed buffer to be replaced.
- `new_data` is the new loop description — a fresh `Loops` object,
  typically constructed by wrapping the old `inner_fn` rather than
  rebuilding it.
- `operations` is the graph's linear operation list. It is passed
  explicitly so the helper does not have to reach into ambient graph
  state to find it.
- `pass_name` and `reason` are provenance annotations that will be
  attached to the replacement so later inspection can identify which
  pass produced it and why.

The comment on lines 1181–1182 is load-bearing:

```python
# Always wrap the original inner_fn via WrapperHandler; never rebuild
# index expressions from scratch (they go stale — see issue #2797).
```

This is the same "wrap, do not reconstruct" rule from section 2,
recorded at the point where it is easiest to violate. The referenced
issue is the historical incident where a loop-level pass rebuilt an
`inner_fn` and produced index expressions whose rank did not match the
strides of the buffer they were addressing — a bug that survived the
transforming pass and only surfaced when a later pass replayed the
callable during dependency extraction.

Construction of the replacement (lines 1183–1191):

```python
new_buf = ComputedBuffer(
    name=op.get_name(),
    layout=op.layout,
    data=new_data,
    _split_size=op._split_size,
    _original_inner_fn=op._original_inner_fn,
    _original_ranges=op._original_ranges,
    _original_reduction_ranges=op._original_reduction_ranges,
)
```

Each keyword argument corresponds to one invariant:

- `name=op.get_name()` — preserve buffer identity. All consumer
  `inner_fn`s continue to see loads against the same buffer name.
- `layout=op.layout` — preserve the layout object. Consumers'
  addressing arithmetic remains valid.
- `data=new_data` — install the new loop description. This is the
  only field that changes.
- `_split_size` and the three `_original_*` fields — preserve the
  loop-shape metadata that the default-sizes analysis reads. Without
  these, that analysis re-infers ranges from the current body, which
  may already be a transformed version.

Operation-identity restoration (line 1192):

```python
new_buf.operation_name = op.operation_name
```

Buffers and operations are separate namespaces. The `ComputedBuffer`
constructor above sets the buffer name; this line separately sets the
operation name. Both must match the object being replaced or downstream
lookups that use operation names will not find the replacement.

Provenance (lines 1193–1194):

```python
preserve_provenance(op, new_buf, pass_name=pass_name, reason=reason)
copy_op_metadata(op, new_buf)
```

Two calls with two different jobs. `preserve_provenance` extends the
transform-history chain — it records that this pass replaced `op` with
`new_buf`, using the human-readable `pass_name` and `reason`, so that
later debug output can reconstruct the sequence of transforms that
produced any given node. `copy_op_metadata` carries over the
backend-specific metadata that upstream tooling does not know about.

Cache invalidation (line 1195):

```python
ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)
```

The `get_default_sizes_body` method is decorated with an on-instance
cache. If the old computed buffer had already been asked for its
default sizes body during earlier lowering, and if the cache-key
mechanism identifies instances in a way that could confuse the new
object with the old, the new object could return the cached result
computed against the old `data`. Explicitly clearing the cache on the
new instance closes that hole. This is the only place in the helper
that reaches into an upstream implementation detail, and it is
deliberate — the alternative is a silently stale analysis.

Splice (lines 1197–1198):

```python
op_idx = operations.index(op)
operations[op_idx] = new_buf
```

Find the old operation in the graph's linear list and overwrite that
slot with the new one. This preserves topological order — anything
that was before `op` is still before `new_buf`, and anything that was
after it is still after — without the pass having to reason about
where the new object should go.

Return (line 1199):

```python
return new_buf
```

The caller now has a reference to the replacement in case it needs to
attach further metadata, register it in a graph-wide lookup table
(this helper does not do that itself — see the next section), or hand
it to a subsequent pass.

---

## 5. What the helper does not do — and why that matters

This helper handles the *object-level* invariants: identity fields,
metadata, loop shape, provenance, position in the operation list, and
the one known stale-cache hole.

It does not touch:

- **Graph-wide lookup tables.** The compiler maintains a
  name-to-buffer index that resolves a buffer name to the current
  buffer object. Because the helper reuses the old name, this index
  still points at the *old* object after the splice. Callers must
  update it — many of them do, immediately after calling this helper.
- **Reverse-user information.** A separate name-to-users index answers
  "which operations depend on this buffer?" Since the buffer name is
  preserved, reverse-user information remains structurally correct.
  But if a pass changes which buffer a consumer reads from (not the
  case here), that index has to be repaired at the call site.
- **Read/write dependency caches on other operations.** If the
  transformation changes what the loop body loads or stores, the
  cached read/write summary on the *replacement* is fresh (it is a
  new object), but summaries cached on unrelated operations that had
  already been analyzed remain valid — the helper's own concerns are
  local.

The reason to name these explicitly is that a caller reading this
helper in isolation might assume "call this and everything is
consistent." That is not the contract. The helper covers exactly the
work that is common to every replacement of a computed buffer's body.
Anything that varies by call site — updating graph-wide indexes,
redirecting consumers, invalidating third-party caches — remains the
caller's responsibility.

Concentrating the common part in one function has a specific benefit:
when a new invariant is discovered (as has happened repeatedly in this
codebase's history), only this helper has to learn about it. Passes
that already use it inherit the fix. Passes that open-code the same
sequence of steps must each be found and updated separately.

---

## 6. A concrete example of the same protocol, open-coded

### 6.0 Why this is worth calling out

The previous sections established two things:

1. Replacing the body of a computed buffer requires holding **nine
   distinct invariants** coherent at once (section 4).
2. Those invariants are not orthogonal — several of them (identity,
   layout, provenance, cache validity, and list position) each
   correspond to a specific piece of state that a *different*
   downstream reader depends on. A replacement that gets eight of
   them right and one of them wrong produces a graph that still
   loads, still lowers, still enters the scheduler, and fails
   somewhere else (section 3).

Given that shape of problem, concentrating the nine steps in one
helper is not a stylistic preference. It is the mechanism that keeps
the invariant list *knowable*: a compiler engineer who needs to
understand what a computed-buffer replacement entails can read one
function, and a compiler engineer who discovers a tenth invariant can
add it to one place and know that every call site benefits.

That mechanism only holds to the extent that all
computed-buffer replacements actually go through the helper. If any
call site open-codes the same nine steps in parallel, the
single-source-of-truth property is lost even though the code still
works today. The purpose of this section is to show one such site,
explain precisely what is wrong with it, and describe the minimal
change that restores the property.

The site chosen for this walkthrough is inside a layout-conversion
pass that inserts intermediate buffers between a producer and its
consumers to change the on-device layout of a value. Whenever the
pass retargets a consumer to read from a newly-inserted buffer, it
also has to replace the consumer's computed buffer with a fresh
object — for exactly the reasons described in section 3. This is the
same operation the helper exists to perform.

- **File:** `torch_spyre/_inductor/insert_restickify.py`
- **Permalink to the replacement block:**
  <https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/insert_restickify.py#L240-L264>

For comparison, the helper this document walks through:

- **File:** `torch_spyre/_inductor/pass_utils.py`
- **Permalink:**
  <https://github.com/torch-spyre/torch-spyre/blob/2b3ca93f12cc2571031c63514b723bc54aa55703/torch_spyre/_inductor/pass_utils.py#L1181-L1199>

At the pinned SHA, `insert_restickify.py` does not import
`replace_computed_buffer_body`. The nine-step protocol is spelled out
inline instead.

### 6.1 The two blocks, side by side

The helper (`pass_utils.py`, lines 1181–1199):

```python
# Always wrap the original inner_fn via WrapperHandler; never rebuild
# index expressions from scratch (they go stale — see issue #2797).
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

The open-coded version (`insert_restickify.py`, lines 240–264):

```python
# Reconstruct ComputedBuffer as a fresh object so the instance-keyed cache
# on get_default_sizes_body can be cleanly invalidated below.
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
preserve_provenance(
    op,
    new_consumer_buffer,
    pass_name="insert_restickify",
    reason="redirect consumer to restickified input",
)
copy_op_metadata(op, new_consumer_buffer)
# Replace op in the operations list with the reconstructed buffer.
operations[op_index] = new_consumer_buffer
V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer

# Invalidate the sizes/body cache so it is recomputed on next access with the patched inner_fn.
ComputedBuffer.get_default_sizes_body.clear_cache(new_consumer_buffer)
```

Setting aside the annotation comment, the local variable name, and the
inline literals for `pass_name` and `reason`, these two blocks are
performing the same nine-item procedure from section 4:

| Invariant                                | Helper                                            | Open-coded site                                   |
|------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| Buffer name                              | `name=op.get_name()`                              | `name=op.get_name()`                              |
| Layout                                   | `layout=op.layout`                                | `layout=op.layout`                                |
| Loop description                         | `data=new_data`                                   | `data=op.data`  (mutated in place just above)     |
| Loop-shape metadata                      | four `_split_size` / `_original_*` kwargs         | four `_split_size` / `_original_*` kwargs         |
| Operation name                           | `new_buf.operation_name = op.operation_name`      | `new_consumer_buffer.operation_name = op.operation_name` |
| Provenance chain                         | `preserve_provenance(...)`                        | `preserve_provenance(...)`                        |
| Backend-specific metadata                | `copy_op_metadata(op, new_buf)`                   | `copy_op_metadata(op, new_consumer_buffer)`       |
| Instance-cache invalidation              | `ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)` | `ComputedBuffer.get_default_sizes_body.clear_cache(new_consumer_buffer)` |
| Splice into operation list               | `operations[op_idx] = new_buf`                    | `operations[op_index] = new_consumer_buffer`      |

The open-coded site does one thing the helper does not: it also updates
the graph-wide name-to-buffer index on line 261:

```python
V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer
```

This is the caller-responsibility item flagged in section 5. It is
correct here. It is also the kind of step that is easy to leave out
elsewhere, which is exactly why the helper's docstring names it
explicitly.

### 6.2 What is wrong with the open-coded block

The block is not incorrect today. It performs all nine invariants
plus the graph-index update, in the correct order, using the same
constructor arguments and the same helper calls as the centralized
version. Someone reading it in isolation would find nothing to fix.

The problem is not in what the block does. The problem is in what
the block *is*: a second, independent copy of the computed-buffer
replacement protocol.

Three specific properties of the current state make that copy a
liability rather than a stylistic issue.

**It splits ownership of the invariant list.** The helper's contract
is documented in one place — its docstring names the fields it
preserves, its comment on lines 1181–1182 names the rule it enforces
(wrap, never rebuild), and section 5 above names the two caller
responsibilities it deliberately does not own. The open-coded block
carries none of that documentation. A reader who arrives at the
open-coded block first has no way to know that the seven kwargs to
the constructor, the `operation_name` assignment, the two provenance
calls, and the cache-clear form a *contract* rather than a
convenience — that each of them corresponds to a specific downstream
reader that will misbehave if it is skipped.

**It bypasses the single-place-to-fix property.** The helper exists
so that a new invariant, once discovered, can be added in one
function and every call site benefits. The open-coded block breaks
that property for exactly this call site. If a future change adds a
new required step — a new metadata field to copy, a second cached
analysis to invalidate, an additional loop-shape field on the
`ComputedBuffer` constructor — the helper will be updated but the
open-coded block will silently continue doing the previous nine
steps.

**It is not detectable by static means.** A grep for
`ComputedBuffer(` will find the constructor call, but the fact that
the surrounding lines are meant to satisfy the same nine-invariant
contract is not encoded anywhere the compiler or linter can see. The
two blocks look like independent code. They are, in fact, coupled by
a contract that lives only in the helper's docstring and in a
compiler engineer's head.

### 6.3 Why the failure this creates is quiet

Given how many downstream readers depend on computed-buffer state,
it might seem that a divergence between the two blocks would surface
loudly — an assertion failure, a wrong-shape error, a
`KeyError` on a missing lookup. In practice it will not.

Every invariant in the nine-item list is protective in the same
direction: it prevents *some specific downstream reader* from seeing
inconsistency. Miss one invariant and one downstream reader
misbehaves; the other eight downstream readers still see a
well-formed graph. Concretely, at this call site:

- Buffer name is preserved, so the graph-wide name-to-buffer lookup
  keeps resolving.
- Operation name is preserved, so operation-level lookups keep
  resolving.
- Layout is preserved, so consumers' addressing arithmetic still
  matches.
- Provenance is preserved, so debug output still attributes work to
  the right pass.
- The operation-list splice is in place, so topological order still
  holds.

If the open-coded block drifts by one item — say it stops copying a
newly-added backend metadata field — none of the checks above will
fail. The mutation site will not raise. The graph will look correct
in every top-level view. The only observable difference will be
whatever piece of state the missed invariant governs, and that
difference will only surface when some later pass reads that piece
of state and acts on it. The failure will therefore appear one or
two passes downstream, in code that has no obvious connection to
the pass that produced the drift, and its stack trace will point at
a reader that is behaving correctly given the state it was handed.

This is the failure mode that motivates centralizing the protocol in
the first place. It is also the failure mode that any duplicate site
reintroduces.

### 6.4 What the minimal repair looks like

The open-coded block can be replaced by two lines that delegate to the
helper and then add the one caller-responsibility step:

```python
new_consumer_buffer = replace_computed_buffer_body(
    op,
    op.data,
    operations,
    pass_name="insert_restickify",
    reason="redirect consumer to restickified input",
)
V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer
```

The change is purely structural. It removes about twenty-two lines of
open-coded state management and keeps the one line that the helper
declines to own on the caller's behalf. Once the two sites use the
same helper, any future invariant learned in one place propagates to
the other for free.

There is a variant refactor worth mentioning even if it is out of
scope for this document: the graph-wide `name_to_buffer` update is the
same one line at every caller. It is a candidate for a follow-on
change that either (a) extends the helper to accept the graph object
and perform the update itself, or (b) introduces a thin graph-editor
wrapper that owns both the object-level and graph-level halves of the
protocol. Either version would remove the last piece of duplicated
state-management knowledge from every caller of this helper.

---

## 7. What to take away

The forty lines of `replace_computed_buffer_body` are a compact record
of what it takes to modify one node of a loop-level IR without
breaking anything downstream. Read as a list of independent concerns,
they enumerate the pieces of state that a computed-buffer replacement
has to keep coherent:

1. Buffer name.
2. Operation name.
3. Layout.
4. Loop description (`data`).
5. Loop-shape metadata (`_split_size`, `_original_*`).
6. Provenance chain.
7. Backend-specific metadata.
8. On-instance cache for the default-sizes analysis.
9. Position in the operation list.

Each item corresponds to a specific downstream reader that would
misbehave if the item were skipped. The helper's value is not in any
one line — most of those lines could be typed correctly by any careful
author on any given day. The value is that all nine concerns live in
one place, are checked in code review together, and get updated
together when the compiler's assumptions shift.
