# `enable_spyre_lowerings` — RLock scope and process-global registry mutation

- **Id:** UF-03
- **Category:** upstream-fragility
- **Created:** 2026-08-20
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** likely (static reading confirms lock scope and mutation set; the wall-clock cost of holding the lock across a real compile has to be measured on a pod — see [../../needs-pod/04-parallel-compile-metamorphic.py](../../needs-pod/04-parallel-compile-metamorphic.py))
- **Status:** open

## Summary

`enable_spyre_lowerings` (torch-spyre `_inductor/lowering.py:173–279`)
is the context manager that swaps torch-spyre-authored lowerings into
**upstream Inductor's process-global** `torch._inductor.lowering.lowerings`
dict for the duration of a Spyre compile, and restores the previous
entries on exit. It coordinates concurrent entries via a module-level
`threading.RLock` and a nesting counter.

Two facts from a straight reading of the source are load-bearing:

1. **The lock is held across the entire `yield`** — meaning across
   the entire body of the caller's `with enable_spyre_lowerings():`
   block. In torch-spyre's case, that caller is
   `_inductor/patches.py:161–174`, which nests the CM inside four
   other CMs and then executes `torch._inductor.compile_fx`'s inner
   compile inside the block. **The lock is held for the duration of
   an Inductor compile, not just for the ms of setup/teardown of the
   `lowerings` dict.**

2. **`threading.RLock` protects the *same thread* from itself but
   serializes *other* threads.** Two threads running Inductor
   compiles that both enter `enable_spyre_lowerings` will serialize
   at the outer `with _lowerings_lock:` on line 183, one after the
   other, for the whole compile of the first thread.

This may be a correctness *requirement* — the mutation set below
would race disastrously without serialization — but the scope choice
means torch-spyre has effectively opted every process that imports it
out of concurrent Inductor compilation when *either* thread is
compiling for Spyre. Every claim below is anchored to `file:line` at
`torch-spyre@fea0c4be901e1383b1f700dbad8887128b0fcb27`.

## The RLock declaration

`_inductor/lowering.py:59–61`:

```python
# A module-level lock + nesting counter to make the CM reentrant/thread-safe
_lowerings_lock = threading.RLock()
_lowerings_nesting = 0
```

Both are **module-level globals** in `torch_spyre._inductor.lowering`.
Every torch-spyre import in a Python process shares one lock and one
counter. There is no per-`GraphLowering` or per-compile lock. The
comment "reentrant/thread-safe" is accurate about what `RLock`
provides in isolation — the same thread may re-acquire without
deadlock — but the *scope* of the region protected is not documented.

## The context manager, with the yield placement

`_inductor/lowering.py:173–279` — verbatim structure with call-out
line numbers:

```
173  @contextmanager
174  def enable_spyre_lowerings():
175      """...reentrant and safe under nested usage."""
182      global _lowerings_nesting
183      with _lowerings_lock:                    # <── ACQUIRE
184          first_enter = (_lowerings_nesting == 0)
185          _lowerings_nesting += 1
187          if first_enter:                       # setup phase (only on outermost entry)
188          …
239          try:
242              yield                             # <── caller's body runs HERE
243          finally:
244              _lowerings_nesting -= 1
245              last_exit = (_lowerings_nesting == 0)
246              if last_exit:                     # teardown phase (only on outermost exit)
247              …
279                                                # <── RELEASE (end of `with` block)
```

**The yield on line 242 is inside `with _lowerings_lock:` (opened
line 183).** The `with`-statement does not close until execution
leaves line 279, so the entire yielded-to block runs while the lock
is held. Because it is an `RLock`, the same thread can re-enter (the
setup/teardown phases skip on nested entry), but a *different* thread
that tries to enter will block on line 183 for the entire duration of
the outermost caller's `with` body.

## Every process-global mutation inside the CM

Mutations are grouped by "first-enter setup" vs "last-exit teardown".
Only the outermost enter and outermost exit run these. Every one of
them mutates state that is **shared across the entire torch process**,
because the state lives on `torch._inductor.lowering` (upstream module
globals).

### First-enter setup (lines 187–239)

| Site         | State mutated                                                           | Kind    |
|--------------|-------------------------------------------------------------------------|---------|
| `:188`       | `enable_spyre_lowerings._removed_fallbacks = {}`                        | attr on CM func (bookkeeping) |
| `:189–191`   | `unregister_lowerings(fallback_ops, lowering.lowerings, …)` — pops entries out of `torch._inductor.lowering.lowerings` | upstream registry mutation |
| `:195–197`   | `register_fallback_over_decomp(fallback_ops)` — calls `lowering.make_fallback(overload, override_decomp=True)` on each overload that carries an in-tree decomposition | upstream registry mutation |
| `:198–204`   | Writes `spyre_lowerings[op]` into `lowering.lowerings[op]` for every torch-spyre-authored lowering, saving displaced values into `saved_intree_lowerings` | upstream registry mutation |
| `:207–235`   | Installs `_impl_lower_aten_clamp`, `_impl_lower_aten_clamp_min`, `_impl_lower_aten_clamp_max` into `lowering.lowerings` at each `aten.clamp*` overload | upstream registry mutation |
| `:238`       | `enable_spyre_lowerings._saved_aten_lowerings = saved`                  | attr on CM func (bookkeeping) |
| `:239`       | `enable_spyre_lowerings._saved_lowerings = saved_intree_lowerings`      | attr on CM func (bookkeeping) |

The **five distinct writes to `lowering.lowerings`** are the load —
`:190` (bulk pop via `unregister_lowerings`), `:195` (bulk
`make_fallback`), `:204` (bulk replace with Spyre lowerings), and the
five `_save_set` calls at `:232–235` that overwrite five specific
`aten.clamp*` overloads.

### Yielded region (line 242)

`yield` on `:242` inside `with _lowerings_lock:`. **Any code the
caller runs — including all of Inductor's post-grad passes, the
scheduler, codegen, and any Spyre-specific pipeline steps — runs
while the lock is held.** In torch-spyre, the caller is `patches.py:161`:

```
161      with (
162          spyre_data_types(),
163          enable_spyre_lowerings(),         # <── this CM
164          V.set_real_inputs(example_inputs),
165          V.set_choices_handler(SpyreHeuristics()),
166          torch._inductor.config.patch(new_config),
167      ):
168          try:
169              yield                         # ← patches.py's yield to compile_fx
170          finally:
171              …
```

which is in turn nested inside `_inductor/__init__.py:156–157`:

```
156                  with enable_spyre_context(example_inputs):
157                      return _orig(gm, example_inputs, *args, **kwargs)
```

`_orig` is `torch._inductor.compile_fx.compile_fx`. So the actual
temporal scope of the `_lowerings_lock` acquisition is:

**setup of Spyre lowerings** → **entire duration of upstream
`compile_fx`** → **teardown of Spyre lowerings** → **release**.

### Last-exit teardown (lines 246–279)

| Site         | State restored                                                          | Kind    |
|--------------|-------------------------------------------------------------------------|---------|
| `:248–253`   | Restores or pops the five `aten.clamp*` overloads in `lowering.lowerings` | upstream registry restore |
| `:255`       | `enable_spyre_lowerings._saved_aten_lowerings = {}`                     | attr on CM func (cleanup) |
| `:257–266`   | For every `spyre_lowerings` entry, restores the saved in-tree value or pops the entry | upstream registry restore |
| `:269–271`   | For every overload added by `register_fallback_over_decomp`, pops from `lowering.lowerings` and `lowering.fallbacks.discard(overload)` | upstream registry restore (fallbacks set too) |
| `:272`       | `enable_spyre_lowerings._added_fallbacks = []`                          | attr on CM func (cleanup) |
| `:273–275`   | `restore_lowerings(enable_spyre_lowerings._removed_fallbacks, lowering.lowerings)` — puts the originally-removed fallback lowerings back | upstream registry restore |
| `:278`       | `enable_spyre_lowerings._saved_lowerings = {}`                          | attr on CM func (cleanup) |
| `:279`       | `enable_spyre_lowerings._removed_fallbacks = {}`                        | attr on CM func (cleanup) |

Teardown restores exactly the four bulk mutations setup performed,
in reverse: aten.clamp overloads → spyre_lowerings entries →
decomp-carrying fallbacks → originally-removed fallbacks.

### Fixture on the CM function itself

Six attributes stored on the `enable_spyre_lowerings` function object
survive across a `with` block:

- `enable_spyre_lowerings._removed_fallbacks` — dict of saved
  overloads popped out at first-enter (`:188`).
- `enable_spyre_lowerings._added_fallbacks` — list of overloads
  added by `register_fallback_over_decomp` (`:195`).
- `enable_spyre_lowerings._saved_aten_lowerings` — the five
  `aten.clamp*` overloads saved before Spyre versions were installed (`:238`).
- `enable_spyre_lowerings._saved_lowerings` — the pre-Spyre values
  of every `spyre_lowerings` key that was already in
  `lowering.lowerings` (`:239`).

These four are **also process-global** (a function object has one
`__dict__` per process) — they exist between compiles, cleared to
empty on teardown. A crash between `:187` (first-enter set up begins)
and `:239` (setup completes) leaves them in a partially-populated
state, but the `finally:` block on `:243` still runs the full
teardown, which reads them with `getattr(…, …, {})` defaults — so a
crash mid-setup will not KeyError, though it may fail to restore
state that setup never got to save. (Whether *any* Inductor state
is durable across a compile crash under Spyre is untested; see
`needs-pod/04-parallel-compile-metamorphic.py` scenario c.)

## Scope argument: setup/teardown-only vs entire-compile

The question the audit needs to answer plainly: does the RLock scope
protect *only* the setup and teardown of the upstream registry
mutation, or does it hold across the entire compile that the caller
runs inside the `with` block?

### The scope is the entire compile.

The `yield` on `:242` is **inside** the `with _lowerings_lock:` block
that begins on `:183`. Python's `with`-statement releases the context
manager at block exit — not at `yield`. So the lock is acquired at
`:183`, held through the setup phase (`:187–239`), held across the
yield (`:242`), held through the finally-driven teardown
(`:243–279`), and released at the end of the `with` block.

A hypothetical alternative that held the lock only across the
mutation windows would look like:

```python
@contextmanager
def enable_spyre_lowerings():
    global _lowerings_nesting
    with _lowerings_lock:                         # acquire for setup only
        first_enter = (_lowerings_nesting == 0)
        _lowerings_nesting += 1
        if first_enter:
            …setup…
    try:
        yield                                     # <── OUTSIDE the lock
    finally:
        with _lowerings_lock:                     # acquire for teardown only
            _lowerings_nesting -= 1
            if _lowerings_nesting == 0:
                …teardown…
```

That is not what `enable_spyre_lowerings` does. The narrower scope
above would let two threads run their compiles in parallel while
serializing only the ms-scale registry-swap windows — but it would
also let a second thread enter the `yield` region while the first
thread had already installed its Spyre lowerings, so **both threads
would see the same, shared, torch-spyre-installed `lowering.lowerings`
dict during their compiles**. Whether that is a correctness problem
depends on whether Inductor's post-grad and lowering passes read
`lowering.lowerings` off the module (yes — via
`torch._inductor.lowering.lowerings` imported into every pass) and
whether they mutate it (mostly no; `make_fallback` and the
`decompositions` bookkeeping are the exceptions).

The wide scope torch-spyre took is defensible: it prevents a
concurrent CPU-only compile from reading Spyre-installed lowerings
that would misdirect it. But it is **not free** — it serializes
Inductor compilation. See below.

### What this locks out

- Two Spyre compiles in two threads of the same process: full
  serialization. Thread B blocks on `:183` for the entire duration of
  thread A's compile.

- One Spyre compile and one non-Spyre compile in two threads: the
  non-Spyre compile does *not* enter `enable_spyre_lowerings` (it is
  reached only under `if uses_spyre:` in
  `_inductor/__init__.py:139`), so the non-Spyre compile does not
  try to acquire the lock. Thread B may run in parallel — **but it
  reads a `lowering.lowerings` that has been mutated by thread A's
  Spyre setup**, because the mutation is process-global and not
  copy-on-write. See `04-monkey-patches-outside-patches-py.md` and
  `needs-pod/05-process-contamination.py` for the CPU-contamination
  metamorphic.

- Recursive Spyre compile (Spyre compile that itself triggers another
  Spyre compile from inside a lowering): OK — same thread re-enters
  the RLock, the nesting counter increments (`:185`) and the setup
  path is skipped (`:187` gates on `first_enter`). Teardown is
  similarly gated on `last_exit == True` (`:246`). But a bug that
  double-decrements or forgets to enter would leave the process in a
  state where either setup runs again (double-install) or teardown
  never fires (permanent mutation of `lowering.lowerings`).

### What the wall-clock cost is

Static reading cannot answer this. `needs-pod/04-parallel-compile-metamorphic.py`
instruments the CM with hold-time measurement so the pod run can
report the difference between (a) two serial Spyre compiles, (b) two
Spyre compiles submitted to two threads, and (c) one Spyre + one CPU
compile submitted to two threads. If (a) and (b) are within noise, the
lock is doing no real damage. If (b) is close to 2× (a), Spyre has
made two-thread Inductor compilation serial.

## Related follow-ups

- The setup phase mutates two more registries that are not
  `lowering.lowerings`: `lowering.fallbacks` (implicit set discarded
  from at `:271`), and the return of `lowering.make_fallback` (which
  writes `override_decomp=True` at `:163`). The `discard` cleanup at
  `:271` is important — see the ledger row for `_added_fallbacks`.

- The two positional/pass-list surgeries in `patches.py`
  (`joint_graph.pass_patterns.pop()` and
  `post_grad.pass_patterns[2]`) are **not** inside
  `enable_spyre_lowerings`. They are protected by no lock at all —
  see [01-patches-ledger.md](01-patches-ledger.md) rows L12 and L15.

- `enable_spyre_context` (the outer CM at
  `_inductor/patches.py:41–174`) also mutates
  `torch._prims_common._computation_dtype_map`, `Loops.has_large_inner_fn`,
  `GraphLowering._update_scheduler`, `SchedulerNode.has_side_effects`,
  `GraphTransformObserver.apply_graph_pass`, `joint_graph.pass_patterns`,
  and `post_grad.pass_patterns[2].patterns` — none of them under
  `_lowerings_lock`. The lock **does not cover the fully monkeypatched
  set**; it covers only the lowerings-dict slice. A concurrent thread
  reading `SchedulerNode.has_side_effects` between two Spyre compiles
  sees whatever the last enter-or-exit left there, unsynchronized.
