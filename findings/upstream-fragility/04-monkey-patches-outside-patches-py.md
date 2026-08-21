# Process-global monkey-patches outside `patches.py`

- **Id:** UF-04
- **Category:** upstream-fragility
- **Created:** 2026-08-20
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** confirmed (static; every entry is a `file:line` on the pinned tree)
- **Status:** open

## Summary

Torch-spyre's process-global mutations of upstream state extend well
beyond `_inductor/patches.py`. This finding enumerates every
mutation-of-torch site the audit found outside `patches.py`, so a
reviewer counting "patch sites" against upstream sees the real load.

Grouped by mechanism, the sites are:

- **Direct attribute assignment on `torch` classes / modules** — 11 sites.
- **Method-slot swaps on upstream types (Python method-assign)** — 7 sites.
- **`torch._dynamo` public helpers used to mutate global registration state** — 2 sites.
- **`torch.library.custom_op` decorators (permanent global namespace registrations)** — 22 sites (`_inductor/customops.py`).
- **`torch.library.register_kernel`** — 4 sites (`_inductor/customops.py`).
- **`torch.library.Library("aten", "IMPL", …)` handle creation and `.impl(...)` calls** — 2 handle creations + N impl calls (see below).
- **Module-level writes into upstream registry dicts** — 3 sites (`inplaceable_ops`, `hbm_pool_sizes`, `GUARD_VALUE_DISPATCH`).
- **`object.__setattr__` writes on upstream IR-node instances** — 17 sites (already tabulated in [02-private-api-inventory.md](02-private-api-inventory.md); referenced not repeated here).
- **Upstream `register_*` API calls** — 4 sites (documented as extension-point calls, not monkey-patches).

None of these are inside `_inductor/patches.py`. Every one is
executed at import time or first-use time and *permanently* mutates
the state of the torch process. Anchors are `file:line` on
`torch-spyre@fea0c4be901e1383b1f700dbad8887128b0fcb27`.

## 1. Direct attribute assignment on `torch` classes / modules

All at import time (via `_autoload_impl` in `torch_spyre/__init__.py`
or `_patch_tensor_for_spyre` in `torch_spyre/_monkey_patch.py`).

| # | Site                                          | LHS (mutated)                                                                 | RHS (installed)                                                     |
|---|-----------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------|
| 1 | `_monkey_patch.py:208`                        | `torch.Tensor.__repr__`                                                       | `spyre_aware_repr` — CPU-copy-and-splice-device-string wrapper      |
| 2 | `_monkey_patch.py:209`                        | `torch.Tensor.device_tensor_layout`                                           | `device_tensor_layout` — new method (not upstream); reads the Spyre C++ layout |
| 3 | `_monkey_patch.py:210`                        | `torch.Tensor._spyre_tensor_patched`                                          | `True` — Spyre-private marker on `torch.Tensor`                     |
| 4 | `_monkey_patch.py:211`                        | `torch.Tensor.to`                                                             | `spyre_to` — wraps upstream `.to` with D2D-cast-through-CPU logic and layout-carrying variants |
| 5 | `_monkey_patch.py:228`                        | `torch.empty`                                                                 | `spyre_empty` — normalizes size arg, forwards to `empty_with_layout` when `device_layout` present |
| 6 | `_monkey_patch.py:348`                        | `torch._dynamo.guards.GuardBuilder.TENSOR_MATCH`                              | `_spyre_TENSOR_MATCH` — adds a lambda guard on `device_tensor_layout()` for Spyre tensors |
| 7 | `_monkey_patch.py:399`                        | `torch._inductor.codecache.FxGraphHashDetails.__init__`                       | `_spyre_init` — appends `spyre_layouts` to the hash-detail record   |
| 8 | `_monkey_patch.py:400`                        | `torch._inductor.codecache.FxGraphHashDetails._spyre_hash_patched`            | `True` — marker on `FxGraphHashDetails`                             |
| 9 | `torch_spyre/__init__.py:369`                 | `torch.profiler.profile.__init__`                                             | `_init_with_spyre_profiler` — self-restoring wrapper that lazy-imports `torch_spyre._C` on first profile |
| 10 | `torch_spyre/__init__.py:379`                | `torch._dynamo.config.cache_size_limit`                                       | `1024` (upstream default is 8)                                      |
| 11 | `torch_spyre/__init__.py:392`                | `torch._C._accelerator_isAllocatorInitialized`                                | `_patched_isAllocatorInitialized` — swallows `RuntimeError("not a DeviceAllocator")` and returns False |
| 12 | `model_utils.py:365`                          | `nn.Module.to`                                                                | `_spyre_module_to` — reroutes `.to("spyre")` through `load_model_to_spyre` |

Site #10 (`torch._dynamo.config.cache_size_limit = 1024`) is a
config-flag override, not a code substitution, so it is
qualitatively different from the others; the risk profile is the
same as the `patches.py` L1/L2/L8/L9/L10 config overrides.

Sites #1, #2, #4, #5, #6, #7, #9, #11, #12 replace upstream method /
attribute *implementations* with torch-spyre-authored ones. **Every
subsequent caller in the process — Spyre and non-Spyre alike — hits
the Spyre implementation.** Site #4 (`torch.Tensor.to`) is the
broadest: `.to("cpu")`, `.to("cuda")`, `.to(dtype)` on any tensor in
the process runs through `spyre_to` first; the comment at
`_monkey_patch.py:213–226` explicitly documents this is "a
process-global registration affecting every user of `torch.Tensor.to`,
not just Spyre" and justifies it.

## 2. Method-slot swaps carrying `# type: ignore[method-assign]`

Enumerated separately because mypy's `method-assign` code marks these
as **known signature mismatches** — the type-checker is being told to
ignore them. The five sites in `_inductor/patches.py` are covered by
[01-patches-ledger.md](01-patches-ledger.md); the sites outside
`patches.py`:

| Site                        | LHS                                                    | Purpose                                                       |
|-----------------------------|--------------------------------------------------------|---------------------------------------------------------------|
| `model_utils.py:365`        | `nn.Module.to`                                         | (Site #12 above.)                                             |

All other `method-assign` suppressions across the tree are inside
`patches.py`. The absence of the suppression on the
`_monkey_patch.py` sites #1, #2, #4, #5, #6, #7, #9, #11 above is
worth calling out: those swaps *should* also carry
`method-assign` — that they do not is a mypy false-negative
(compounded by the fact that `torch.Tensor` is loosely typed in stub
files, and `torch._C._accelerator_isAllocatorInitialized` is a
C-shim). Only `nn.Module.to` was typed strictly enough that mypy
flagged the swap.

## 3. `torch._dynamo` public helpers used to mutate global state

Two sites use torch-public APIs whose *effect* is a process-global
mutation:

- `_monkey_patch.py:227`: `torch._dynamo.allow_in_graph(torch.Tensor.to)` — marks the just-installed `spyre_to` as a Dynamo leaf. Global; every Dynamo trace in the process treats `.to` as a leaf after this call.
- `_inductor/propagate_hints.py:62`: `@torch.compiler.allow_in_graph` on a module-level function. Adds one function to the process-global allowed-in-graph set.

Neither is monkey-patching — they call documented torch APIs — but
each mutates a global registry.

## 4. `torch.library` — permanent operator-namespace additions

`_inductor/customops.py` uses `torch.library.custom_op` at module-level
to register 22 Spyre-namespace ops:

- `spyre::softplus`, `layer_norm`, `exx2`, `layernormscale`, `layernormnorm`, `rms_norm`, `topkvalue`, `topkindex`, `gelu`, `silu`, `clamp`, `empty`, `logical_not`, plus 9 more at `:234, :282, :307, :324, :401, :581, :599, :634, :754, :786` (batched_matmul, restickify, copy_forced/opaque_copy_/overwrite variants, dequantize/quantize fp8, etc.).

These decorators register the ops **into `torch.ops.spyre` at import
time**, permanently. They are not undoable in-process. Four
`@torch.library.register_kernel` decorators additionally register
CPU-side implementations for four of these ops
(`_inductor/customops.py:294,317,360, one more`).

`_inductor/decompositions.py:198–199` creates two
`torch.library.Library` handles at first `_register_spyre_dispatchkey_kernels_permanently()`
call:

```
198  _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
199  _spyre_lib = Library("aten", "IMPL", "PrivateUse1")
```

These are stored in module-level globals and then have `.impl(op, fn)`
called for every op in `spyre_decompositions` that lacks an existing
PrivateUse1 kernel (`_inductor/decompositions.py:202–208`). The
handles are **permanent** — once created, releasing them destroys
the registered kernels, but torch-spyre keeps them alive via the
module globals. This is process-global registration, gated by an
idempotency flag `_dispatchkey_kernels_registered`
(`_inductor/decompositions.py:99` and `:210`).

## 5. Module-level writes into upstream registry dicts

Three writes happen at module import time or first-use time and
mutate upstream registry dicts directly:

- `_inductor/customops.py:395` — `inplaceable_ops[torch.ops.spyre.overwrite_f.default] = InplaceableOp(...)`. This writes into `torch._inductor.fx_passes.reinplace.inplaceable_ops` at import time. Permanent.
- `_monkey_patch.py:346` — `GUARD_VALUE_DISPATCH["_spyre_TENSOR_MATCH"] = _spyre_reuse_spec` (guarded by an `ImportError` fallback at `:306` — see the `try/except ImportError: else:` block). Permanent.
- `_inductor/hbm_pool_planning.py:514` — `V.graph.hbm_pool_sizes = …` at first pass entry. Per-`GraphLowering` in principle, but the `V.graph` reference for a given compile persists past the compile's end unless torch's own teardown drops it.

## 6. Upstream extension-point `register_*` calls

These are not monkey-patches — torch exposes `register_*` for
device-registration explicitly — but they are still process-global
and worth listing so the surface is complete:

- `torch_spyre/__init__.py:288`: `torch.utils.rename_privateuse1_backend(DEVICE_NAME)` — permanently renames the PrivateUse1 backend to `"spyre"` in the running process.
- `torch_spyre/__init__.py:289`: `torch._register_device_module(DEVICE_NAME, make_spyre_module())` — permanent.
- `torch_spyre/_inductor/__init__.py:191`: `register_interface_for_device(DEVICE_NAME, SpyreInterface)` — permanent.
- `torch_spyre/_inductor/__init__.py:204`: `register_device_op_overrides(device=DEVICE_NAME, device_op_overrides=SpyreDeviceOpOverrides())` — permanent.
- `torch_spyre/_inductor/__init__.py:211`: `register_backend_for_device(DEVICE_NAME, SuperDSCScheduling, SpyrePythonWrapperCodegen, device_custom_config=config)` — permanent.
- `torch_spyre/__init__.py:343`: `dist.Backend.register_backend(DISTRIBUTED_BACKEND_NAME, _create_spyre_ccl_backend, devices=[DEVICE_NAME])` — permanent.

## 7. `enable_spyre_compile_fx_wrapper` — the `compile_fx` slot swap

`torch_spyre/_inductor/__init__.py:60–170`:

- `_inductor/__init__.py:169`: `cfx.compile_fx = _wrapper` — replaces `torch._inductor.compile_fx.compile_fx` with the Spyre-aware wrapper.
- `_inductor/__init__.py:170`: `cfx._spyre_wrapped = True` — idempotency marker.

Guarded by `_autoload_lock` (a `threading.Lock` at `_inductor/__init__.py:26`) and the `_spyre_wrapped` marker for idempotency, but the effect is permanent: after `_light_autoload()` runs, `torch._inductor.compile_fx.compile_fx` in the process is torch-spyre's wrapper, and every compile — Spyre or otherwise — routes through it. The wrapper's own `if uses_spyre:` gate (`:139`) keeps the Spyre setup off the non-Spyre path, but the *wrapper itself* is on every compile path in the process.

## 8. Module-level *torch state mutation* summary

Combining sections 1, 3, 4, 5, 6, 7, torch-spyre performs at least
**56 distinct process-global mutations of torch state** outside
`_inductor/patches.py`:

- 12 direct attribute assignments (§1).
- 2 `allow_in_graph`-family mutations of Dynamo globals (§3).
- 22 `torch.library.custom_op` registrations + 4 `register_kernel` (§4).
- 2 `Library.__init__` handle creations (§4) + N `.impl(...)` calls (bounded by len(`spyre_decompositions`); scanner-observed but not further enumerated).
- 3 direct registry-dict writes (§5).
- 6 `register_*` public-API calls (§6).
- 1 `compile_fx` slot swap (§7) + 1 idempotency marker.

The `patches.py` ledger — 16 site rows — is only about **22% of the
torch-state-mutation surface torch-spyre installs into the process.**
The other 78% happens outside the CM, at import time or
first-lazy-init time, and is not undone by the CM's teardown path.

## Idempotency posture

Torch-spyre defends against double-application at each of these
sites, using one of three patterns:

1. **Attribute marker on the target class**: `torch.Tensor._spyre_tensor_patched` (`_monkey_patch.py:210`), `FxGraphHashDetails._spyre_hash_patched` (`_monkey_patch.py:364,400`), `_spyre_module_to._spyre_patched` (`model_utils.py:364`), `cfx._spyre_wrapped` (`_inductor/__init__.py:65,68,170`).
2. **Module-global boolean**: `_dispatchkey_kernels_registered` (`_inductor/decompositions.py:99,193,210`), `_autoload._ran` (`_inductor/__init__.py:181,185,218`).
3. **`threading.Lock` around the check-then-set**: `_autoload_lock` (`_inductor/__init__.py:26,67,68,184–185`).

These prevent double-install if `_autoload` fires twice (the comment
at `torch_spyre/__init__.py:257` "guard if autoload may run more than
once" indicates this has happened in practice). They do **not**
prevent *concurrent* first-installs from stepping on each other in
the small windows between marker-read and marker-write for cases 1
and 2. `_autoload_lock` in case 3 does. Whether the case-1/2 sites
matter in practice depends on whether autoload is ever reached from
two threads simultaneously; testing that is out of scope for
static reading.
