# Ledger of upstream optimizations suppressed or replaced in `patches.py`

- **Id:** UF-01
- **Category:** upstream-fragility
- **Created:** 2026-08-20
- **Revision manifest:** [reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md](../../reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)
- **Confidence:** confirmed (static reading; every claim is line-anchored against fetched sources at the three pinned SHAs)
- **Status:** open

## Summary

`torch_spyre/_inductor/patches.py` mutates upstream Inductor state in
**sixteen** distinct places when the Spyre compilation context is
entered (one physical site per row of the human-readable ledger
below; the `SchedulerNode.has_side_effects` row (L14) is one physical
site with two independent branches — L14a and L14b — that carry
different verdicts, so the machine-readable ledger has **17 verdict
rows**). The mutations fall into three `kind`s (the `mutation |
config | extension-point` axis the validator checks against — see
the "Machine-readable ledger" table further down):

| Kind (validator axis) | Count | What it is                                                                                                     |
|-----------------------|-------|----------------------------------------------------------------------------------------------------------------|
| `config`              | 5     | Config-flag overrides that flip an upstream default (rows L1, L2, L8, L9, L10).                                |
| `extension-point`     | 5     | Config-flag "install" slots that upstream deliberately exposes for a Spyre-authored pass (rows L3–L7).         |
| `mutation`            | 7     | Class-method monkey-patches (L11, L13, L14a, L14b, L16) + positional pass-list surgery (L12, L15). L15 also mutates a `PatternMatcherPass` registry entry (`extra_check` swap) at the same site. |

The five pure config overrides simply flip an upstream default that
is unchanged between torch-spyre's supported baseline
(`v2.13.0` @ `cf30153c4c131c8164ee7798e5022d810682e2cb`) and current
pytorch main (`c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`) — they are
`still-required` opinion overrides, not obsolete workarounds. The
five `extension-point` slots (`pre_grad_custom_pass`,
`post_grad_custom_pre_pass`, `post_grad_custom_post_pass`,
`_pre_fusion_custom_pass`, `_post_fusion_custom_pass`) install
Spyre-authored passes into slots upstream deliberately exposes;
those are `still-required` and are not "suppressions" in the same
sense.

The three positional / method-signature couplings —
`joint_graph.pass_patterns.pop()`, `post_grad.pass_patterns[2]`, and
the `is_valid_addmm_fusion` extra-check swap — are `needs-testing`:
the positional shape (`len(joint_graph.pass_patterns) == 2`,
`len(post_grad.pass_patterns) == 3`, and
`is_valid_addmm_fusion` being an identity-comparable module-level
symbol) is unchanged between v2.13 and main, but the position is
undocumented and can shift silently. `is_valid_addmm_fusion` picked up
a new pre-condition (`_PRESERVE_FLEX_GEMM_GEMM_OP`) on main that
torch-spyre would silently discard when it replaces `extra_check`.

The `SchedulerNode.has_side_effects` override references
`MutationLayoutSHOULDREMOVE` — the class name declares itself
scheduled for removal. Removal upstream would silently break the
Spyre override.

Verdict headline (one row per physical override site; row L14 has
two branches with distinct verdicts, so it appears twice below and
the totals sum to 17):

| Verdict            | Count | Overrides                                                                                                                                                                             |
|--------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `still-required`   | 13    | 5 config-flag overrides (L1, L2, L8, L9, L10) + 5 extension-point installs (L3–L7) + `Loops.has_large_inner_fn` (L11) + `GraphLowering._update_scheduler` (L13) + `GraphTransformObserver.apply_graph_pass` (L16). |
| `needs-testing`    | 3     | `joint_graph.pass_patterns.pop()` (L12), `SchedulerNode.has_side_effects` branch (a) (L14a), `post_grad.pass_patterns[2]` walk (L15).                                                  |
| `possibly-obsolete`| 0     | —                                                                                                                                                                                     |
| `unknown`          | 1     | `SchedulerNode.has_side_effects` branch (b) — the `MutationLayoutSHOULDREMOVE` branch (L14b) — needs a runtime probe to confirm the branch still fires on main.                        |

Counts above are derived from the "Machine-readable ledger" table
lower in this file. `scripts/validate_metadata.py` reads that table
and cross-checks these headline totals against the row-by-row data.

## Files and symbols

- torch-spyre: [`torch_spyre/_inductor/patches.py`](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py) — `enable_spyre_context` (L40–L174), `patch_inductor_fusions` (L180–L218)
- upstream v2.13.0: `torch/_inductor/{config,ir,graph,scheduler}.py`, `torch/_inductor/fx_passes/{joint_graph,post_grad}.py`, `torch/fx/passes/graph_transform_observer.py`
- upstream main: same paths

## Observed behavior

### Ledger

Rows are keyed by (source location in `patches.py`) → (upstream
target). "Introducing commit" is the torch-spyre SHA that made the
line what it is at HEAD (from `gh api graphql` blame on
`torch_spyre/_inductor/patches.py` @ `fea0c4b`).

| # | Override (patches.py L)                                              | Category              | Upstream symbol                                            | Introducing commit                              | Why (comment / commit)                                                                                       | Verdict         | Notes |
|---|---------------------------------------------------------------------|-----------------------|------------------------------------------------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|-------|
| 1 | `"split_reductions": False` (L82)                                    | config-flag           | `torch._inductor.config.split_reductions` (v2.13 L1005; main L1061) | `0aa4fff9fc32` (Use `config.patch` CM, 2026-04-04) | Not documented in file. Spyre backend has no equivalent to split-reduction codegen.                          | still-required  | Upstream default `True` at both baselines. |
| 2 | `"benchmark_harness": False` (L83)                                   | config-flag           | `torch._inductor.config.benchmark_harness` (v2.13 L271; main L275) | `0aa4fff9fc32`                                  | Not documented. Benchmark harness inserts extra code paths for CPU/GPU benchmarking; irrelevant for Spyre.   | still-required  | Upstream default `True` at both baselines. |
| 3 | `"pre_grad_custom_pass": CustomPreGradPasses()` (L84)                | config-flag (install) | `torch._inductor.config.pre_grad_custom_pass` (v2.13 L313; main L317) | `dc68f7316352` (Matmul padding, 2026-04-10)     | Spyre installs its own pre-grad pipeline. Slot is a supported extension point.                               | still-required  | Not a suppression; extension-point use. |
| 4 | `"post_grad_custom_pre_pass": CustomPrePasses()` (L85)               | config-flag (install) | `torch._inductor.config.post_grad_custom_pre_pass` (v2.13 L300; main L304) | `0aa4fff9fc32`                                  | Spyre installs its own post-grad-pre pass. Supported extension point.                                        | still-required  | Not a suppression. |
| 5 | `"post_grad_custom_post_pass": CustomPostPasses()` (L86)             | config-flag (install) | `torch._inductor.config.post_grad_custom_post_pass` (v2.13 L301; main L305) | `0aa4fff9fc32`                                  | Same.                                                                                                        | still-required  | Not a suppression. |
| 6 | `"_pre_fusion_custom_pass": CustomPreFusionPasses()` (L87)           | config-flag (install) | `torch._inductor.config._pre_fusion_custom_pass` (v2.13 L318; main L322) | `0aa4fff9fc32`                                  | Underscore-prefixed extension slot (private-ish). Spyre installs pre-fusion pass.                            | still-required  | Slot name starts with `_` — semi-private extension. |
| 7 | `"_post_fusion_custom_pass": CustomPostFusionPasses()` (L88)         | config-flag (install) | `torch._inductor.config._post_fusion_custom_pass` (v2.13 L325; main L329) | `0aa4fff9fc32`                                  | Same.                                                                                                        | still-required  | Same slot-privacy note as #6. |
| 8 | `"unroll_reductions_threshold": 1` (L91)                             | config-flag           | `torch._inductor.config.unroll_reductions_threshold` (v2.13 L991; main L1047) | `0aa4fff9fc32`                                  | L89–90 comment: "avoid the optimization of turning small matmuls into non-matmuls" (points at `ir.py#L1580`). | still-required  | Upstream default `8` at both baselines. The `ir.py#L1580` URL points to `main`, not a pinned SHA — the comment link rots the moment upstream renumbers. |
| 9 | `"permute_fusion": False` (L93)                                      | config-flag           | `torch._inductor.config.permute_fusion` (v2.13 L1518; main L1642) | `0aa4fff9fc32`                                  | L92 comment: "Disable fusing of mm + permute/transpose for now."                                             | still-required  | Upstream default `False` at both baselines. **This override is a no-op on the default configuration.** Only meaningful if a caller has already set `permute_fusion=True` via env var `TORCHINDUCTOR_PERMUTE_FUSION` before entering the context. |
| 10 | `"allow_buffer_reuse": False` (L94)                                 | config-flag           | `torch._inductor.config.allow_buffer_reuse` (v2.13 L252; main L252) | `d15afee671d2` (Disable buffer reuse, 2026-04-16) | Inline comment: "For now, as buffer reuse does not consider stride_map."                                     | still-required  | Upstream default `True` at both baselines. "For now" language is a stale-workaround smell; the underlying stride_map story would need to be reviewed to reclassify. |
| 11 | `Loops.has_large_inner_fn = lambda self, threshold=None: True` (L101) | monkey-patch          | `torch._inductor.ir.Loops.has_large_inner_fn` (v2.13 L1081; main L1158) | `81b10920716d` (Consolidate Spyre Inductor Patching, 2026-02-19) | L99 comment: "Force all operations to be realized when LoopLevel IR is initially constructed"                | still-required  | Signature `(self, threshold=None) -> bool` unchanged v2.13→main. Semantics unchanged: `num_ops > threshold`. Spyre forces early realization to keep the LLIR shape predictable. |
| 12 | `joint_graph.pass_patterns.pop()` (L107)                             | positional pass surgery | `torch._inductor.fx_passes.joint_graph.pass_patterns` (v2.13 L53–56; main L55–58) | `81b10920716d`                                  | L106 comment: "disable mul_softmax_pattern and div_softmax_pattern for now"                                  | needs-testing   | List is `[patterns, PatternMatcherPass()]` at both baselines — 2 elements. `.pop()` removes index 1. `mul_softmax_pattern` and `div_softmax_pattern` register into `pass_patterns[1]` (v2.13 L1054/L1089; main L1123/L1158), so the pop correctly disables both. **Fragility:** if upstream ever appends a third entry, `.pop()` would silently disable the new one and leave softmax patterns enabled. |
| 13 | `GraphLowering._update_scheduler = _spyre_update_scheduler` (L130)   | monkey-patch          | `torch._inductor.graph.GraphLowering._update_scheduler` (v2.13 L2592; main L2983) | `9d319f8b6550` (Move stickification etc. to pre-scheduler, 2026-04-14); refined by `65508a025f55` (LX relayout guard, 2026-08-14) and `2e935febe58b` (recover_spyre_hints, 2026-08-19) | Docstring block L114–L127: nested compiler contexts may wrap this hook more than once; recovery of `__spyre_dim_hints` must run *after* `decompose_auto_functionalized` retracing has installed new FX nodes. | still-required  | Upstream `_update_scheduler` body is verbatim identical v2.13 → main. Spyre's wrapper calls `old_update_scheduler(self)` unconditionally, so upstream behavior is preserved even after Spyre passes run. |
| 14 | `SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects` (L159) | monkey-patch          | `torch._inductor.scheduler.SchedulerNode.has_side_effects` (v2.13 L2563; main L3043) | `0658128f2b30` (WSR refactor, 2026-07-28); MutationLayoutSHOULDREMOVE branch added by `2e935febe58b` | Two branches: (a) L132–143 comment block — `coarse_tile.py` inserts a cross-tile-iteration copy-out whose live-out edge is invisible to the scheduler's single-pass DCE; must force `has_side_effects → True` when `_coarse_tile_force_live` is set. (b) L149–156 — `ComputedBuffer` with `MutationLayoutSHOULDREMOVE` writes into a pre-existing buffer (`copy_forced` dst); scheduler DCE cannot see that use. | needs-testing (branch a) / unknown (branch b) | Upstream `has_side_effects` body is verbatim identical v2.13 → main (both use `@cache_on_self` and check `device_assert_async` before `super().has_side_effects()`). **Fragility 1:** upstream method is `@cache_on_self` decorated; Spyre replaces the attribute *after* decoration, which is correct here because the decorator is applied at class-body evaluation and Spyre's replacement runs at CM entry — but any future upstream refactor that inlines the DCE-decision anywhere else would silently bypass the Spyre branch. **Fragility 2 (unknown):** `MutationLayoutSHOULDREMOVE` still exists in `torch/_inductor/ir.py` at both baselines (v2.13 L4886; main L5189), but the class name explicitly declares itself for removal. If upstream ever retires it, the branch becomes dead code silently — no import error, since the import at L20 would fail loudly. If upstream *renames* it, ImportError. If upstream keeps the class but stops emitting `MutationLayoutSHOULDREMOVE` layouts on the DCE path, the branch is dead and the copy_forced dst starts vanishing. |
| 15 | `post_grad.pass_patterns[2].patterns.values()` walk that swaps `extra_check` to `lambda x: False` (L186–L199) | positional pass surgery + registry mutation | `torch._inductor.fx_passes.post_grad.pass_patterns[2]` and `torch._inductor.fx_passes.post_grad.is_valid_addmm_fusion` (v2.13 L1776; main L1999) | `1bd17cc9153f` (Spyre hint improvements, 2026-07-24) | L182–L183 comment: "disable addmm fusion. The fusion will be undone by the decomposition that is registered in torch-spyre, but the hints are lost in the process." | needs-testing   | `pass_patterns` is a 3-element list at both baselines (v2.13 L85; main L86). `is_valid_addmm_fusion` is registered as `extra_check` on the two addmm patterns that live in `pass_patterns[2]` (v2.13 L1808/L1818; main L2037/L2047). Position `[2]` and the identity of the sentinel function are stable v2.13 → main. **Fragility 1:** on main, `is_valid_addmm_fusion` gained a new early return (`_PRESERVE_FLEX_GEMM_GEMM_OP` guard, main L2000–L2004) — Spyre's `lambda x: False` replacement discards that guard silently. Since the replacement is always-False the semantic is preserved (both would refuse the fusion), but a future guard change that flips to *permitting* the fusion in some case would be silently overridden by Spyre. **Fragility 2:** the assertion `addmm_fusion_found` protects the position invariant but only detects the case where the sentinel object comparison fails — it does not detect a case where upstream *adds a second addmm-style extra_check* whose object identity is different. |
| 16 | `GraphTransformObserver.apply_graph_pass = <wrapped>` (L218)         | monkey-patch          | `torch.fx.passes.graph_transform_observer.GraphTransformObserver.apply_graph_pass` (v2.13 L92; main L92) | `1bd17cc9153f`                                  | Wrapper populates `gm.meta[OBSERVER_HOOKS_KEY]` with the current `passname`/`subsystem` so downstream Spyre passes can see which upstream pass invoked them. Not documented inline. | still-required  | `apply_graph_pass(self, pass_fn)` signature identical v2.13 → main. The wrapper preserves the return value and cleans up on exit. The only v2.13→main diff in that file is `trace.provenance_tracking_level` → `effective_provenance_tracking_level()` (line 48), which is inside `__init__` and does not affect `apply_graph_pass`. Note: `patch_inductor_fusions` is called at module import (permanent monkey-patch) rather than scoped to a CM — a distinct fragility from the CM-scoped patches above. |

Row count above is 16 (one per physical override site). The
`SchedulerNode.has_side_effects` row (L14) has two independent
branches with different verdicts and both are broken out in the
verdict headline (as L14a and L14b) and in the machine-readable
ledger below.

### Machine-readable ledger

The table below is the source of truth for the counts in the
"Summary" section. `scripts/validate_metadata.py` parses this table
and asserts that (a) every row has the required columns, (b) the
`kind` column is drawn from `{mutation, config, extension-point}`,
(c) the `verdict` column is drawn from
`{still-required, needs-testing, possibly-obsolete, unknown}`, and
(d) the headline counts match the row-by-row totals derived here.
Row `id` uses the `L<n>` scheme from the human-readable table above;
L14 appears twice (L14a and L14b) to carry the two branch verdicts
independently.

<!-- machine-readable-ledger:begin -->

| id  | kind             | target                                                                                          | evidence-link                                                                                                                                                        | verdict            |
|-----|------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| L1  | config           | `torch._inductor.config.split_reductions`                                                       | [patches.py#L82](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L82)                       | still-required     |
| L2  | config           | `torch._inductor.config.benchmark_harness`                                                      | [patches.py#L83](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L83)                       | still-required     |
| L3  | extension-point  | `torch._inductor.config.pre_grad_custom_pass`                                                   | [patches.py#L84](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L84)                       | still-required     |
| L4  | extension-point  | `torch._inductor.config.post_grad_custom_pre_pass`                                              | [patches.py#L85](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L85)                       | still-required     |
| L5  | extension-point  | `torch._inductor.config.post_grad_custom_post_pass`                                             | [patches.py#L86](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L86)                       | still-required     |
| L6  | extension-point  | `torch._inductor.config._pre_fusion_custom_pass`                                                | [patches.py#L87](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L87)                       | still-required     |
| L7  | extension-point  | `torch._inductor.config._post_fusion_custom_pass`                                               | [patches.py#L88](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L88)                       | still-required     |
| L8  | config           | `torch._inductor.config.unroll_reductions_threshold`                                            | [patches.py#L91](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L91)                       | still-required     |
| L9  | config           | `torch._inductor.config.permute_fusion`                                                         | [patches.py#L93](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L93)                       | still-required     |
| L10 | config           | `torch._inductor.config.allow_buffer_reuse`                                                     | [patches.py#L94](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L94)                       | still-required     |
| L11 | mutation         | `torch._inductor.ir.Loops.has_large_inner_fn`                                                   | [patches.py#L101](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L101)                     | still-required     |
| L12 | mutation         | `torch._inductor.fx_passes.joint_graph.pass_patterns` (positional pop)                          | [patches.py#L107](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L107)                     | needs-testing      |
| L13 | mutation         | `torch._inductor.graph.GraphLowering._update_scheduler`                                         | [patches.py#L130](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L130)                     | still-required     |
| L14a| mutation         | `torch._inductor.scheduler.SchedulerNode.has_side_effects` (branch: `_coarse_tile_force_live`)  | [patches.py#L132-L143](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L132-L143)           | needs-testing      |
| L14b| mutation         | `torch._inductor.scheduler.SchedulerNode.has_side_effects` (branch: `MutationLayoutSHOULDREMOVE`) | [patches.py#L149-L156](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L149-L156)           | unknown            |
| L15 | mutation         | `torch._inductor.fx_passes.post_grad.pass_patterns[2]` + `is_valid_addmm_fusion` swap           | [patches.py#L186-L199](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L186-L199)           | needs-testing      |
| L16 | mutation         | `torch.fx.passes.graph_transform_observer.GraphTransformObserver.apply_graph_pass`              | [patches.py#L218](https://github.com/torch-spyre/torch-spyre/blob/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py#L218)                     | still-required     |

<!-- machine-readable-ledger:end -->

Row-by-row totals derived from the table above:

- Physical override sites (rows, counting L14 as one): **16**
- Verdict slots (rows with L14 split into L14a + L14b): **17**
- `kind = config`: **5** (L1, L2, L8, L9, L10)
- `kind = extension-point`: **5** (L3, L4, L5, L6, L7)
- `kind = mutation`: **7** (L11, L12, L13, L14a, L14b, L15, L16)
- `verdict = still-required`: **13** (L1–L11, L13, L16)
- `verdict = needs-testing`: **3** (L12, L14a, L15)
- `verdict = possibly-obsolete`: **0**
- `verdict = unknown`: **1** (L14b)

## Upstream behavior

Grouped by row, citing line ranges. All line numbers refer to
`/tmp/pt_v213/<file>` (from
`https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/...`)
and `/tmp/pt_main/<file>` (from
`https://raw.githubusercontent.com/pytorch/pytorch/c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62/...`).

**Rows 1–10 (config-flag overrides).**

- `split_reductions`: v2.13 L1005 and main L1061 both declare
  `split_reductions = os.getenv("TORCHINDUCTOR_SPLIT_REDUCTIONS", "1") == "1"` — default `True`.
- `benchmark_harness`: v2.13 L271 and main L275 both `benchmark_harness = True`.
- `pre_grad_custom_pass`: v2.13 L313, main L317 — both `= None` typed as `CustomGraphPassType`.
- `post_grad_custom_pre_pass` / `post_grad_custom_post_pass`: v2.13 L300–L301, main L304–L305 — both `= None`.
- `_pre_fusion_custom_pass` / `_post_fusion_custom_pass`: v2.13 L318 and L325, main L322 and L329 — declared with the same type expression; used with equivalent semantics.
- `unroll_reductions_threshold`: v2.13 L991 and main L1047 both `= 8`. Referenced from `ir.py` at v2.13 L1732/L2310 and main L1905/L2518 (both compare `int(reduction_numel) < config.unroll_reductions_threshold`; setting to `1` disables the small-reduction unroll for `reduction_numel >= 1`, effectively always).
- `permute_fusion`: v2.13 L1518 and main L1642 both `= os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"` — default `False`.
- `allow_buffer_reuse`: v2.13 L252 and main L252 both `= True`.

**Row 11 (`Loops.has_large_inner_fn`).**

- v2.13 L1081–L1093: reads `config.realize_opcount_threshold`, falls
  back to CPU or default threshold, returns
  `self.inner_fn_opcount().num_ops > threshold`.
- main L1158–L1161: identical semantics, refactored into
  `self.inner_fn_opcount().num_ops > self.get_realize_opcount_threshold(threshold)`.
- Signature `(self, threshold: int | None = None) -> bool` unchanged.
- There are also `has_large_inner_fn` methods on `IRNode` (v2.13 L823;
  main L872) and on the `TritonTemplateBuffer`-adjacent class (v2.13
  L9855; main L10759). Spyre patches only `Loops.has_large_inner_fn`;
  the other two remain live.

**Row 12 (`joint_graph.pass_patterns`).**

- v2.13 L53–L56 and main L55–L58 both declare
  `pass_patterns = [patterns, PatternMatcherPass()]` — a length-2
  list.
- `mul_softmax_pattern` and `div_softmax_pattern` register into
  `pass_patterns[1]` at v2.13 L1054 and L1089 and at main L1123 and
  L1158.
- `.pop()` at L107 of `patches.py` removes index 1 (the
  `PatternMatcherPass()`), which is precisely where the softmax
  patterns register. Comment claim is correct at both baselines.

**Row 13 (`GraphLowering._update_scheduler`).**

- v2.13 L2592–L2601 and main L2983–L2992: verbatim identical body
  (`with config.patch("triton.store_cubin", False): self.scheduler = Scheduler(self.operations)`).
- `codegen()` calls `self._update_scheduler()` unconditionally (v2.13
  L2607; main L2998), which is the entry point Spyre's wrapper wraps.

**Row 14 (`SchedulerNode.has_side_effects`).**

- v2.13 L2562–L2567 and main L3042–L3047: verbatim identical body,
  including the `@cache_on_self` decorator, the `_body is None`
  guard, and the `device_assert_async` check delegating to
  `super().has_side_effects()`.
- `MutationLayoutSHOULDREMOVE` referenced by Spyre is defined at v2.13
  L4886 and main L5189. The class name self-declares as scheduled for
  removal.

**Row 15 (`post_grad.pass_patterns[2]` + `is_valid_addmm_fusion`).**

- v2.13 L85–L89 and main L86–L90: `pass_patterns = [PatternMatcherPass(), PatternMatcherPass(), PatternMatcherPass()]`.
- Two addmm patterns with `extra_check=is_valid_addmm_fusion` and
  `pass_dict=pass_patterns[2]` at v2.13 L1800–L1818 and main
  L2029–L2047.
- `is_valid_addmm_fusion` body:
  - v2.13 L1776–L1797: shape / dtype / expandability checks.
  - main L1999–L2026: same checks preceded by an
    `_PRESERVE_FLEX_GEMM_GEMM_OP` early-return.
- No new addmm-style patterns were added between v2.13 and main;
  count of matches to swap is still 2 in both cases.

**Row 16 (`GraphTransformObserver.apply_graph_pass`).**

- v2.13 L92–L103 and main L92–L103: identical signature and body.
- Attributes `self.passname` and `self.subsystem` used by the Spyre
  wrapper are set in `__init__` at v2.13 L38–L39 and main L38–L39 in
  both baselines.
- The only diff in the file between v2.13 and main is line 48
  (`trace.provenance_tracking_level == 1` →
  `effective_provenance_tracking_level() == 1`), inside `__init__`
  and unrelated to `apply_graph_pass`.

## Hidden assumption or duplicated knowledge

Every row of the ledger is a private assertion that some upstream
behavior is wrong-for-Spyre. Each such assertion has an implicit
temporal scope — "as of the SHA we were tested against." That scope
is never written down. The load-bearing invariants encoded here that
no test enforces:

1. **Positional shape of `joint_graph.pass_patterns` is exactly 2**
   (row 12). `.pop()` is silently incorrect the day upstream appends a
   third entry.
2. **Positional shape of `post_grad.pass_patterns` is exactly 3, and
   the position that holds addmm patterns is index `[2]`** (row 15).
   `patches.py` protects this with `assert addmm_fusion_found`, but
   the assert only fires if the identity comparison against
   `is_valid_addmm_fusion` fails — it does not detect *migration* of
   the same sentinel to another pass_dict index, nor addition of a
   second addmm sentinel.
3. **`Loops.has_large_inner_fn` is the sole entry point for
   opcount-driven early realization decisions** (row 11). Both
   sibling methods on `IRNode` (v2.13 L823; main L872) and on the
   template-buffer class (v2.13 L9855; main L10759) are left
   unpatched. The assumption is that only `Loops.has_large_inner_fn`
   is called on the Spyre code path — untested.
4. **`SchedulerNode.has_side_effects` is the sole DCE hook** (row
   14). `BaseSchedulerNode.has_side_effects` at v2.13 L1463 / main
   L1894 and other override sites (v2.13 L2157, L2848; main L2602,
   L3342) are unpatched.
5. **`MutationLayoutSHOULDREMOVE` will keep existing and keep being
   emitted on ComputedBuffers whose writes the scheduler cannot
   otherwise see** (row 14, branch b). The class name itself is a
   loud upstream promise-to-remove.
6. **`is_valid_addmm_fusion` is the sole predicate whose flip
   suppresses addmm fusion in `pass_patterns[2]`** (row 15). Upstream
   could split it into two sentinels (one per pattern) without
   breaking any test — Spyre's assertion would still pass and one of
   the two entries would remain live.
7. **The `permute_fusion=False` override is meaningful** (row 9). It
   is a no-op unless someone has set `TORCHINDUCTOR_PERMUTE_FUSION=1`
   in the environment. This is documented nowhere.
8. **`GraphTransformObserver.apply_graph_pass` is called with the
   `gm.meta` mutable and readable across nested passes** (row 16).
   The wrapper installs `pass`/`subsystem` in `gm.meta[OBSERVER_HOOKS_KEY]`
   on enter and pops on exit — this only works if `self.gm` is the
   same object that downstream Spyre passes read from. `__init__`
   at both baselines stores `self.gm = gm` verbatim, so today this is
   safe.

All eight of these invariants would benefit from being turned into
one-line assertions at `enable_spyre_context` entry — a runtime
version-mismatch surface, not a silent slippage.

## Evidence

Verbatim quotes from `patches.py` @ `fea0c4b`
(`https://raw.githubusercontent.com/torch-spyre/torch-spyre/fea0c4be901e1383b1f700dbad8887128b0fcb27/torch_spyre/_inductor/patches.py`):

L82–L95, config overrides:

```python
    new_config = {
        "split_reductions": False,
        "benchmark_harness": False,
        "pre_grad_custom_pass": CustomPreGradPasses(),
        "post_grad_custom_pre_pass": CustomPrePasses(),
        "post_grad_custom_post_pass": CustomPostPasses(),
        "_pre_fusion_custom_pass": CustomPreFusionPasses(),
        "_post_fusion_custom_pass": CustomPostFusionPasses(),
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        "unroll_reductions_threshold": 1,
        # Disable fusing of mm + permute/transpose for now.
        "permute_fusion": False,
        "allow_buffer_reuse": False,  # For now, as buffer reuse does not consider stride_map.
    }
```

L97–L107, `Loops.has_large_inner_fn` + `joint_graph.pass_patterns.pop`:

```python
    from torch._inductor.ir import Loops

    # Force all operations to be realized when LoopLevel IR is initially constructed
    old_loop = Loops.has_large_inner_fn
    Loops.has_large_inner_fn = lambda self, threshold=None: True

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()
```

L109–L130, `GraphLowering._update_scheduler` wrapper:

```python
    old_update_scheduler = GraphLowering._update_scheduler

    _pre_scheduling_pass = CustomPreSchedulingPasses()

    def _spyre_update_scheduler(self: GraphLowering) -> None:
        # Nested compiler contexts may wrap this hook more than once. The
        # graph-mutating pre-scheduling pipeline runs once per GraphLowering.
        if not getattr(self, "_spyre_pre_scheduling_complete", False):
            # recover_spyre_hints runs here (after all post-grad FX passes including
            # decompose_auto_functionalized) rather than in CustomPostPasses.
            # decompose_auto_functionalized replaces auto_functionalized_v2 nodes
            # via make_fx retracing, which creates new FX nodes whose meta["custom"]
            # only contains hints from the innermost scope. Running recovery here
            # ensures the final FX graph (post-decomposition) gets the full hint set.
            gm = self.graph.owning_module
            if gm is not None and "__spyre_dim_hints" in gm.meta:
                recover_spyre_hints(self.graph)
            _pre_scheduling_pass(self)
            setattr(self, "_spyre_pre_scheduling_complete", True)
        old_update_scheduler(self)

    GraphLowering._update_scheduler = _spyre_update_scheduler  # type: ignore[method-assign]
```

L132–L159, `SchedulerNode.has_side_effects` wrapper:

```python
    # coarse_tile.py's nested output-dim + reduction-dim tiling
    # (_propagate_tiled_reduction_op) inserts a copy-out op
    # (_insert_reduction_copy_op) that mutates a pre-loop accumulation buffer
    # (accum_full) so its updated value is visible to the NEXT outer-tile
    # iteration's copy-in. That cross-iteration read has no representation in
    # the single-pass, pre-unroll IR the scheduler's own dead_node_elimination
    # walks, so a copy-out with no other downstream reader looks dead and is
    # removed — even though it is required for correctness. Mark such ops
    # with _coarse_tile_force_live (see _insert_reduction_copy_op) and force
    # SchedulerNode.has_side_effects() to report True for them, mirroring how
    # upstream itself protects effectful FallbackKernels from the same DCE
    # pass (torch/_inductor/lowering.py, effectful op handling).
    old_scheduler_node_has_side_effects = SchedulerNode.has_side_effects

    def _spyre_scheduler_node_has_side_effects(self: SchedulerNode) -> bool:
        if getattr(self.node, "_coarse_tile_force_live", False):
            return True
        # ComputedBuffers with MutationLayoutSHOULDREMOVE write into a
        # pre-existing buffer (e.g. copy_forced dst). The scheduler's own DCE
        # doesn't know about this layout convention and marks them dead when
        # no downstream op reads the output name. Keep them live.
        if isinstance(self.node, ComputedBuffer) and isinstance(
            self.node.layout, MutationLayoutSHOULDREMOVE
        ):
            return True
        return old_scheduler_node_has_side_effects(self)

    SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects  # type: ignore[method-assign]
```

L180–L218, `patch_inductor_fusions` (module-permanent; not scoped to
the CM):

```python
def patch_inductor_fusions():
    import torch._inductor.fx_passes.post_grad

    # disable addmm fusion. The fusion will be undone by the decomposition that is
    # registered in torch-spyre, but the hints are lost in the process
    addmm_fusion_found = False
    for entries in torch._inductor.fx_passes.post_grad.pass_patterns[
        2
    ].patterns.values():
        for entry in entries:
            if (
                entry.extra_check
                == torch._inductor.fx_passes.post_grad.is_valid_addmm_fusion
            ):
                entry.extra_check = lambda x: False
                addmm_fusion_found = True

    assert addmm_fusion_found, (
        "Couldn't find addmm fusion. This patch needs to be reviewed."
    )

    # Install observer patch
    from torch.fx.passes.graph_transform_observer import GraphTransformObserver

    _original = GraphTransformObserver.apply_graph_pass

    @wraps(GraphTransformObserver.apply_graph_pass)
    def apply_graph_pass(self, pass_fn):
        meta = self.gm.meta.get(OBSERVER_HOOKS_KEY, {})
        self.gm.meta[OBSERVER_HOOKS_KEY] = meta
        meta["pass"] = self.passname
        meta["subsystem"] = self.subsystem
        try:
            return _original(self, pass_fn)
        finally:
            meta.pop("pass", None)
            meta.pop("subsystem", None)

    GraphTransformObserver.apply_graph_pass = apply_graph_pass
```

Upstream context excerpts:

`torch/_inductor/fx_passes/joint_graph.py` L53–L56 (v2.13) and L55–L58 (main):

```python
pass_patterns = [
    patterns,
    PatternMatcherPass(),
]
```

`torch/_inductor/fx_passes/joint_graph.py` v2.13 L1054 (identical
pattern at main L1123):

```python
        pass_dict=pass_patterns[1],
```

`torch/_inductor/fx_passes/post_grad.py` L85–L89 (v2.13) and L86–L90 (main):

```python
# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]
```

`is_valid_addmm_fusion` on main (L1999–L2004) — first four lines
that Spyre's `lambda x: False` swap silently discards:

```python
def is_valid_addmm_fusion(match):
    if any(
        node.target is aten.mm.default and node.meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP)
        for node in match.nodes
    ):
        return False
```

## Reproducer or proof

Static reading is sufficient for the ledger claims. Runtime
reproducers apply per verdict:

- **`still-required` config overrides (rows 1–10, most of 11, 13,
  16):** the "removal test" is: comment the override out, compile the
  Spyre test suite. Expected result: correctness regression or kernel
  count change. This is the passing-today / would-fail-if-broken
  variant of the proof (README template case b).
- **`needs-testing` (rows 12, 15, and branch a of 14):** unit test
  that asserts on positional invariants at CM entry. Sketch given
  under "Suggested change" below.
- **`unknown` (branch b of 14):** requires either an upstream release
  note declaring `MutationLayoutSHOULDREMOVE` gone, or a runtime
  probe:
  ```python
  from torch._inductor.ir import MutationLayoutSHOULDREMOVE, ComputedBuffer
  # After compiling a Spyre graph that includes copy_forced:
  # enumerate scheduler nodes, count those whose .node is a ComputedBuffer
  # with MutationLayoutSHOULDREMOVE layout. If count == 0 on both v2.13
  # and main under the same input, the branch has never fired and is dead.
  # If count > 0 on v2.13 and == 0 on main, upstream stopped emitting
  # that layout on this path — the copy_forced dst is now vanishing.
  ```

## Compile-time impact

Not measured. Config overrides themselves are effectively free (a
dict update inside `config.patch`). Monkey-patches replace class
attributes at CM entry (constant overhead). The
`patch_inductor_fusions` walk over `pass_patterns[2].patterns.values()`
is O(N patterns) at import time, N is small (dozens at most in
practice) — negligible.

## Runtime impact

The overrides *are* the runtime-impact surface — every one of them
changes what upstream Inductor emits. Splitting apart per-flag
impact is a separate audit (would need `TORCH_LOGS=inductor` diffs
with each override toggled).

## Correctness impact

**Row 12 (`joint_graph.pass_patterns.pop()`) — silent scope
expansion risk.** If upstream appends a new pass at index 2 (making
the list length 3) and rearranges softmax patterns to any other
index, `.pop()` would remove the *new* pass and leave the softmax
patterns active. No test would fail; the compiled graph would
silently start being reshaped in a way torch-spyre never expected.

**Row 14b (`MutationLayoutSHOULDREMOVE` branch) — silent DCE of
`copy_forced` dst.** If upstream stops emitting `MutationLayoutSHOULDREMOVE`
on the `copy_forced` code path (renames the layout, moves the
mutation to a different mechanism), the branch is dead. The copy_forced
destination buffer then reverts to being invisible to the
scheduler's DCE and is dropped — a wrong-answer regression on any
model using `copy_forced`. This is the bug this branch was written
to prevent, and the branch predicate is the only thing keeping it
from occurring. There is currently no test that would fail if the
branch were removed while `MutationLayoutSHOULDREMOVE` still exists
but stops being emitted here.

**Row 15 (`is_valid_addmm_fusion` swap) — silent hint loss on new
addmm-style patterns.** If upstream adds a second addmm-style pattern
in `pass_patterns[2]` with a *different* `extra_check` sentinel, the
Spyre walker would leave that pattern's fusion enabled. The
downstream decomposition would still undo the fusion but the hints
would be lost — the exact regression the comment says the patch is
protecting against.

For the removal test on `possibly-obsolete` verdicts: there are none
in this batch, so no removal test is required. If a later audit
reclassifies row 9 (`permute_fusion=False`) or row 10
(`allow_buffer_reuse=False`) to `possibly-obsolete` on the grounds
that the upstream implementation now handles Spyre's stride model
correctly, the test would be: enter `enable_spyre_context` with the
override commented out, compile a representative model that exercises
buffer reuse across strided layouts, verify (a) numerical
equivalence, (b) no change in scheduled kernel count, (c) no change
in peak HBM.

## Measurement needed (if any)

None for the ledger claims themselves — they are static. Two
follow-up measurements would strengthen row-level verdicts:

1. **Row 14b runtime probe.** On the dev pod:
   ```bash
   # In a Spyre venv with torch-spyre installed and TORCH_LOGS=inductor
   python -c "import torch_spyre; import torch; \
     from torch._inductor.ir import MutationLayoutSHOULDREMOVE; \
     print('present:', MutationLayoutSHOULDREMOVE)"
   # Then compile the copy_forced test model and count SchedulerNodes
   # whose .node.layout is MutationLayoutSHOULDREMOVE.
   ```
2. **Row 12 / row 15 position probe.** Run:
   ```python
   from torch._inductor.fx_passes import joint_graph, post_grad
   assert len(joint_graph.pass_patterns) == 2
   assert len(post_grad.pass_patterns) == 3
   sentinels_addmm = sum(
     1
     for entries in post_grad.pass_patterns[2].patterns.values()
     for entry in entries
     if entry.extra_check == post_grad.is_valid_addmm_fusion
   )
   assert sentinels_addmm == 2
   ```
   as a pytest-executable check at `enable_spyre_context` entry.

## Suggested change

Consolidate all sixteen overrides into a single
`torch_spyre/_inductor/upstream_compat.py` module structured as
follows.

1. A single `SUPPORTED_TORCH_VERSIONS = (">=2.13,<2.14",)` constant
   at the top, with an explicit `assert` at CM entry against
   `torch.__version__`.
2. One named function per override, each self-contained and
   documented with (a) what upstream default is, (b) why it is wrong
   for Spyre, (c) the removal test that would prove the override
   still matters. E.g.:
   ```python
   def override_split_reductions() -> ContextManager:
       """Spyre backend has no split-reduction codegen; upstream default True.
       Removal test: enable, compile MODEL_X, expect kernel count change.
       """
       return _config_flag("split_reductions", False, default_upstream=True)
   ```
3. Explicit positional-invariant assertions replacing implicit ones:
   ```python
   def _assert_joint_graph_shape() -> None:
       from torch._inductor.fx_passes import joint_graph
       if len(joint_graph.pass_patterns) != 2:
           raise UpstreamShapeMismatch(
             "joint_graph.pass_patterns changed length; "
             "the softmax-pattern pop was written against len==2. "
             "Re-audit before proceeding."
           )
   ```
   with analogous checks for `post_grad.pass_patterns` length,
   addmm-sentinel count, and `MutationLayoutSHOULDREMOVE` existence.
4. `patch_inductor_fusions` becomes CM-scoped rather than a
   module-level permanent mutation. Today the addmm-fusion swap and
   the `GraphTransformObserver.apply_graph_pass` wrap persist for the
   whole process lifetime after `patch_inductor_fusions()` is called
   once — that is invisible to a caller who sees only
   `enable_spyre_context` as the "entry point."
5. `permute_fusion=False` (row 9) should either grow a comment
   explaining that it only matters when
   `TORCHINDUCTOR_PERMUTE_FUSION=1` is set externally, or be removed
   with a note that the default is False anyway.

Each override becomes a single named symbol, and the ledger in this
finding becomes trivially machine-checkable against that module.

## Skill / contract update

Create `contracts/upstream-private-api.yaml` (previewed in
`contracts/README.md` as a planned stub). First-pass schema, keyed
by upstream symbol:

```yaml
- symbol: torch._inductor.fx_passes.joint_graph.pass_patterns
  kind: list  # positional access
  assumptions:
    - "length == 2"
    - "index 1 is the PatternMatcherPass into which mul_softmax_pattern and div_softmax_pattern register"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L107 (joint_graph.pass_patterns.pop())
  last_verified_at:
    v2.13.0: cf30153c4c131c8164ee7798e5022d810682e2cb
    main:    c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62
  drift_signature: "grep -c '^pass_patterns = \\[' torch/_inductor/fx_passes/joint_graph.py"

- symbol: torch._inductor.fx_passes.post_grad.pass_patterns
  kind: list  # positional access
  assumptions:
    - "length == 3"
    - "index 2 holds the addmm/bmm-style patterns"
    - "is_valid_addmm_fusion is a single module-level sentinel used as extra_check for exactly two entries"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L186-L199

- symbol: torch._inductor.ir.Loops.has_large_inner_fn
  kind: method
  assumptions:
    - "signature (self, threshold: int | None = None) -> bool"
    - "sole entry point for opcount-driven early realization decisions on the Spyre code path"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L100-L101

- symbol: torch._inductor.graph.GraphLowering._update_scheduler
  kind: method
  assumptions:
    - "called exactly once per GraphLowering by codegen()"
    - "wrappable by attribute reassignment"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L109-L130

- symbol: torch._inductor.scheduler.SchedulerNode.has_side_effects
  kind: method
  assumptions:
    - "decorated with @cache_on_self; attribute reassignment replaces the decorated descriptor"
    - "sole DCE hook consulted for SchedulerNode instances"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L144-L159

- symbol: torch._inductor.ir.MutationLayoutSHOULDREMOVE
  kind: class
  assumptions:
    - "class name declares removal intent; behavior may be silently retired"
    - "emitted on ComputedBuffer layouts whose writes the scheduler DCE cannot see (e.g. copy_forced dst)"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L20 (import)
    - torch_spyre/_inductor/patches.py:L149-L156 (has_side_effects branch)
  removal_signal: "grep -c 'class MutationLayoutSHOULDREMOVE' torch/_inductor/ir.py"

- symbol: torch._inductor.fx_passes.post_grad.is_valid_addmm_fusion
  kind: function (identity-compared)
  assumptions:
    - "identity-stable module-level function"
    - "unique sentinel: sole extra_check function used for addmm fusion decisions in pass_patterns[2]"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L192

- symbol: torch.fx.passes.graph_transform_observer.GraphTransformObserver.apply_graph_pass
  kind: method
  assumptions:
    - "signature (self, pass_fn: Callable[[Graph], T]) -> T | None"
    - "self.gm.meta is mutable and readable by nested passes"
    - "self.passname and self.subsystem are set in __init__"
  local_dependents:
    - torch_spyre/_inductor/patches.py:L204-L218

- symbol: torch._inductor.config.<flag>
  kind: config-flag (bulk)
  flags:
    - split_reductions   # default True
    - benchmark_harness  # default True
    - unroll_reductions_threshold  # default 8
    - permute_fusion     # default False  (Spyre override is no-op unless env var set)
    - allow_buffer_reuse # default True
    - pre_grad_custom_pass       # install slot
    - post_grad_custom_pre_pass  # install slot
    - post_grad_custom_post_pass # install slot
    - _pre_fusion_custom_pass    # install slot (name is "private")
    - _post_fusion_custom_pass   # install slot (name is "private")
  local_dependents:
    - torch_spyre/_inductor/patches.py:L81-L95
```

The drift-watch workflow (referenced from `contracts/README.md`) can
diff the raw source at each `symbol`'s upstream file between
`last_verified_at.main` and the current pytorch `HEAD` on a schedule
and file a `findings/upstream-fragility/*.md` finding when the
`drift_signature` grep changes.
