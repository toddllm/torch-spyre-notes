# Audit-deepening coverage matrix — 2026-08-20

Machine-generated join of the Phase 1 / Phase 5 scanner results
against the Phase 0 dynamic-coverage data captured on the pod. This
file is the audit-database's "what did we actually cover?" ledger for
the `audit-deepening-2026-08-20` branch. Every table below is
regenerable from the pinned inputs — nothing here is hand-written
prose about numbers that live elsewhere.

## Inputs

- **torch-spyre SHA:** `fea0c4be901e1383b1f700dbad8887128b0fcb27`
  (2026-08-20)
- **pytorch supported baseline:** `v2.13.0` @
  `cf30153c4c131c8164ee7798e5022d810682e2cb`
- **pytorch main HEAD:** `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62`
  (2026-08-20)
- **Static-analysis inputs:** `scans/results/*.json` +
  `scans/results/pass_dependency_graph.dot`
- **Dynamic-analysis inputs:**
  [`measurements/2026-08-20/data/test_flash.jsonl`](../measurements/2026-08-20/data/test_flash.jsonl)
  (12,022 events; a single `test_flash.py` cold compile on the
  `a5-deepview` dev pod, instrumented via
  [`measurements/2026-08-20/patches/instrument_read_writes.py`](../measurements/2026-08-20/patches/instrument_read_writes.py)
  after the Phase 0 exclusive-timing / object-identity rework)
- **Parent manifest:**
  [`reports/2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md`](2026-08-20__torch-spyre-fea0c4b__pytorch-c3ebaab.md)

## Scanner totals

| Scanner | Files scanned | Total hits | Result file |
|---|---:|---:|---|
| `graph_mutations` | 292 | 63 | [`scans/results/graph_mutations.json`](../scans/results/graph_mutations.json) |
| `private_api` | 101 | 756 | [`scans/results/private_api.json`](../scans/results/private_api.json) |
| `repeated_analysis` | 302 | 124 | [`scans/results/repeated_analysis.json`](../scans/results/repeated_analysis.json) |
| `list_surgery` | 302 | 53 | [`scans/results/list_surgery.json`](../scans/results/list_surgery.json) |
| `workarounds` | 731 | 500 | [`scans/results/workarounds.json`](../scans/results/workarounds.json) |
| `test_smells` | 148 | 766 | [`scans/results/test_smells.json`](../scans/results/test_smells.json) |
| `runtime_artifact` | 3 fixtures | see fixtures | [`scans/results/runtime_artifact.json`](../scans/results/runtime_artifact.json) |
| `pass_dependency_graph` | n/a (graph builder) | see DOT | [`scans/results/pass_dependency_graph.dot`](../scans/results/pass_dependency_graph.dot) |

The `pass_dependency_graph` scanner emits a DOT file plus a plain-text
edge summary
([`scans/results/pass_dependency_graph.report.txt`](../scans/results/pass_dependency_graph.report.txt));
counts vary with the pass registry it was pointed at and are not
directly comparable to the hit-count column.

## Workload dynamic coverage (test_flash.jsonl)

- Events captured: **12,022**
  - `get_read_writes_raw`: 7,050
  - `op_read_writes_memo`: 4,971
  - `instrument_installed`: 1
- Distinct caller functions observed: **24**

The workload was a single `test_flash.py` cold compile, so the
absolute event counts are one compile's worth of dependency-extraction
work, not a broad sample. See the "Laptop-only limitations and
needs-pod backlog" section below for what a fuller sweep would look
like.

## Hot-caller cross-check (workload × `repeated_analysis`)

For each `caller_func` observed calling `get_read_writes` in the
`test_flash` workload, the number of scan hits under
`scans/repeated_analysis.py` that anchor to a call site in that
function's body. A `no` in the *Scan covers?* column means the hot
caller was seen dynamically but the static scanner missed the call
site — a scanner-precision gap.

| Caller function | Workload calls | Scan hits | Scan covers? |
|---|---:|---:|:---:|
| `_redirect_consumers` | 2,460 | 1 | yes |
| `_build_indirect_load_subs` | 2,431 | 1 | yes |
| `iteration_space_from_op` | 2,130 | 1 | yes |
| `_prepare_per_core_view` | 1,318 | 2 | yes |
| `_is_index_or_indirectly_accessed` | 490 | 2 | yes |
| `collect_tensor_deps` | 410 | 1 | yes |
| `get_free_symbol_uses` | 213 | 0 | **no** |
| `op_read_writes` | 205 | 1 | yes |
| `span_reduction` | 205 | 1 | yes |
| `apply_splits` | 205 | 1 | yes |
| `live_operations` | 197 | 2 | yes |
| `propagate_spyre_tensor_layouts` | 197 | 3 | yes |
| `_resolve_copy_back_candidates` | 197 | 3 | yes |
| `compute_future_min_cost` | 197 | 1 | yes |
| `_compute_last_use` | 197 | 1 | yes |
| `beam_global_min_cost` | 197 | 1 | yes |
| `work_distribution` | 189 | 1 | yes |
| `_op_inputs_good_for_lx_inplace` | 185 | 1 | yes |
| `get_fill_order` | 177 | 0 | **no** |
| `validate_ops` | 165 | 1 | yes |
| `get_read_indices` | 16 | 0 | **no** |
| `insert_bmm_padding` | 16 | 1 | yes |
| `_cost_model_divide_op` | 16 | 2 | yes |
| `_clone_output_splits` | 8 | 2 | yes |

Three coverage misses stand out:

- **`get_free_symbol_uses` (213 calls, 0 scan hits).** Reads
  read/write sets indirectly via `get_read_writes()`-derived symbol
  sets; the scanner's caller-anchoring only matches direct calls.
- **`get_fill_order` (177 calls, 0 scan hits).** Upstream Inductor
  code path — the scanner walks the torch-spyre tree only, and the
  workload's calls originate in `torch/_inductor/ir.py:5362` (per the
  instrumentation caller_file field). This is a scanner-scope
  limitation, not a bug: the scan is pinned to `torch-spyre`.
- **`get_read_indices` (16 calls, 0 scan hits).** Same shape as
  `get_free_symbol_uses` — indirect read-set consumer.

Two of the three misses are honest scope calls (the scanner is
scoped to torch-spyre). The `get_free_symbol_uses` / `get_read_indices`
class is a real scanner-precision gap — the scanner should also flag
functions that receive read/write sets from a caller. Filed as a
scanner follow-up in the outstanding-issues section below.

## Scan-only callers (`repeated_analysis` flagged, workload never
called)

Total scanner-only callers: **72**. These are candidates for a wider
workload sweep — the scanner sees them statically but `test_flash`
doesn't exercise them. The top 15 by scan-hit count:

| Scanner-flagged caller | Scan hits |
|---|---:|
| `collect_lx_relayout_plans` | 4 |
| `collect_feeders` | 3 |
| `_cd_parent_matches` | 2 |
| `_clone_divisions_and_matches` | 2 |
| `_cores` | 2 |
| `_enum_split_options` | 2 |
| `_general_tile_advance` | 2 |
| `_hbm_pattern` | 2 |
| `_insert_combine_op` | 2 |
| `_insert_copy_op` | 2 |
| `_insert_one_read_copy` | 2 |
| `_matmul_axis_parse` | 2 |
| `_matmul_features` | 2 |
| `_row_split` | 2 |
| `enforce_indirect_access_layout` | 2 |

The full list is in `scans/results/repeated_analysis.json`; a wider
workload sweep on the pod is queued under
[`needs-pod/02-profile-test-flash.sh`](../needs-pod/02-profile-test-flash.sh)
and
[`needs-pod/03-instrument-hot-callers.sh`](../needs-pod/03-instrument-hot-callers.sh).

## Findings per category (on-disk)

| Category | Count |
|---|---:|
| `correctness` | 1 |
| `compile-time` | 2 |
| `runtime` | 0 |
| `duplication` | 0 |
| `upstream-fragility` | 4 |
| `test-gaps` | 0 |
| `maintainability` | 0 |

Total findings on-disk under this manifest: **7**.

## Contracts populated

| Contract | Status | Motivating findings / scans |
|---|---|---|
| [`contracts/dependency-extraction.md`](../contracts/dependency-extraction.md) | populated | `findings/compile-time/02`, `scans/repeated_analysis.py`, Phase 0 test_flash measurements |
| [`contracts/graphlowering.md`](../contracts/graphlowering.md) | populated | `findings/correctness/01`, `scans/graph_mutations.py` |
| [`contracts/computed-buffer.md`](../contracts/computed-buffer.md) | populated | `cases/replace-computed-buffer-body.md`, `scans/graph_mutations.py` |
| [`contracts/pass-matrix.md`](../contracts/pass-matrix.md) | populated | `findings/upstream-fragility/03`, `scans/pass_dependency_graph.py` |
| `contracts/scheduler.md` | stub | not this run |
| `contracts/layouts.md` | stub | not this run |
| `contracts/upstream-private-api.yaml` | stub | Phase 1 `private_api.json` is the seed |

## Laptop-only limitations and needs-pod backlog

Every finding in this batch is either static (proven by line-anchored
reading against the three pinned SHAs) or backed by a single cold
compile of `test_flash.py` on the `a5-deepview` dev pod. The
following remain queued for the pod:

- [`needs-pod/01-constant-graph-output-repro.sh`](../needs-pod/01-constant-graph-output-repro.sh)
  — adversarial repro of the `dedup_and_promote_constants` guard-vs-drop
  misalignment under `findings/correctness/01-...-not-observed.md`.
- [`needs-pod/02-profile-test-flash.sh`](../needs-pod/02-profile-test-flash.sh)
  — full-compile cProfile pass; picks hot callers by measured wall
  time so scanner-only callers get a chance to fire.
- [`needs-pod/03-instrument-hot-callers.sh`](../needs-pod/03-instrument-hot-callers.sh)
  — instruments the top-N callers from step 02 using the Phase 2
  patches (`instrument_build_indirect_load_subs.py`,
  `instrument_get_fill_order.py`, `instrument_iteration_space.py`).
- [`needs-pod/04-parallel-compile-metamorphic.py`](../needs-pod/04-parallel-compile-metamorphic.py)
  — quantifies the wall-clock cost of the process-global
  `enable_spyre_lowerings` RLock under concurrent compiles
  (finding `03-lowering-registry-lock.md`).
- [`needs-pod/05-process-contamination.py`](../needs-pod/05-process-contamination.py)
  — interleaved Spyre / non-Spyre compiles in one process to surface
  patch state that outlives the context manager
  (finding `04-monkey-patches-outside-patches-py.md`).

The Phase 2 profiling harness (`profile_whole_compile.py`,
`analyze_profile.py`, `paired_measure.py`) has been dry-run on the
laptop against fixture JSONL, but the wall-clock numbers it emits are
only meaningful once the pod runs the driver scripts above.

## Outstanding issues (surfaced by this coverage pass)

1. **`get_free_symbol_uses` / `get_read_indices` scanner miss.** The
   `repeated_analysis.py` scanner flags direct callers of
   `get_read_writes` but misses functions that consume the derived
   symbol / index sets. Extend the scanner or add a companion pass.
2. **Workload breadth.** `test_flash.py` reaches only 24 of 96
   scanner-flagged callers (24 workload / 72 scan-only). The
   `needs-pod/02-...` sweep should widen the workload set.
3. **Findings-category imbalance.** Three of the seven investigation
   classes (`runtime`, `duplication`, `test-gaps`, `maintainability`)
   have zero findings on-disk under this manifest. The Phase 1
   scanners (`test_smells`, `workarounds`, `graph_mutations`) already
   produced 500+ hits between them; the gap is that hits haven't been
   promoted to findings yet.
