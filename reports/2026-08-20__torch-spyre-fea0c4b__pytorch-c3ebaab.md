# Audit manifest — 2026-08-20

## Revisions

- **torch-spyre main HEAD:** `fea0c4be901e1383b1f700dbad8887128b0fcb27` (2026-08-20)
- **pytorch supported baseline:** `v2.13.0` @ `cf30153c4c131c8164ee7798e5022d810682e2cb`
- **pytorch main HEAD:** `c3ebaabaf8fe1d1bf25475e86fddafbcbd339e62` (2026-08-20)

## Declared version constraint

From `torch-spyre/pyproject.toml`:

```toml
"torch~=2.13.0",
```

`~=2.13.0` compatible-release matches `>=2.13.0, <2.14`. The
supported baseline for this audit is therefore `v2.13.0`. Any finding
that claims "upstream does X" is asserting behavior at
`cf30153c...` unless it explicitly names the main-HEAD SHA above.

## Environment

**Static analysis:**
- Auditor host: laptop (macOS).
- All static findings under `findings/correctness/`,
  `findings/upstream-fragility/`, and the initial static portions of
  `findings/compile-time/` were produced by reading source at the
  three pinned SHAs above.

**Runtime measurement:**
- Pod: `tdeshane-compiler-timing-dev-v2` (node `p1-worker-23`) on the
  `a5-deepview` cluster.
- torch-spyre worktree: `$HOME/torch-spyre-work/torch-spyre` checked
  out to `fea0c4be901e1383b1f700dbad8887128b0fcb27` for this audit.
  `_C.so` rebuilt via `make setup` on the pod against the pod's
  current SDK; `import torch_spyre._C` and `torch.device("spyre")`
  both verified working after rebuild.
- torch: `2.13.0+cpu` (matches declared `torch~=2.13.0`).
- Python: 3.12.
- Workload: torch-spyre's own `test_flash` closure at baseline dims
  (B=1, H=8, D=128, Lq=512, Lk=1024, b_block_size=1, h_block_size=4,
  q_block_size=256, kv_block_size=512), driven by
  [`measurements/2026-08-20/scripts/run_test_flash.py`](../measurements/2026-08-20/scripts/run_test_flash.py).
- Instrumentation:
  [`measurements/2026-08-20/patches/instrument_read_writes.py`](../measurements/2026-08-20/patches/instrument_read_writes.py)
  — monkey-patches `ComputedBuffer.get_read_writes` and
  `torch_spyre._inductor.pass_utils.op_read_writes` at import time
  and writes JSONL. No torch-spyre source modification.
- Cold-compile hygiene: `TORCHINDUCTOR_CACHE_DIR` wiped between runs.
- Raw data: [`measurements/2026-08-20/data/test_flash.jsonl`](../measurements/2026-08-20/data/test_flash.jsonl)
  (12,022 records, 90 s cold compile).
- A synthetic dedup-stress workload
  ([`run_synthetic_dedup.py`](../measurements/2026-08-20/scripts/run_synthetic_dedup.py))
  was attempted to isolate the O(D · N) scan; it aborted mid-compile
  with a Spyre-backend layout restriction
  (`Unexpected stick expression 1`) before
  `dedup_and_promote_constants` ran. Partial data at
  [`data/synthetic_dedup_partial.jsonl`](../measurements/2026-08-20/data/synthetic_dedup_partial.jsonl)
  covers only `deadcode_elimination`. The `test_flash` numbers were
  sufficient to close the compile-time findings without it.

## Scope of this run

Three investigations, all static:

1. `dedup_constants` correctness edge case (`_redirect_consumers`
   early-returning on graph outputs while `_drop_constant` runs
   unconditionally) plus the O(D × N) scan pattern.
2. `get_read_writes()` and `op_read_writes()` call-site inventory
   across torch-spyre — enclosing loop nesting, memoization
   coverage, cache miss surface. Measurement of per-call cost is
   flagged as dev-pod work.
3. `patches.py` optimization-suppression ledger — for every upstream
   config override and monkey-patch, record the introducing commit,
   the upstream state at that commit, the upstream state at
   `v2.13.0`, and the upstream state at current `main`.

## Findings produced

- [`findings/correctness/01-dedup-constants-graph-output-not-a-bug.md`](../findings/correctness/01-dedup-constants-graph-output-not-a-bug.md)
  — `_redirect_consumers` guard vs `_drop_constant` misalignment is
  unreachable; proved by enumerating every `SpyreConstantFallback`
  construction site. Filed `not-a-bug` with a suggested defensive
  assertion.
- [`findings/compile-time/01-dedup-constants-quadratic-scan.md`](../findings/compile-time/01-dedup-constants-quadratic-scan.md)
  — `_redirect_consumers` scans `graph.operations` per duplicate
  constant and calls raw `get_read_writes()` on every op it visits;
  measured at 2,460 calls / 616 ms — the #1 hottest
  `get_read_writes` site in the whole `test_flash` compile. Status
  `open`, confidence `reproduced`.
- [`findings/compile-time/02-get-read-writes-inventory.md`](../findings/compile-time/02-get-read-writes-inventory.md)
  — 101 static call sites; 12,021 runtime calls / 1.86 s in a
  90 s cold compile; 68 of 101 sites bypass the memoized helper;
  same op re-analyzed 50–54 times.
- [`findings/upstream-fragility/01-patches-ledger.md`](../findings/upstream-fragility/01-patches-ledger.md)
  — 15 upstream overrides in `patches.py`; 11 `still-required`, 3
  `needs-testing`, 1 `unknown`. Names `_PRESERVE_FLEX_GEMM_GEMM_OP`
  as a new upstream-main pre-condition that the addmm-fusion swap
  silently discards.

## Deferred to a future run

- Verdict on the one `unknown` row in the `patches.py` ledger
  (`SchedulerNode.has_side_effects` for `MutationLayoutSHOULDREMOVE`
  writes on `copy_forced`) — needs a runtime probe.
- Removal tests for the three `needs-testing` overrides in
  `patches.py`.
- A synthetic dedup-stress workload that actually reaches
  `dedup_and_promote_constants`; the initial attempt hit a
  Spyre-backend layout restriction before the pass ran.
- The other seven investigation classes from the operating brief
  (list surgery, FX recompile, test smells beyond the one flagged,
  positional upstream coupling audit, semantic-diff between v2.13
  and main for tracked symbols, stale-workaround sweep,
  duplicated-knowledge cluster analysis of `NameSwapHandler`
  variants). These get their own manifests.
- Adversarial audit of the `test_no_orphans_in_name_to_buffer` test
  claim from the operating brief. Needs a runnable pytest on the
  pod.
