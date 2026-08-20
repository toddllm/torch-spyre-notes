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

- Auditor host: laptop, static analysis only.
- Python: N/A for this batch (no runs).
- Notes: Any measurement-based finding produced under this manifest
  must include a "Measurement needed" section pointing at the dev
  pod. See individual findings for the specific pod and commands.

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

*Populated as each investigation completes:*

- `findings/correctness/*.md` — dedup_constants edge case
- `findings/compile-time/*.md` — dedup_constants O(D×N), get_read_writes inventory
- `findings/upstream-fragility/*.md` — patches.py ledger

## Deferred to a future run

- Runtime numbers for `get_read_writes` call frequency and per-call
  latency. Requires a running torch-spyre compilation on the dev
  pod; the inventory finding names the exact commands.
- The other seven investigation classes from the operating brief
  (list surgery, FX recompile, test smells beyond the one flagged,
  positional upstream coupling audit, semantic-diff between v2.13
  and main for tracked symbols, stale-workaround sweep,
  duplicated-knowledge cluster analysis of `NameSwapHandler`
  variants). These get their own manifests.
- Adversarial audit of the `test_no_orphans_in_name_to_buffer` test
  claim from the operating brief. Deferred to the dedup_constants
  investigation's follow-up (needs a runnable pytest).
