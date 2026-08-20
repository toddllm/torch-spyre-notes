# Contracts

What torch-spyre assumes about upstream Inductor and about its own
loop-level IR.

## What lives here

One file per contract. A contract is a set of invariants + the
upstream symbols they depend on + the local sites that maintain (or
implicitly rely on) those invariants.

Contracts are the durable output of the audit. A finding under
`findings/` is a specific defect; a contract is the general rule the
defect reveals. When a finding is resolved, the contract is updated
so that the next audit or the next AI reviewer knows to check it.

## Planned contracts (stubs)

Each is a placeholder for now — populated as findings resolve.

- `graphlowering.md` — buffer/operation registration and identity;
  `name_to_buffer` / `name_to_users` / `name_to_op` /
  `removed_buffers` conventions; `_update_scheduler` hook.
- `computed-buffer.md` — the nine-invariant replacement contract
  from `cases/replace-computed-buffer-body.md`; the wrap-not-rebuild
  rule for `inner_fn`; stride-aware redirection.
- `dependency-extraction.md` — what `get_read_writes()` and
  `extract_read_writes()` cost; when a result is safe to cache; what
  invalidates it.
- `scheduler.md` — dead-code semantics, side-effect handling, when
  the scheduler is constructed relative to LLIR passes.
- `layouts.md` — layout / stride semantics; when a name-only
  redirect is unsafe.
- `upstream-private-api.yaml` — machine-readable index of every
  private upstream symbol torch-spyre depends on, with local
  dependent sites. Feeds the upstream-drift-watch workflow.

## Convention: contracts cite findings, and vice versa

A contract's "Evidence" section links every finding that motivated
it. A finding's "Skill / contract update" section names the
contract file it should update. The two directions stay in sync
through the audit review process.
