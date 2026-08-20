# torch-spyre-notes

An audit database for the torch-spyre ↔ PyTorch Inductor boundary.

## What this repo is

An out-of-tree PyTorch backend that uses upstream Inductor as a
compiler framework is coupled to Inductor internals that are not
designed as a stable public compiler-pass API. That coupling shows up
as duplicated logic, repeated expensive analysis, suppressed upstream
optimizations, stale compatibility workarounds, and implicit
invariants that no test enforces. Each of those is discoverable, but
only if the discovery is versioned, categorized, and reproducible.

This repo stores the results of that discovery.

## Layout

```
torch-spyre-notes/
├── README.md                    # this file — operating brief
├── cases/                       # microscope: one helper or pass, first-principles
├── contracts/                   # what the code assumes about upstream
├── findings/                    # evidence + reproduction per issue
│   ├── correctness/
│   ├── compile-time/
│   ├── runtime/
│   ├── duplication/
│   ├── upstream-fragility/
│   ├── test-gaps/
│   └── maintainability/
├── scans/                       # AST/grep tools that surface candidates
└── reports/                     # one manifest per audit run: SHAs + env
```

The case-study files under `cases/` explain individual helpers and
passes for humans. The material under `contracts/`, `findings/`,
`scans/`, and `reports/` is the audit database itself.

## The three-repo model

Every finding is expressed against **three exact revisions**:

- **A.** `torch-spyre` at the SHA being audited.
- **B.** `pytorch` at the release torch-spyre officially supports
  (currently `v2.13.0`, per `pyproject.toml`'s `torch~=2.13.0`).
- **C.** `pytorch` at current `main`.

**B** answers *"are we correct today?"* **C** answers *"what is about
to become obsolete or break?"* A finding that names only "PyTorch does
X" is incomplete — say which of A/B/C the claim is about.

Reports under `reports/` freeze the exact SHAs for one audit run. See
`reports/README.md` for the manifest format.

## The ten investigation classes

Every finding belongs to exactly one class. This taxonomy exists so
that "duplicate code" and "this adds 80 ms to compilation" do not end
up in the same bucket.

| Class                     | What to look for                                                                             | Directory                     |
|---------------------------|----------------------------------------------------------------------------------------------|-------------------------------|
| Duplicated compiler logic | Two implementations of equivalent graph edits, handlers, metadata copying                    | `findings/duplication/`       |
| Repeated expensive analysis | `get_read_writes()`, SymPy tracing, coordinate calculation, dependency extraction         | `findings/compile-time/`      |
| Repeated whole-graph scans | Loops over all operations once per buffer / op / duplicate                                 | `findings/compile-time/`      |
| Repeated list surgery     | `operations.index/remove/insert` inside loops                                                | `findings/compile-time/`      |
| Redundant compiler work   | Recompile/retrace/recalculate the same result multiple times per compilation                  | `findings/compile-time/`      |
| Suppressed upstream opts  | Config flags or monkey patches that disable a PyTorch optimization                            | `findings/upstream-fragility/`|
| Stale workarounds         | TODOs, "for now," version-specific hacks whose upstream cause may be gone                    | `findings/upstream-fragility/`|
| Implicit LLIR contracts   | Manual updates to `name_to_buffer`, `name_to_users`, caches, provenance, outputs             | `findings/correctness/` or `duplication/` |
| Weak / vacuous tests      | Tests that would stay green if the invariant they claim to protect were deliberately broken   | `findings/test-gaps/`         |
| Upstream semantic drift   | Anything touching `torch._inductor`, `torch._dynamo`, private registries, class fields       | `findings/upstream-fragility/`|

## Ways of looking that a clone detector misses

Two identical functions are ordinary duplication. Duplicated
**knowledge** is different: five passes each know a *different*
partial rule about the same contract (e.g., replacing a computed
buffer requires updating `name_to_buffer`; another pass knows it
requires clearing `get_default_sizes_body`; a third knows about
provenance forwarding; a fourth about `name_to_users`; a fifth about
index rewrites when strides differ). Those functions can look
completely different to a clone detector while collectively
representing five partial copies of the same compiler contract.

Cluster mutation sites by *semantic operation* — replace a
`ComputedBuffer`, redirect a buffer dependency, insert an operation,
retire an operation, clone a buffer, change layout, translate
indexing — not by exact syntax. See `contracts/` for the
authoritative list of semantic operations this codebase performs.

## Rules of the audit

1. **Never produce an unversioned finding.** Every finding names the
   `reports/*.md` manifest it was written against.
2. **Do not call something a performance bug without measurement.**
   Torch-Spyre already times each pre-scheduling pass with
   `time.perf_counter()` and logs elapsed ms. Use those first. Only
   instrument demonstrated hot passes. Findings that require a
   running compilation must include the exact command list under a
   "Measurement needed" section.
3. **Do not call something a correctness bug without a reproducer or
   proof of impossibility.** When a suspicious branch appears,
   either write the test that forces it, or prove the invariant that
   makes it unreachable and cite the code that enforces the
   invariant.
4. **Audit tests adversarially.** For every test that claims to
   enforce an internal invariant, temporarily break the production
   invariant and confirm the test fails. A test that stays green is
   evidence in itself.
5. **Compare local workarounds to both B and C.** For every
   monkey-patch, disabled config, `TODO`, or "we believe" comment:
   check the upstream state at v2.13.0 and at main. Some workarounds
   have outlived their bugs.
6. **Prefer systemic fixes.** One reusable helper over five local
   patches. One centralized cache/invalidation policy over ad-hoc
   memoization. One graph mutation API over manual side-table
   repair. One compatibility adapter over scattered version
   conditionals.
7. **Convert every confirmed lesson into three things:** a regression
   test, the smallest reusable helper or assertion that encodes the
   invariant, and an update to the relevant contract file. A bug
   should make future reviews smarter, not merely disappear.

## The finding schema

See `findings/README.md` for the exact template every finding follows.
Every finding must include, at minimum: category, revision manifest,
files/symbols, observed behavior, upstream behavior at B and C, the
hidden assumption or duplicated knowledge, evidence (line-anchored
citations), reproducer or proof, compile-time impact, runtime impact,
correctness impact, confidence, suggested change, and any
skill/contract update the finding implies.
