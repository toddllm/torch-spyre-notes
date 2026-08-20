# Findings

One finding per file. Filename: `<seq>-<kebab-slug>.md` within its
category subdirectory. Category subdirectories are:

- `correctness/` — the code produces the wrong answer under some
  condition, or an invariant it relies on is not enforced.
- `compile-time/` — the compiler does work it does not need to do.
  Runtime output of the compiled program is unchanged.
- `runtime/` — the compiled program does work it does not need to
  do. Compile-time cost may be unchanged.
- `duplication/` — the same knowledge exists in multiple places, and
  the copies can drift.
- `upstream-fragility/` — the code depends on a private or
  positional aspect of upstream that could change silently.
- `test-gaps/` — a test claims to enforce an invariant but does not
  fail when the invariant is broken.
- `maintainability/` — the code is correct and fast but structurally
  costly to modify.

A finding belongs to exactly one category. If it looks like it
belongs to two, split it.

## Template

Every finding uses this template. Sections marked *required* must be
present in some form. Sections marked *optional* may be omitted if
they truly do not apply — but "not measured yet" is not the same as
"does not apply" and should say so explicitly.

```markdown
# <short imperative title>

- **Category:** correctness | compile-time | runtime | duplication | upstream-fragility | test-gaps | maintainability
- **Revision manifest:** [reports/<date>__<ts-sha>__<pt-sha>.md](../../reports/<...>.md)
- **Confidence:** proven | reproduced | plausible | speculative
- **Status:** open | needs-measurement | needs-repro | resolved | not-a-bug

## Summary
<one paragraph — what was found, in the language a compiler engineer would use>

## Files and symbols
- torch-spyre: `path/to/file.py` — `symbol_name` (line range, permalinked)
- upstream v2.13.0: `torch/_inductor/...` — `Symbol.method` (line range, permalinked)
- upstream main: `torch/_inductor/...` — `Symbol.method` (line range, permalinked)

## Observed behavior
<what the code does today, at the pinned torch-spyre SHA>

## Upstream behavior
- **v2.13.0 (supported baseline):** <what upstream does at revision B>
- **main:** <what upstream does at revision C>

## Hidden assumption or duplicated knowledge
<the contract, invariant, or knowledge fragment that the code encodes
implicitly. This is the load-bearing part of the finding — it is what
distinguishes "here's some code" from "here's a maintenance liability">

## Evidence
<verbatim code excerpts with line anchors. If quoting, quote exactly —
no paraphrase. If claiming a call is inside a nested loop, quote both
loops>

## Reproducer or proof
<one of:
  (a) a runnable test that fails today,
  (b) a runnable test that passes today and would fail if the
      production code were changed in a specified way,
  (c) an impossibility proof: cite the invariant and the code that
      enforces it, showing the suspicious state cannot occur.>

## Compile-time impact
<not measured | <numbers with method>>

## Runtime impact
<not measured | <numbers with method>>

## Correctness impact
<none | <specific model / input class that produces wrong output, or the invariant that would be violated>>

## Measurement needed (if any)
<exact commands, exact environment (which dev pod, which torch-spyre
worktree), what to capture>

## Suggested change
<smallest structural fix that removes the whole class, if such a fix
exists. Not "rewrite everything" — a specific helper or assertion or
schema change.>

## Skill / contract update
<what a future audit or an AI reviewer should learn from this finding.
Point at the specific file under contracts/ that should be updated, or
propose a new one.>
```

## Confidence levels

- **proven** — a test in this repo, or a runnable reproducer, exhibits
  the behavior. Anyone can run it and see the result.
- **reproduced** — behavior was reproduced at least once by the
  author; the exact command list is in the finding, but no permanent
  test exists yet.
- **plausible** — static reading of the code makes the behavior look
  likely, but no run has been done.
- **speculative** — reasoning from upstream diffs or from analogy to
  another finding; no reading of the specific code has confirmed it.

Findings at `plausible` or `speculative` should be labeled as such in
the title (e.g., "*Plausible:* dedup runs `_drop_constant` on skipped
graph outputs"). The label goes away when confidence rises.

## Status transitions

- New finding → `open`.
- Finding needs runtime numbers → `needs-measurement`. The
  "Measurement needed" section has to be filled in for this status.
- Finding needs a reproducer script → `needs-repro`.
- Finding fix landed upstream (torch-spyre or pytorch) → `resolved`.
  Include the commit SHA and short reasoning about why the fix
  closes the finding.
- Finding investigated and no bug exists → `not-a-bug`. Include the
  impossibility proof. Do not delete — the negative result is
  evidence.
