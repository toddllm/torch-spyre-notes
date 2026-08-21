# Reports

One report per audit run. Filename:

```
<YYYY-MM-DD>__torch-spyre-<short-sha>__pytorch-<short-sha>.md
```

A report freezes the exact revisions and environment under which a
batch of findings was produced. Every finding references its report
in the "Revision manifest" field so that the finding remains
interpretable even after all three trees have moved.

## Manifest format

```markdown
# Audit manifest — <date>

## Revisions

- **torch-spyre main HEAD:** `<full sha>` (`<commit date>`)
- **pytorch supported baseline:** `<tag>` @ `<full sha>` (per torch-spyre `pyproject.toml`)
- **pytorch main HEAD:** `<full sha>` (`<commit date>`)

## Declared version constraint

<quote from torch-spyre pyproject.toml: `torch~=X.Y.Z`>

## Environment

- Auditor host: <e.g., laptop static, or a Spyre-capable dev host for measurement>
- Python: <version if a run happened; N/A for static-only reports>
- Notes: <anything else that affects reproducibility>

## Scope of this run

<what was investigated in this batch. Point at the findings/*.md
files produced under this manifest>

## Findings produced

- [findings/<category>/<file>.md](../findings/<...>) — <one-line summary>
- ...

## Deferred to a future run

<items in scope that could not be completed here, and why. Usually
"needs measurement on the dev pod" or "needs a longer archaeological
git-log walk on pytorch">
```

## When to write a new report

- Any time you resolve fresh revisions (torch-spyre `main` has moved,
  or you're checking against a new PyTorch main SHA).
- Any time the audit environment changes (running on the dev pod
  vs. static laptop analysis).
- Any time findings from a distinct question are produced as a batch.

Do not overwrite old reports. Old reports are the audit trail.
