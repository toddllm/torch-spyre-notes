#!/usr/bin/env python3
"""Validate audit-repo metadata: finding front-matter, ledger rows, SHAs, links, counts.

Checks performed:
1.  Every finding under `findings/**/*.md` (excluding `findings/README.md`)
    has a bullet-style front-matter with the required keys:
    `id`, `category`, `status`, `confidence`, `created`.
2.  `status` is drawn from {open, in-progress, resolved, not-a-bug,
    not-observed, superseded}.
3.  `confidence` is drawn from {plausible, likely, confirmed}.
4.  Every row of the `<!-- machine-readable-ledger:begin -->` / `:end`
    table in `findings/upstream-fragility/01-patches-ledger.md` has:
    `id`, `kind`, `target`, `evidence-link`, `verdict`.
    `kind` must be one of {mutation, config, extension-point}.
    `verdict` must be one of
    {still-required, needs-testing, possibly-obsolete, unknown}.
5.  Every 40-hex-character SHA cited in a finding or in the top-level
    README matches one of the pinned SHAs declared in the manifest
    report(s) under `reports/`.
6.  Every internal Markdown link `[text](path)` in a finding, in
    `findings/README.md`, in `README.md`, and in `reports/*.md`
    resolves to an existing file inside the repo. Links may contain a
    `#fragment` — the fragment is dropped for the file-exists check.
7.  Any per-category finding-count claim made in `README.md` matches
    the actual number of finding files under
    `findings/<category>/*.md`. If `README.md` makes no such claim,
    this check is skipped (no error).
8.  Prose count claims (e.g. "N upstream overrides", "N physical
    override sites", "N verdict rows", "N config overrides",
    "N mutation sites", "N extension-point") in any .md file under
    the repo (excluding the patches ledger itself and .git/) match
    the canonical totals derived from the machine-readable ledger.
9.  Every `torch-spyre@<SHORT_SHA>:...` shorthand citation uses the
    exact short SHA declared as `TORCH_SPYRE_SHORT_SHA` below —
    catches drift when someone re-pins.

Exit status:
    0   no errors, no warnings that promote to error mode
    1   any error printed to stderr

Usage:
    scripts/validate_metadata.py                # validate the repo
    scripts/validate_metadata.py --repo <path>  # validate a different tree

The script uses only the Python standard library so it can run in the
pre-commit hook without a virtualenv.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants — the vocabulary the audit repo commits to.
# ---------------------------------------------------------------------------

# Task spec §2:
ALLOWED_STATUS = {
    "open",
    "in-progress",
    "resolved",
    "not-a-bug",
    "not-observed",
    "superseded",
}
ALLOWED_CONFIDENCE = {"plausible", "likely", "confirmed"}

REQUIRED_FINDING_KEYS = ("id", "category", "status", "confidence", "created")

ALLOWED_LEDGER_KIND = {"mutation", "config", "extension-point"}
ALLOWED_LEDGER_VERDICT = {
    "still-required",
    "needs-testing",
    "possibly-obsolete",
    "unknown",
}
REQUIRED_LEDGER_COLUMNS = ("id", "kind", "target", "evidence-link", "verdict")

CATEGORY_DIRS = (
    "correctness",
    "compile-time",
    "runtime",
    "duplication",
    "upstream-fragility",
    "test-gaps",
    "maintainability",
)

SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

# Bullet-style front-matter: `- **Key:** value`
# (The colon lives *inside* the `**…**` bolding, per the templates in
# `findings/README.md`.)
FRONT_MATTER_LINE_RE = re.compile(
    r"^-\s+\*\*(?P<key>[A-Za-z][A-Za-z0-9 /_-]*):\*\*\s*(?P<value>.+?)\s*$"
)

LEDGER_BEGIN = "<!-- machine-readable-ledger:begin -->"
LEDGER_END = "<!-- machine-readable-ledger:end -->"

# The one-and-only short SHA that torch-spyre citations must use. If a
# future audit re-pins torch-spyre, bump this value together with the
# full SHAs recorded in the manifest reports; the validator will then
# fail any lingering shorthand that still points at the old SHA.
TORCH_SPYRE_SHORT_SHA = "fea0c4b"

# Short-SHA shorthand of the form `torch-spyre@<hex>:...`. We only flag
# citations whose hex prefix is *shorter* than a full 40-char SHA — the
# full form is validated by the manifest cross-check (rule 5).
TORCH_SPYRE_SHORT_RE = re.compile(
    r"torch-spyre@(?P<sha>[0-9a-f]{4,39})(?=[:`\s])"
)

# Prose count claims we cross-check against the ledger's row-derived
# totals. Each entry pairs a regex (that captures a leading integer `N`)
# with a callable that, given the parsed ledger totals, returns the
# expected value for `N`. The regex is deliberately anchored on the
# phrase alone — we don't try to reject synonyms, we just verify the
# ones we *do* recognize.
#
# The `ledger totals` dict passed to the callables has keys:
#   physical_rows       — count of rows in the machine-readable table
#   verdict_rows        — same as physical_rows unless an id ends in
#                         'a' or 'b' (branch split)
#   by_kind             — dict of kind → count
#   by_verdict          — dict of verdict → count
PROSE_COUNT_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "upstream overrides",
        re.compile(r"\b(\d+)\s+upstream\s+overrides?\b", re.IGNORECASE),
        "physical_rows",
    ),
    (
        "physical override sites",
        re.compile(r"\b(\d+)\s+physical\s+override\s+sites?\b", re.IGNORECASE),
        "physical_rows",
    ),
    (
        "verdict rows",
        re.compile(r"\b(\d+)\s+verdict\s+rows?\b", re.IGNORECASE),
        "verdict_rows",
    ),
    (
        "config overrides",
        re.compile(r"\b(\d+)\s+config\s+overrides?\b", re.IGNORECASE),
        "kind:config",
    ),
    (
        "mutation sites",
        re.compile(r"\b(\d+)\s+mutation\s+sites?\b", re.IGNORECASE),
        "kind:mutation",
    ),
    (
        "extension-point",
        re.compile(r"\b(\d+)\s+extension-points?\b", re.IGNORECASE),
        "kind:extension-point",
    ),
]

# The count claims we recognize in README.md, matching phrases like
# "findings/correctness (N)" or "N findings under correctness/". We
# stay conservative — if the README doesn't have a machine-derivable
# count, we skip.
README_COUNT_RE = re.compile(
    r"findings/(?P<category>[a-z-]+)/?\s*[(]\s*(?P<count>\d+)\s*[)]"
)


# ---------------------------------------------------------------------------
# Error accumulator.
# ---------------------------------------------------------------------------

@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    checks_run: int = 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def summary(self) -> str:
        line = (
            f"validate_metadata: {self.checks_run} check(s) run, "
            f"{len(self.errors)} error(s)"
        )
        if self.errors:
            return line + " — FAIL"
        return line + " — OK"


# ---------------------------------------------------------------------------
# Front-matter parsing (bullet style: `- **Key:** value`).
# ---------------------------------------------------------------------------

def parse_front_matter(text: str) -> dict[str, str]:
    """Extract `- **Key:** value` bullets appearing at the top of a finding.

    The scan stops at the first blank line that follows a bullet block,
    the first `## ` heading, or the end of the file — whichever comes
    first. Keys are lower-cased.
    """
    fm: dict[str, str] = {}
    seen_a_bullet = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            break
        if not line.strip():
            if seen_a_bullet:
                break
            continue
        m = FRONT_MATTER_LINE_RE.match(line)
        if m:
            seen_a_bullet = True
            key = m.group("key").strip().lower().replace(" ", "-")
            value = m.group("value").strip()
            fm[key] = value
        # Non-bullet, non-heading lines are ignored (e.g. the H1 title).
    return fm


# ---------------------------------------------------------------------------
# Ledger table parsing.
# ---------------------------------------------------------------------------

def parse_ledger_rows(ledger_text: str) -> tuple[list[dict[str, str]], list[str]]:
    """Extract rows from the fenced `<!-- machine-readable-ledger:begin -->`
    Markdown table.

    Returns (rows, errors). Each row is a dict keyed by the (lower-cased,
    stripped) header cell. Row-level structural problems are reported
    to `errors`; per-column semantic checks are done by the caller.
    """
    errors: list[str] = []
    if LEDGER_BEGIN not in ledger_text or LEDGER_END not in ledger_text:
        errors.append(
            "patches ledger: missing "
            f"`{LEDGER_BEGIN}` / `{LEDGER_END}` fence — cannot parse "
            "machine-readable ledger table"
        )
        return [], errors

    block = ledger_text.split(LEDGER_BEGIN, 1)[1].split(LEDGER_END, 1)[0]
    table_lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith("|")
    ]
    if len(table_lines) < 3:
        errors.append(
            f"patches ledger: expected header + separator + at least "
            f"one data row inside the fence, found {len(table_lines)} "
            "table line(s)"
        )
        return [], errors

    def split_row(line: str) -> list[str]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return cells

    header = [h.lower() for h in split_row(table_lines[0])]
    # Second line is the separator (---).
    rows: list[dict[str, str]] = []
    for i, line in enumerate(table_lines[2:], start=3):
        cells = split_row(line)
        if len(cells) != len(header):
            errors.append(
                f"patches ledger row {i}: column count {len(cells)} "
                f"differs from header column count {len(header)}"
            )
            continue
        rows.append(dict(zip(header, cells)))
    return rows, errors


# ---------------------------------------------------------------------------
# Manifest-report SHA extraction.
# ---------------------------------------------------------------------------

def pinned_shas_from_reports(reports_dir: Path) -> tuple[set[str], list[str]]:
    """Every 40-hex SHA that appears in any file under `reports/` is a
    pinned SHA for the audit. Returns (shas, errors).
    """
    errors: list[str] = []
    shas: set[str] = set()
    if not reports_dir.is_dir():
        errors.append(f"reports directory not found: {reports_dir}")
        return shas, errors
    any_manifest = False
    for report in sorted(reports_dir.glob("*.md")):
        if report.name.lower() == "readme.md":
            continue
        any_manifest = True
        text = report.read_text(encoding="utf-8", errors="replace")
        shas.update(SHA_RE.findall(text))
    if not any_manifest:
        errors.append(f"no manifest reports found under {reports_dir}")
    return shas, errors


# ---------------------------------------------------------------------------
# Internal-link resolution.
# ---------------------------------------------------------------------------

def check_internal_links(
    md_path: Path, repo_root: Path, report: Report
) -> None:
    """For every `[text](target)` in `md_path`, resolve internal targets
    (no scheme) against the file's directory and confirm the file
    exists inside `repo_root`. External URLs (http/https/mailto/data)
    are skipped.
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    md_dir = md_path.parent

    for m in MD_LINK_RE.finditer(text):
        target = m.group(2).strip()
        if not target:
            continue
        if ":" in target.split("/", 1)[0]:
            # `http:`, `https:`, `mailto:`, `data:` etc. — external.
            continue
        if target.startswith("#"):
            # Same-file fragment, skip.
            continue
        if "<" in target or ">" in target:
            # Template placeholder like `../../reports/<...>.md` in
            # findings/README.md's schema template. Not a real link.
            continue
        # Drop the fragment for file-exists check.
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        # `?query=...` isn't used in this repo but strip defensively.
        path_part = path_part.split("?", 1)[0]
        # Resolve relative to the file's directory.
        try:
            resolved = (md_dir / path_part).resolve()
        except OSError as exc:
            report.error(
                f"{md_path.relative_to(repo_root)}: cannot resolve link "
                f"target {target!r}: {exc}"
            )
            continue
        # The link must land inside the repo (defensive; a `..` chain
        # that escapes is a lint failure too).
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            report.error(
                f"{md_path.relative_to(repo_root)}: internal link "
                f"{target!r} escapes repo root ({resolved})"
            )
            continue
        if not resolved.exists():
            report.error(
                f"{md_path.relative_to(repo_root)}: internal link "
                f"{target!r} does not resolve to an existing file "
                f"(expected {resolved})"
            )


# ---------------------------------------------------------------------------
# Finding front-matter validation.
# ---------------------------------------------------------------------------

def check_finding_front_matter(
    finding_path: Path, repo_root: Path, report: Report
) -> None:
    text = finding_path.read_text(encoding="utf-8", errors="replace")
    fm = parse_front_matter(text)
    rel = finding_path.relative_to(repo_root)
    for key in REQUIRED_FINDING_KEYS:
        report.checks_run += 1
        if key not in fm:
            report.error(
                f"{rel}: front-matter missing required key `{key}`"
            )
    # Status vocabulary.
    if "status" in fm:
        report.checks_run += 1
        # Some findings use parenthetical qualifiers, e.g. `open (needs
        # measurement)`; take just the head token before whitespace or
        # a parenthesis.
        head = re.split(r"[\s(]", fm["status"].strip(), 1)[0]
        if head not in ALLOWED_STATUS:
            report.error(
                f"{rel}: status {fm['status']!r} not in "
                f"{sorted(ALLOWED_STATUS)}"
            )
    # Confidence vocabulary.
    if "confidence" in fm:
        report.checks_run += 1
        head = re.split(r"[\s(]", fm["confidence"].strip(), 1)[0]
        if head not in ALLOWED_CONFIDENCE:
            report.error(
                f"{rel}: confidence {fm['confidence']!r} not in "
                f"{sorted(ALLOWED_CONFIDENCE)}"
            )


# ---------------------------------------------------------------------------
# Ledger row validation and count-cross-check.
# ---------------------------------------------------------------------------

def check_ledger(
    ledger_path: Path, repo_root: Path, report: Report
) -> dict[str, object] | None:
    """Validate the ledger table and return the row-derived canonical totals,
    or `None` if the table could not be parsed. Callers (rule 8) cross-check
    prose claims against the returned totals.
    """
    text = ledger_path.read_text(encoding="utf-8", errors="replace")
    rows, parse_errors = parse_ledger_rows(text)
    rel = ledger_path.relative_to(repo_root)
    for err in parse_errors:
        report.error(f"{rel}: {err}")
    if not rows:
        return None
    # Per-row schema.
    seen_ids: set[str] = set()
    for i, row in enumerate(rows, start=1):
        for col in REQUIRED_LEDGER_COLUMNS:
            report.checks_run += 1
            if col not in row or not row[col]:
                report.error(
                    f"{rel}: ledger row {i} missing required column "
                    f"`{col}`"
                )
        if "id" in row and row["id"]:
            if row["id"] in seen_ids:
                report.error(
                    f"{rel}: ledger row {i} duplicate id {row['id']!r}"
                )
            seen_ids.add(row["id"])
        if "kind" in row and row["kind"]:
            report.checks_run += 1
            if row["kind"] not in ALLOWED_LEDGER_KIND:
                report.error(
                    f"{rel}: ledger row {i} kind {row['kind']!r} not in "
                    f"{sorted(ALLOWED_LEDGER_KIND)}"
                )
        if "verdict" in row and row["verdict"]:
            report.checks_run += 1
            if row["verdict"] not in ALLOWED_LEDGER_VERDICT:
                report.error(
                    f"{rel}: ledger row {i} verdict {row['verdict']!r} "
                    f"not in {sorted(ALLOWED_LEDGER_VERDICT)}"
                )

    # Cross-check headline counts against the row-derived totals.
    kind_counts: dict[str, int] = {k: 0 for k in ALLOWED_LEDGER_KIND}
    verdict_counts: dict[str, int] = {v: 0 for v in ALLOWED_LEDGER_VERDICT}
    for row in rows:
        k = row.get("kind", "")
        v = row.get("verdict", "")
        if k in kind_counts:
            kind_counts[k] += 1
        if v in verdict_counts:
            verdict_counts[v] += 1
    # The headline table in the Summary section names each count with a
    # `**N**` marker plus a phrase; we parse those out lazily and
    # cross-check. If the headline doesn't declare a count for a
    # dimension, we don't error — but we do check the ones it does.
    # Pattern: `| `kind` | N |` or `| ``kind`` | N |`
    for kind, expected in kind_counts.items():
        pattern = re.compile(
            r"\|\s*`?" + re.escape(kind) + r"`?\s*\|\s*(\d+)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            declared = int(m.group(1))
            report.checks_run += 1
            if declared != expected:
                report.error(
                    f"{rel}: headline count for kind `{kind}` is "
                    f"{declared} but row-derived total is {expected}"
                )
    for verdict, expected in verdict_counts.items():
        pattern = re.compile(
            r"\|\s*`?" + re.escape(verdict) + r"`?\s*\|\s*(\d+)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            declared = int(m.group(1))
            report.checks_run += 1
            if declared != expected:
                report.error(
                    f"{rel}: headline count for verdict `{verdict}` "
                    f"is {declared} but row-derived total is {expected}"
                )

    # Physical vs verdict rows: rows whose id ends with a lowercase 'a' or
    # 'b' are branch-splits of a single physical override site. Total
    # `verdict_rows` counts every ledger row; `physical_rows` collapses
    # each `<base>a` + `<base>b` pair into one physical site.
    verdict_rows = len(rows)
    branch_bases: set[str] = set()
    for row in rows:
        rid = row.get("id", "")
        if rid and rid[-1:] in ("a", "b"):
            branch_bases.add(rid[:-1])
    # Each unique branch base counts as one physical row; its two branch
    # rows would otherwise have been counted as two.
    physical_rows = verdict_rows - len(branch_bases)

    totals: dict[str, object] = {
        "physical_rows": physical_rows,
        "verdict_rows": verdict_rows,
        "by_kind": dict(kind_counts),
        "by_verdict": dict(verdict_counts),
    }
    return totals


# ---------------------------------------------------------------------------
# SHA cross-check.
# ---------------------------------------------------------------------------

def check_shas(
    md_paths: Iterable[Path],
    pinned: set[str],
    repo_root: Path,
    report: Report,
) -> None:
    for path in md_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in SHA_RE.finditer(text):
            sha = m.group(0)
            report.checks_run += 1
            if sha not in pinned:
                report.error(
                    f"{path.relative_to(repo_root)}: SHA {sha} is not "
                    f"one of the pinned SHAs declared in any "
                    f"reports/*.md manifest"
                )


# ---------------------------------------------------------------------------
# README count cross-check.
# ---------------------------------------------------------------------------

def check_readme_counts(
    readme_path: Path, repo_root: Path, report: Report
) -> None:
    """If README.md claims a per-category finding count via the
    `findings/<category>/ (N)` phrase, verify N matches the on-disk
    count of `<seq>-<slug>.md` files (excluding `README.md`).
    """
    if not readme_path.is_file():
        return
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    for m in README_COUNT_RE.finditer(text):
        category = m.group("category")
        declared = int(m.group("count"))
        cat_dir = repo_root / "findings" / category
        if not cat_dir.is_dir():
            report.error(
                f"README.md claims count for `findings/{category}/` but "
                f"that directory does not exist"
            )
            continue
        actual = sum(
            1
            for p in cat_dir.glob("*.md")
            if p.name.lower() != "readme.md"
        )
        report.checks_run += 1
        if actual != declared:
            report.error(
                f"README.md claims `findings/{category}/` has "
                f"{declared} finding(s) but directory contains {actual}"
            )


# ---------------------------------------------------------------------------
# Prose-count cross-check (rule 8).
# ---------------------------------------------------------------------------

def _expected_for(totals: dict[str, object], key: str) -> int | None:
    """Resolve a prose-pattern key against the ledger totals.

    `key` is either a top-level totals key ("physical_rows",
    "verdict_rows") or `"kind:<k>"` / `"verdict:<v>"` for a subtotal.
    Returns None if the totals don't define that key.
    """
    if ":" in key:
        top, sub = key.split(":", 1)
        bucket = totals.get({"kind": "by_kind", "verdict": "by_verdict"}[top])
        if isinstance(bucket, dict):
            val = bucket.get(sub)
            if isinstance(val, int):
                return val
        return None
    val = totals.get(key)
    return val if isinstance(val, int) else None


def check_prose_counts(
    md_paths: Iterable[Path],
    totals: dict[str, object],
    repo_root: Path,
    report: Report,
) -> None:
    """For every recognized prose count claim in each Markdown file
    (excluding the ledger itself, checked separately by `check_ledger`),
    assert the leading integer matches the ledger-derived expected total.
    Unrecognized phrasings are ignored — this is a check on the claims
    we *do* recognize, not an exhaustive grep for undeclared claims.
    """
    for path in md_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report.error(
                f"{path.relative_to(repo_root)}: cannot read for "
                f"prose-count cross-check: {exc}"
            )
            continue
        rel = path.relative_to(repo_root)
        for label, pattern, key in PROSE_COUNT_PATTERNS:
            expected = _expected_for(totals, key)
            if expected is None:
                # No ledger baseline for this key — nothing to check
                # against. (Shouldn't happen with the current key set,
                # but be defensive if someone widens PROSE_COUNT_PATTERNS
                # without extending the ledger totals.)
                continue
            for m in pattern.finditer(text):
                declared = int(m.group(1))
                report.checks_run += 1
                if declared != expected:
                    # Compute a line number for the match so the operator
                    # can jump straight to the mismatch.
                    line_no = text.count("\n", 0, m.start()) + 1
                    report.error(
                        f"{rel}:{line_no}: prose claim "
                        f"{declared} {label!r} contradicts "
                        f"ledger-derived total {expected}"
                    )


# ---------------------------------------------------------------------------
# torch-spyre short-SHA drift check (rule 9).
# ---------------------------------------------------------------------------

def check_torch_spyre_short_sha(
    md_paths: Iterable[Path], repo_root: Path, report: Report
) -> None:
    """Flag any `torch-spyre@<short>:...` shorthand whose short SHA is
    not the pinned `TORCH_SPYRE_SHORT_SHA`. Full 40-hex citations are
    ignored here — rule 5 covers them via the manifest cross-check.
    """
    for path in md_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report.error(
                f"{path.relative_to(repo_root)}: cannot read for "
                f"short-SHA check: {exc}"
            )
            continue
        rel = path.relative_to(repo_root)
        for m in TORCH_SPYRE_SHORT_RE.finditer(text):
            sha = m.group("sha")
            report.checks_run += 1
            if sha != TORCH_SPYRE_SHORT_SHA:
                line_no = text.count("\n", 0, m.start()) + 1
                report.error(
                    f"{rel}:{line_no}: `torch-spyre@{sha}:...` short-SHA "
                    f"citation does not match pinned "
                    f"`{TORCH_SPYRE_SHORT_SHA}`"
                )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def find_findings(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    findings_root = repo_root / "findings"
    if not findings_root.is_dir():
        return out
    for path in sorted(findings_root.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        out.append(path)
    return out


def all_markdown_for_link_check(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for name in ("README.md",):
        p = repo_root / name
        if p.is_file():
            out.append(p)
    for sub in ("findings", "reports"):
        d = repo_root / sub
        if d.is_dir():
            out.extend(sorted(d.rglob("*.md")))
    return out


def all_markdown_under_repo(
    repo_root: Path, exclude: Iterable[Path] = ()
) -> list[Path]:
    """Every .md file under `repo_root` except those under `.git/` and
    any explicit `exclude` paths (matched by resolved absolute path).
    Used by rules 8 and 9, which scan the whole audit repo for drift.
    """
    excluded = {p.resolve() for p in exclude}
    out: list[Path] = []
    for path in sorted(repo_root.rglob("*.md")):
        # Skip anything under a .git directory anywhere in the tree.
        try:
            parts = path.relative_to(repo_root).parts
        except ValueError:
            continue
        if any(part == ".git" for part in parts):
            continue
        if path.resolve() in excluded:
            continue
        out.append(path)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: parent of this script's directory)",
    )
    args = ap.parse_args(argv)

    repo_root: Path = args.repo.resolve()
    if not repo_root.is_dir():
        print(
            f"validate_metadata: repo root not found: {repo_root}",
            file=sys.stderr,
        )
        return 1

    report = Report()

    # 1. Front-matter on every finding.
    findings = find_findings(repo_root)
    for f in findings:
        check_finding_front_matter(f, repo_root, report)

    # 2. Ledger structure — returns canonical totals for rule 8.
    ledger = repo_root / "findings" / "upstream-fragility" / "01-patches-ledger.md"
    ledger_totals: dict[str, object] | None = None
    if ledger.is_file():
        ledger_totals = check_ledger(ledger, repo_root, report)

    # 3. SHA cross-check across findings and README.
    pinned, sha_errs = pinned_shas_from_reports(repo_root / "reports")
    for err in sha_errs:
        report.error(err)
    md_for_shas = list(findings)
    top_readme = repo_root / "README.md"
    if top_readme.is_file():
        md_for_shas.append(top_readme)
    check_shas(md_for_shas, pinned, repo_root, report)

    # 4. Internal links resolve.
    for md in all_markdown_for_link_check(repo_root):
        check_internal_links(md, repo_root, report)

    # 5. README category-count cross-check.
    check_readme_counts(top_readme, repo_root, report)

    # 6. Prose count claims cross-checked against the ledger totals
    #    (rule 8). Scan every .md under the repo except the ledger
    #    itself (which is already checked in step 2) and .git/.
    all_md = all_markdown_under_repo(repo_root, exclude=[ledger])
    if ledger_totals is not None:
        check_prose_counts(all_md, ledger_totals, repo_root, report)

    # 7. Short-SHA drift: `torch-spyre@<short>:...` must use the pinned
    #    short SHA (rule 9). Scan every .md, including the ledger.
    check_torch_spyre_short_sha(
        all_markdown_under_repo(repo_root), repo_root, report
    )

    for line in report.errors:
        print(f"ERROR: {line}", file=sys.stderr)
    print(report.summary(), file=sys.stderr)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
