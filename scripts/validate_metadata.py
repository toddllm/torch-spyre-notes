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

def check_ledger(ledger_path: Path, repo_root: Path, report: Report) -> None:
    text = ledger_path.read_text(encoding="utf-8", errors="replace")
    rows, parse_errors = parse_ledger_rows(text)
    rel = ledger_path.relative_to(repo_root)
    for err in parse_errors:
        report.error(f"{rel}: {err}")
    if not rows:
        return
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

    # 2. Ledger structure.
    ledger = repo_root / "findings" / "upstream-fragility" / "01-patches-ledger.md"
    if ledger.is_file():
        check_ledger(ledger, repo_root, report)

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

    for line in report.errors:
        print(f"ERROR: {line}", file=sys.stderr)
    print(report.summary(), file=sys.stderr)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
