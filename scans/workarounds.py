#!/usr/bin/env python3
# Copyright 2026 Todd Deshane. Apache-2.0.
"""Scanner: find temporary comments, TODO markers, PyTorch version refs,
and other "workaround" language in the pinned Torch-Spyre source tree.

Design notes:
- Stdlib only (re, tokenize, ast, pathlib, json, argparse) so we can run
  without the torch-spyre venv.
- Two passes per file:
    1. tokenize pass to enumerate comments and docstring-adjacent strings,
       giving accurate (line, kind) info.
    2. plain line pass for non-Python files (.md, .rst, .yaml, .toml, .cfg)
       and as a safety net for Python source lines that aren't captured as
       COMMENT / STRING tokens (rare but happens with f-string embeds).
- We categorize each hit and, when a PyTorch version like "PT 2.9" or
  "PyTorch 2.11" is mentioned, we compare it to the pinned supported
  version (parsed from requirements/*.txt) and flag whether the pinned
  version is later.

Usage:
    python workarounds.py \
        --root /tmp/ts-pinned-scan/fea0c4b \
        --out /Users/tdeshane/toddllm/torch-spyre-notes/scans/results/workarounds.json \
        [--top N]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tokenize
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

TORCH_SPYRE_SHA = "fea0c4be901e1383b1f700dbad8887128b0fcb27"
SCANNER_NAME = "workarounds"

# --------------------------------------------------------------------------- #
# Pattern definitions
# --------------------------------------------------------------------------- #
# All patterns are case-insensitive. Order matters only for categorization
# (first match wins), so put the more specific patterns first.

PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # (category, label, regex)
    ("todo",         "TODO",             re.compile(r"\bTODO\b", re.IGNORECASE)),
    ("fixme",        "FIXME",            re.compile(r"\bFIXME\b", re.IGNORECASE)),
    ("fixme",        "XXX",              re.compile(r"\bXXX\b", re.IGNORECASE)),
    ("fixme",        "HACK",             re.compile(r"\bHACK\b", re.IGNORECASE)),
    ("fixme",        "SHOULDREMOVE",     re.compile(r"\bSHOULDREMOVE\b", re.IGNORECASE)),

    ("assumption",   "for now",          re.compile(r"\bfor\s+now\b", re.IGNORECASE)),
    ("assumption",   "temporary",        re.compile(r"\btemporary\b", re.IGNORECASE)),
    ("assumption",   "workaround",       re.compile(r"\bworkaround\b", re.IGNORECASE)),
    ("assumption",   "for the moment",   re.compile(r"\bfor\s+the\s+moment\b", re.IGNORECASE)),
    ("assumption",   "we believe",       re.compile(r"\bwe\s+believe\b", re.IGNORECASE)),
    ("assumption",   "should be safe",   re.compile(r"\bshould\s+be\s+safe\b", re.IGNORECASE)),
    ("assumption",   "assume",           re.compile(r"\bassume(?:d|s)?\b", re.IGNORECASE)),
    ("assumption",   "hopefully",        re.compile(r"\bhopefully\b", re.IGNORECASE)),

    ("upstream",     "once PyTorch",     re.compile(r"\bonce\s+PyTorch\b", re.IGNORECASE)),
    ("upstream",     "upstream fix",     re.compile(r"\bupstream\s+fix\b", re.IGNORECASE)),
    ("upstream",     "when upstream",    re.compile(r"\bwhen\s+upstream\b", re.IGNORECASE)),
    ("upstream",     "waiting on upstream", re.compile(r"\bwaiting\s+on\s+upstream\b", re.IGNORECASE)),

    ("version-ref",  "PT 2.x",           re.compile(r"\bPT\s*2\.(?:9|10|11|12|13)\b", re.IGNORECASE)),
    ("version-ref",  "PyTorch 2.x",      re.compile(r"\bPyTorch\s+2\.(?:9|10|11|12|13)\b", re.IGNORECASE)),

    ("deprecated",   "deprecated",       re.compile(r"\bdeprecated\b", re.IGNORECASE)),
    ("deprecated",   "legacy",           re.compile(r"\blegacy\b", re.IGNORECASE)),
]

# Regex to pull the specific version out of a version-ref hit, so we can
# compare it against the pinned supported version.
VERSION_EXTRACT = re.compile(
    r"\b(?:PT|PyTorch)\s*(2)\.(9|10|11|12|13)\b",
    re.IGNORECASE,
)

# File extensions we treat as text-scannable.
TEXT_EXTS = {
    ".py", ".pyi", ".md", ".rst", ".txt",
    ".toml", ".cfg", ".ini",
    ".yaml", ".yml", ".json",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cu",
    ".sh",
}

# Directories to skip.
SKIP_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "venv", "node_modules",
    "build", "dist", ".tox", ".eggs",
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    file: str
    line: int
    kind: str            # category label
    label: str           # which pattern matched
    snippet: str         # 3-line context snippet
    context: str         # matched line, stripped
    version_referenced: str | None = None
    pinned_version_is_later: bool | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop nulls for clean JSON.
        if d["version_referenced"] is None:
            d.pop("version_referenced")
            d.pop("pinned_version_is_later")
        return d


@dataclass
class Report:
    scanner: str
    torch_spyre_sha: str
    scanned_files: int
    total_hits: int
    pinned_torch_version: str | None
    hits: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scanner": self.scanner,
            "torch_spyre_sha": self.torch_spyre_sha,
            "scanned_files": self.scanned_files,
            "total_hits": self.total_hits,
            "pinned_torch_version": self.pinned_torch_version,
            "hits": [h.to_dict() for h in self.hits],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_version_tuple(v: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", v)
    return tuple(int(p) for p in parts) if parts else (0,)


def find_pinned_torch_version(root: Path) -> str | None:
    """Pull the pinned torch version from requirements/*.txt.

    We prefer run.txt but fall back to build.txt / dev.txt.
    """
    candidates = [
        root / "requirements" / "run.txt",
        root / "requirements" / "build.txt",
        root / "requirements" / "dev.txt",
    ]
    pat = re.compile(r"^\s*torch\s*==\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", re.MULTILINE)
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        yield path


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def build_snippet(lines: list[str], lineno: int) -> str:
    """3-line context snippet centered on lineno (1-indexed)."""
    lo = max(1, lineno - 1)
    hi = min(len(lines), lineno + 1)
    out = []
    for i in range(lo, hi + 1):
        marker = ">>" if i == lineno else "  "
        out.append(f"{marker} {i:>6}: {lines[i - 1]}")
    return "\n".join(out)


def categorize(text: str) -> list[tuple[str, str]]:
    """Return list of (category, label) for every pattern that matches."""
    matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, label, regex in PATTERNS:
        if regex.search(text):
            key = (category, label)
            if key not in seen:
                seen.add(key)
                matches.append(key)
    return matches


def version_annotation(
    text: str, pinned: str | None
) -> tuple[str | None, bool | None]:
    m = VERSION_EXTRACT.search(text)
    if not m:
        return None, None
    ref = f"{m.group(1)}.{m.group(2)}"
    if not pinned:
        return ref, None
    pinned_tuple = parse_version_tuple(pinned)
    ref_tuple = parse_version_tuple(ref)
    return ref, pinned_tuple > ref_tuple


# --------------------------------------------------------------------------- #
# Python-aware pass: tokens (comments + string literals)
# --------------------------------------------------------------------------- #
def scan_python_tokens(source: str) -> Iterator[tuple[int, str, str]]:
    """Yield (lineno, kind, text) for COMMENT and STRING tokens.

    kind is either "comment" or "string".
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], "comment", tok.string
            elif tok.type == tokenize.STRING:
                # Multi-line docstrings: emit one entry per line so the
                # matched line number lands on the actual sentence.
                start_line = tok.start[0]
                lines = tok.string.splitlines() or [tok.string]
                for offset, chunk in enumerate(lines):
                    yield start_line + offset, "string", chunk
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        # Fall back to a plain line scan; the caller does that anyway
        # when we return nothing.
        return


# --------------------------------------------------------------------------- #
# Main scan
# --------------------------------------------------------------------------- #
def scan_file(path: Path, root: Path, pinned: str | None) -> list[Hit]:
    lines = read_lines(path)
    if not lines:
        return []
    rel = str(path.relative_to(root))
    hits: list[Hit] = []
    seen: set[tuple[int, str, str]] = set()  # dedupe (line, category, label)

    def record(line_no: int, text: str) -> None:
        line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else text
        for category, label in categorize(text):
            key = (line_no, category, label)
            if key in seen:
                continue
            seen.add(key)
            ver, later = (None, None)
            if category == "version-ref":
                ver, later = version_annotation(text, pinned)
            hits.append(
                Hit(
                    file=rel,
                    line=line_no,
                    kind=category,
                    label=label,
                    snippet=build_snippet(lines, line_no),
                    context=line_text.strip(),
                    version_referenced=ver,
                    pinned_version_is_later=later,
                )
            )

    # Python-aware pass for .py / .pyi files.
    if path.suffix.lower() in {".py", ".pyi"}:
        try:
            source = path.read_text(errors="replace")
            for lineno, _kind, text in scan_python_tokens(source):
                record(lineno, text)
        except OSError:
            pass

    # Plain line pass — catches non-Python files and any patterns that
    # sit inside code (identifiers, log strings, f-string parts) which
    # the tokenize pass would have handed us anyway; the seen-set dedupes.
    for i, raw in enumerate(lines, start=1):
        record(i, raw)

    return hits


def scan_tree(root: Path, pinned: str | None) -> tuple[int, list[Hit]]:
    scanned = 0
    all_hits: list[Hit] = []
    for path in iter_source_files(root):
        scanned += 1
        all_hits.extend(scan_file(path, root, pinned))
    return scanned, all_hits


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def sort_key(h: Hit) -> tuple:
    # Priority: version-ref/upstream first, then todo/fixme, then rest.
    order = {
        "version-ref": 0,
        "upstream":    1,
        "fixme":       2,
        "todo":        3,
        "assumption":  4,
        "deprecated":  5,
    }
    return (order.get(h.kind, 9), h.file, h.line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Torch-Spyre source root")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--top", type=int, default=20, help="Top-N to print to stdout")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        print(f"ERROR: root {root} is not a directory", file=sys.stderr)
        return 2

    pinned = find_pinned_torch_version(root)
    scanned, hits = scan_tree(root, pinned)
    hits.sort(key=sort_key)

    report = Report(
        scanner=SCANNER_NAME,
        torch_spyre_sha=TORCH_SPYRE_SHA,
        scanned_files=scanned,
        total_hits=len(hits),
        pinned_torch_version=pinned,
        hits=hits,
    )
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    # Human-readable summary.
    print(f"scanner:            {SCANNER_NAME}")
    print(f"torch-spyre sha:    {TORCH_SPYRE_SHA}")
    print(f"scanned files:      {scanned}")
    print(f"pinned torch:       {pinned}")
    print(f"total hits:         {len(hits)}")

    by_cat: dict[str, int] = {}
    for h in hits:
        by_cat[h.kind] = by_cat.get(h.kind, 0) + 1
    if by_cat:
        print("hits by category:")
        for cat in sorted(by_cat):
            print(f"  {cat:12s} {by_cat[cat]:>5}")

    print()
    print(f"top {min(args.top, len(hits))} hits:")
    for h in hits[: args.top]:
        extra = ""
        if h.version_referenced is not None:
            later = "pinned>ref" if h.pinned_version_is_later else "pinned<=ref"
            extra = f"  [ver={h.version_referenced} {later}]"
        print(f"  {h.file}:{h.line}  [{h.kind}/{h.label}]{extra}")
        print(f"      {h.context[:160]}")

    print()
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
