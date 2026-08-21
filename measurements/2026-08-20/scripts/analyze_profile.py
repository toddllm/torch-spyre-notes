"""Analyze a whole-compile profile bundle.

Primary mode (cProfile):
    python analyze_profile.py <profile.prof> [<phase_timings.json>]

Compat mode (instrumentation JSONL — no cProfile output):
    python analyze_profile.py <log.jsonl>

The script detects the input format by content:

- ``.prof`` / marshalled cProfile Stats -> Top-30 by exclusive (tottime)
  stacks, with cumulative (cumtime) alongside; cross-references each
  hot function against the ``repeated_analysis`` scan so a reader can
  see at a glance which hot functions are ALSO called many times in
  many places.
- Instrumentation JSONL (records with a ``kind`` field like
  ``raw_get_read_writes`` / ``get_read_writes_raw`` / ``memo_wrapper_hit``)
  -> compat-mode report: top-30 by count and by summed inclusive_us.
  This proves the analyzer is well-formed against real data even when
  no cProfile output is available yet.

If ``phase_timings.json`` is provided (from ``profile_whole_compile.py``)
we also print a Time-Per-Phase table with per-phase call count and total
elapsed time.

The scan cross-reference file defaults to
``scans/results/repeated_analysis.json`` under the audit repo root (auto-
detected via the script's on-disk location). Override with
``--repeated-scan <path>``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pstats
import statistics
import sys
from typing import Any


AUDIT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DEFAULT_REPEATED_SCAN = os.path.join(
    AUDIT_ROOT, "scans", "results", "repeated_analysis.json"
)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _looks_like_jsonl(path: str) -> bool:
    """Return True iff the first non-blank line parses as JSON with a 'kind'."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return False
    if b"\n" not in head and len(head) == 4096:
        return False
    # Marshalled pstats starts with a marshal-code byte (typically 0x28 '(' or 0x80),
    # which will not decode as UTF-8 JSON.
    text = head.split(b"\n", 1)[0].decode("utf-8", errors="ignore").strip()
    if not text.startswith("{"):
        return False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    # We accept ANY JSONL as compat-mode input; a top-level 'kind' is a
    # strong hint but not required (a stray JSON file would also flow
    # through here and produce a per-record report).
    return isinstance(obj, dict)


def _looks_like_pstats(path: str) -> bool:
    try:
        pstats.Stats(path)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Scan cross-reference
# ---------------------------------------------------------------------------


def load_repeated_index(scan_path: str) -> dict[str, list[dict[str, Any]]]:
    """Return {basename: [hit, ...]} indexed by file basename.

    Basename indexing lets us match cProfile file entries (which carry
    absolute paths) against scan hits recorded from a different install
    root. We also key by (basename, function name) for tighter matches.
    """
    if not os.path.exists(scan_path):
        return {}
    with open(scan_path) as f:
        data = json.load(f)
    by_base: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for hit in data.get("hits", []):
        base = os.path.basename(hit.get("file", ""))
        if base:
            by_base[base].append(hit)
    return by_base


def flag_hot_and_repeated(
    file_path: str,
    func_name: str,
    repeated_by_base: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return matching scan hits whose (basename, enclosing_func) matches."""
    base = os.path.basename(file_path)
    hits = repeated_by_base.get(base, [])
    matches = []
    for h in hits:
        if h.get("enclosing_func") == func_name or h.get("kind") == func_name:
            matches.append(h)
    return matches


# ---------------------------------------------------------------------------
# cProfile mode
# ---------------------------------------------------------------------------


def analyze_pstats(
    prof_path: str,
    repeated_by_base: dict[str, list[dict[str, Any]]],
    top_n: int = 30,
) -> None:
    stats = pstats.Stats(prof_path)
    print(f"=== cProfile summary: {prof_path} ===")
    print(f"function entries: {len(stats.stats)}")
    total_tt = sum(v[2] for v in stats.stats.values())
    total_ct = sum(v[3] for v in stats.stats.values())
    print(f"total tottime (exclusive) across all functions: {total_tt:.3f} s")
    print(f"total cumtime (inclusive) across all functions: {total_ct:.3f} s")

    # stats.stats: {(file, lineno, funcname): (ncalls, ncalls_rec, tottime, cumtime, callers_dict)}
    rows = []
    for (file, lineno, funcname), (nc, ncrec, tt, ct, _callers) in stats.stats.items():
        matches = flag_hot_and_repeated(file, funcname, repeated_by_base)
        rows.append(
            {
                "file": file,
                "lineno": lineno,
                "funcname": funcname,
                "ncalls": nc,
                "tottime_s": tt,
                "cumtime_s": ct,
                "per_call_us": (tt / nc * 1e6) if nc else 0.0,
                "repeated_hits": matches,
            }
        )

    print(f"\n=== top {top_n} by exclusive time (tottime) ===")
    rows_by_tt = sorted(rows, key=lambda r: -r["tottime_s"])[:top_n]
    _print_pstats_table(rows_by_tt)

    print(f"\n=== top {top_n} by inclusive time (cumtime) ===")
    rows_by_ct = sorted(rows, key=lambda r: -r["cumtime_s"])[:top_n]
    _print_pstats_table(rows_by_ct)

    hot_and_repeated = [r for r in rows_by_tt if r["repeated_hits"]]
    print(
        f"\n=== HOT AND REPEATEDLY CALLED "
        f"({len(hot_and_repeated)} of top-{top_n} match scans/results/repeated_analysis.json) ==="
    )
    for r in hot_and_repeated:
        rep_kinds = collections.Counter(h["kind"] for h in r["repeated_hits"])
        rep_summary = ", ".join(f"{k}x{v}" for k, v in rep_kinds.most_common())
        print(
            f"  tottime={r['tottime_s']:.3f}s cumtime={r['cumtime_s']:.3f}s "
            f"ncalls={r['ncalls']:>8}  {os.path.basename(r['file'])}:{r['lineno']} "
            f"in {r['funcname']}()  [{rep_summary}]"
        )
    if not hot_and_repeated:
        print("  (no top-30 entry appears in the repeated-analysis scan)")


def _print_pstats_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"  {'tottime(s)':>10}  {'cumtime(s)':>10}  {'ncalls':>8}  {'per-call(us)':>12}  "
        f"function"
    )
    for r in rows:
        loc = f"{os.path.basename(r['file'])}:{r['lineno']}"
        mark = " *" if r["repeated_hits"] else "  "
        print(
            f"  {r['tottime_s']:>10.3f}  {r['cumtime_s']:>10.3f}  "
            f"{r['ncalls']:>8}  {r['per_call_us']:>12.1f}{mark}"
            f"{r['funcname']} @ {loc}"
        )
    print("  (rows marked * appear in repeated_analysis.json)")


# ---------------------------------------------------------------------------
# Phase timings
# ---------------------------------------------------------------------------


def print_phase_table(phase_path: str) -> None:
    with open(phase_path) as f:
        data = json.load(f)
    print(f"\n=== time per phase: {phase_path} ===")
    hooks = data.get("hooks", {})
    phases = data.get("phases", {})
    print(f"  {'phase':>28}  {'attached':>1}  {'calls':>6}  {'total_ms':>10}  {'avg_us':>10}")
    for phase, entries in phases.items():
        attached = "y" if (hooks.get(phase, {}).get("attached")) else "-"
        n = len(entries)
        total_us = sum((e.get("elapsed_us") or 0.0) for e in entries)
        avg_us = (total_us / n) if n else 0.0
        print(
            f"  {phase:>28}  {attached:>1}  {n:>6}  "
            f"{total_us/1000:>10.1f}  {avg_us:>10.1f}"
        )
    unattached = [p for p, h in hooks.items() if not h.get("attached")]
    if unattached:
        print("  (unattached phases: " + ", ".join(unattached) + ")")


# ---------------------------------------------------------------------------
# JSONL compat mode
# ---------------------------------------------------------------------------


def analyze_jsonl(path: str, top_n: int = 30) -> None:
    """Compat-mode report over an instrumentation JSONL log.

    Handles both the v2 schema
    (``raw_get_read_writes`` / ``memo_wrapper_hit`` / ``memo_wrapper_miss_inclusive``
    / ``memo_wrapper_overhead``) and the legacy schema
    (``get_read_writes_raw`` / ``op_read_writes_memo``).
    """
    records: list[dict[str, Any]] = []
    installed = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "instrument_installed":
                installed.append(r)
                continue
            records.append(r)

    print(f"=== compat mode (JSONL): {path} ===")
    print(f"records: {len(records)}  installed markers: {len(installed)}")
    for m in installed:
        print(
            f"  installed pid={m.get('pid')} schema_version={m.get('schema_version', 1)} "
            f"log_path={m.get('log_path')}"
        )
    if not records:
        print("no records to analyze")
        return

    kinds = collections.Counter(r.get("kind") for r in records)
    print("\n=== kinds ===")
    for k, n in kinds.most_common():
        print(f"  {k}: {n}")

    # Site key = (caller_file basename, caller_line, caller_func).
    def _site(r: dict[str, Any]) -> tuple[str, int, str]:
        cf = r.get("caller_file") or ""
        return (os.path.basename(cf), r.get("caller_line") or 0, r.get("caller_func") or "?")

    # Pick the "inclusive" us: prefer inclusive_us on overhead records
    # (which carry it explicitly), else fall back to elapsed_us.
    def _inclusive_us(r: dict[str, Any]) -> float:
        return float(r.get("inclusive_us", r.get("elapsed_us", 0.0)) or 0.0)

    # For compat-mode "top-30 by count" and "top-30 by inclusive_us"
    # we DO NOT deduplicate: the point is to expose the raw hot sites.
    site_counts: collections.Counter = collections.Counter(_site(r) for r in records)
    site_inclusive: dict[tuple[str, int, str], float] = collections.defaultdict(float)
    for r in records:
        site_inclusive[_site(r)] += _inclusive_us(r)

    print(f"\n=== top {top_n} sites by count ===")
    print(f"  {'count':>8}  {'file':<48}  {'line':>6}  function()")
    for site, n in site_counts.most_common(top_n):
        fn, ln, func = site
        print(f"  {n:>8}  {fn:<48}  {ln:>6}  {func}()")

    print(f"\n=== top {top_n} sites by summed inclusive_us ===")
    print(f"  {'inclusive_ms':>14}  {'count':>8}  {'avg_us':>10}  {'file':<40}  {'line':>6}  function()")
    top_by_time = sorted(site_inclusive.items(), key=lambda kv: -kv[1])[:top_n]
    for site, tot_us in top_by_time:
        fn, ln, func = site
        n = site_counts[site]
        avg_us = tot_us / n if n else 0.0
        print(
            f"  {tot_us/1000:>14.1f}  {n:>8}  {avg_us:>10.1f}  "
            f"{fn:<40}  {ln:>6}  {func}()"
        )

    # Per-record elapsed distribution (raw + hit + overhead when v2).
    all_us = [float(r.get("elapsed_us") or 0.0) for r in records]
    print("\n=== per-record elapsed_us distribution (all kinds) ===")
    if all_us:
        srt = sorted(all_us)
        p90 = srt[int(len(srt) * 0.9)] if len(srt) >= 10 else srt[-1]
        p99 = srt[int(len(srt) * 0.99)] if len(srt) >= 100 else srt[-1]
        print(f"  count={len(all_us)}  min={min(all_us):.1f}  p50={statistics.median(all_us):.1f}"
              f"  p90={p90:.1f}  p99={p99:.1f}  max={max(all_us):.1f}"
              f"  mean={statistics.mean(all_us):.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cProfile output and/or an instrumentation JSONL log."
    )
    parser.add_argument("input", help="profile.prof OR instrumentation .jsonl")
    parser.add_argument(
        "phase_timings",
        nargs="?",
        default=None,
        help="Optional phase_timings.json from profile_whole_compile.py",
    )
    parser.add_argument(
        "--repeated-scan",
        default=DEFAULT_REPEATED_SCAN,
        help=f"Path to repeated_analysis.json (default: {DEFAULT_REPEATED_SCAN})",
    )
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    repeated_by_base = load_repeated_index(args.repeated_scan)
    if not repeated_by_base:
        print(
            f"note: no repeated-analysis scan at {args.repeated_scan} (cross-reference disabled)",
            file=sys.stderr,
        )

    # Order matters: check JSONL first (cheap, textual); only try pstats
    # if it clearly isn't JSONL.
    if _looks_like_jsonl(args.input):
        analyze_jsonl(args.input, top_n=args.top_n)
    elif _looks_like_pstats(args.input):
        analyze_pstats(args.input, repeated_by_base, top_n=args.top_n)
    else:
        print(
            f"input {args.input} is neither JSONL nor cProfile output",
            file=sys.stderr,
        )
        return 3

    if args.phase_timings and os.path.exists(args.phase_timings):
        print_phase_table(args.phase_timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
