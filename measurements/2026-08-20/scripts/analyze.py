"""Summarize a get_read_writes / op_read_writes instrumentation JSONL.

Reads the JSONL emitted by `instrument_read_writes.py` and prints:

- total call count per kind
- inclusive and exclusive time totals (see below)
- unique caller-file:line sites
- cache hit rate on op_read_writes wrapper records
- top-N call sites by count and by exclusive time
- top-N ops by call count (which ops keep getting re-analyzed)
- histogram of per-call `elapsed_us`

Exclusive vs. inclusive time
----------------------------
The op_read_writes wrapper calls into the ComputedBuffer.get_read_writes
wrapper on a cache miss, so summing every record's elapsed_us
double-counts the inner get_read_writes work. The v2 schema tags each
record with a ``kind``:

- raw_get_read_writes
- memo_wrapper_hit
- memo_wrapper_miss_inclusive  (full wall time, including nested raw)
- memo_wrapper_overhead        (miss_inclusive - the raw call it triggered)

Exclusive total =
    sum(raw_get_read_writes) + sum(memo_wrapper_hit) + sum(memo_wrapper_overhead)

Inclusive total (matches the pre-fix analyzer, but tags the double count) =
    sum(raw_get_read_writes) + sum(memo_wrapper_hit) + sum(memo_wrapper_miss_inclusive)

Legacy schema (kinds ``get_read_writes_raw`` / ``op_read_writes_memo``)
is auto-detected: totals are labeled inclusive-only, and any per-kind
breakdown maps them to the closest v2 kind for display.

Usage:
    python analyze.py <log.jsonl>
"""

from __future__ import annotations

import collections
import json
import statistics
import sys

# v2 kinds
KIND_RAW = "raw_get_read_writes"
KIND_HIT = "memo_wrapper_hit"
KIND_MISS_INC = "memo_wrapper_miss_inclusive"
KIND_MISS_OVH = "memo_wrapper_overhead"

# legacy (pre-fix) kinds
LEGACY_RAW = "get_read_writes_raw"
LEGACY_MEMO = "op_read_writes_memo"

REQUIRED_FIELDS = ("op_pyid", "operation_name", "op_type", "seq", "depth")


def _bucket(us: float) -> str:
    if us < 100:
        return "<100us"
    if us < 500:
        return "100-500us"
    if us < 1000:
        return "500us-1ms"
    if us < 5000:
        return "1-5ms"
    if us < 20000:
        return "5-20ms"
    return ">=20ms"


def _has_v2_fields(r: dict) -> bool:
    return all(f in r for f in REQUIRED_FIELDS)


def _op_key(r: dict) -> tuple[str, str]:
    """Operation identity: (op_pyid, operation_name).

    Legacy records lack op_pyid; fall back to ("", op_name) so they
    still bucket sensibly under name alone.
    """
    return (r.get("op_pyid", ""), r.get("operation_name") or r.get("op_name", "<no-name>"))


def main(path: str) -> int:
    records: list[dict] = []
    installed_lines: list[str] = []
    schema_version = None
    missing_field_records = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "instrument_installed":
                installed_lines.append(
                    f"  (installed at pid={r.get('pid')}, argv={r.get('argv')}, "
                    f"schema_version={r.get('schema_version', 1)})"
                )
                if schema_version is None:
                    schema_version = r.get("schema_version", 1)
                continue
            records.append(r)
            if not _has_v2_fields(r):
                missing_field_records += 1

    for ln in installed_lines:
        print(ln)

    if not records:
        print("no records to analyze")
        return 0

    # Auto-detect schema if no installed record announced one.
    kinds_present = {r["kind"] for r in records}
    legacy_mode = (
        schema_version == 1
        or LEGACY_RAW in kinds_present
        or LEGACY_MEMO in kinds_present
    )
    if legacy_mode:
        print(
            "\n!!! LEGACY DATA — instrumentation predates exclusive-timing fix; "
            "totals shown are INCLUSIVE (raw work inside op_read_writes misses is "
            "counted twice). Re-run instrumentation to obtain exclusive numbers. !!!"
        )
    if missing_field_records and not legacy_mode:
        print(
            f"\n!!! WARNING: {missing_field_records} records lack v2 identity fields "
            f"({', '.join(REQUIRED_FIELDS)}); treating as partial schema. !!!"
        )

    by_kind = collections.Counter(r["kind"] for r in records)
    print("\n=== totals ===")
    for k, v in by_kind.most_common():
        print(f"  {k}: {v}")

    # ---- inclusive/exclusive time ----
    if legacy_mode:
        raw_recs = [r for r in records if r["kind"] == LEGACY_RAW]
        memo_recs = [r for r in records if r["kind"] == LEGACY_MEMO]
        hit_recs = [r for r in memo_recs if r.get("cache") == "hit"]
        miss_recs = [r for r in memo_recs if r.get("cache") == "miss"]

        raw_us = sum(r["elapsed_us"] for r in raw_recs)
        hit_us = sum(r["elapsed_us"] for r in hit_recs)
        miss_us = sum(r["elapsed_us"] for r in miss_recs)
        inclusive_total_ms = (raw_us + hit_us + miss_us) / 1000.0

        print("\n=== time totals (LEGACY: inclusive only) ===")
        print(f"  {LEGACY_RAW:>32}: {raw_us/1000:>10.1f} ms  ({len(raw_recs)} calls)")
        print(f"  {LEGACY_MEMO + ' (hit)':>32}: {hit_us/1000:>10.1f} ms  ({len(hit_recs)} calls)")
        print(f"  {LEGACY_MEMO + ' (miss)':>32}: {miss_us/1000:>10.1f} ms  ({len(miss_recs)} calls)")
        print(f"  {'inclusive TOTAL':>32}: {inclusive_total_ms:>10.1f} ms")
        print(
            "  (exclusive total not available: legacy schema has no per-miss "
            "inner-raw attribution.)"
        )
    else:
        raw_recs = [r for r in records if r["kind"] == KIND_RAW]
        hit_recs = [r for r in records if r["kind"] == KIND_HIT]
        miss_inc_recs = [r for r in records if r["kind"] == KIND_MISS_INC]
        ovh_recs = [r for r in records if r["kind"] == KIND_MISS_OVH]

        raw_us = sum(r["elapsed_us"] for r in raw_recs)
        hit_us = sum(r["elapsed_us"] for r in hit_recs)
        miss_inc_us = sum(r["elapsed_us"] for r in miss_inc_recs)
        ovh_us = sum(r["elapsed_us"] for r in ovh_recs)

        exclusive_total_ms = (raw_us + hit_us + ovh_us) / 1000.0
        inclusive_total_ms = (raw_us + hit_us + miss_inc_us) / 1000.0

        print("\n=== time totals ===")
        print(f"  {KIND_RAW:>32}: {raw_us/1000:>10.1f} ms  ({len(raw_recs)} calls)")
        print(f"  {KIND_HIT:>32}: {hit_us/1000:>10.1f} ms  ({len(hit_recs)} calls)")
        print(f"  {KIND_MISS_INC:>32}: {miss_inc_us/1000:>10.1f} ms  ({len(miss_inc_recs)} calls, inclusive)")
        print(f"  {KIND_MISS_OVH:>32}: {ovh_us/1000:>10.1f} ms  ({len(ovh_recs)} calls, exclusive)")
        print(f"  {'EXCLUSIVE TOTAL':>32}: {exclusive_total_ms:>10.1f} ms   "
              f"(raw + hit + overhead — no double-count)")
        print(f"  {'inclusive total':>32}: {inclusive_total_ms:>10.1f} ms   "
              f"(raw + hit + miss_inclusive — legacy comparable)")

    # ---- per-call time distribution (exclusive kinds only when available) ----
    if legacy_mode:
        timing_recs = records
        header = "=== per-call time (us) — all records, INCLUSIVE ==="
    else:
        # Exclude miss_inclusive to avoid double-counting the raw call.
        timing_recs = [
            r for r in records if r["kind"] != KIND_MISS_INC
        ]
        header = "=== per-call time (us) — exclusive kinds (raw + hit + overhead) ==="

    times = [r["elapsed_us"] for r in timing_recs]
    print(f"\n{header}")
    print(f"  count: {len(times)}")
    if times:
        print(f"  min:   {min(times):.1f}")
        print(f"  p50:   {statistics.median(times):.1f}")
        if len(times) >= 20:
            srt = sorted(times)
            print(f"  p90:   {srt[int(len(times)*0.9)]:.1f}")
            print(f"  p99:   {srt[int(len(times)*0.99)]:.1f}")
        print(f"  max:   {max(times):.1f}")
        print(f"  mean:  {statistics.mean(times):.1f}")
        print(f"  total: {sum(times)/1000:.1f} ms")

    print("\n=== per-call time bucket histogram ===")
    hist = collections.Counter(_bucket(r["elapsed_us"]) for r in timing_recs)
    order = ["<100us", "100-500us", "500us-1ms", "1-5ms", "5-20ms", ">=20ms"]
    for b in order:
        print(f"  {b:>10}: {hist.get(b, 0)}")

    # ---- cache stats ----
    print("\n=== cache stats (op_read_writes wrapper) ===")
    if legacy_mode:
        memo = [r for r in records if r["kind"] == LEGACY_MEMO]
        if memo:
            hits = sum(1 for r in memo if r.get("cache") == "hit")
            misses = sum(1 for r in memo if r.get("cache") == "miss")
            print(f"  hits:   {hits}")
            print(f"  misses: {misses}")
            pct = 100.0 * hits / len(memo) if memo else 0
            print(f"  hit rate: {pct:.1f}%")
            hit_times = [r["elapsed_us"] for r in memo if r.get("cache") == "hit"]
            miss_times = [r["elapsed_us"] for r in memo if r.get("cache") == "miss"]
            if hit_times:
                print(
                    f"  hit  p50/mean: {statistics.median(hit_times):.1f}/"
                    f"{statistics.mean(hit_times):.1f} us"
                )
            if miss_times:
                print(
                    f"  miss p50/mean (INCLUSIVE): {statistics.median(miss_times):.1f}/"
                    f"{statistics.mean(miss_times):.1f} us"
                )
    else:
        hits_n = len(hit_recs)
        misses_n = len(miss_inc_recs)
        total_memo = hits_n + misses_n
        print(f"  hits:   {hits_n}")
        print(f"  misses: {misses_n}")
        pct = 100.0 * hits_n / total_memo if total_memo else 0
        print(f"  hit rate: {pct:.1f}%")
        hit_times = [r["elapsed_us"] for r in hit_recs]
        miss_inc_times = [r["elapsed_us"] for r in miss_inc_recs]
        ovh_times = [r["elapsed_us"] for r in ovh_recs]
        if hit_times:
            print(
                f"  hit p50/mean: {statistics.median(hit_times):.1f}/"
                f"{statistics.mean(hit_times):.1f} us"
            )
        if miss_inc_times:
            print(
                f"  miss inclusive p50/mean: {statistics.median(miss_inc_times):.1f}/"
                f"{statistics.mean(miss_inc_times):.1f} us"
            )
        if ovh_times:
            print(
                f"  miss overhead (exclusive) p50/mean: {statistics.median(ovh_times):.1f}/"
                f"{statistics.mean(ovh_times):.1f} us"
            )

    # ---- call-site attribution: use exclusive-eligible records only ----
    print("\n=== top 15 call sites by count ===")
    site_counts: collections.Counter = collections.Counter(
        (r["caller_file"].rsplit("/", 1)[-1], r["caller_line"], r["caller_func"])
        for r in timing_recs
    )
    for (fn, ln, func), n in site_counts.most_common(15):
        print(f"  {n:>6}  {fn}:{ln}  in {func}()")

    label = "inclusive" if legacy_mode else "exclusive"
    print(f"\n=== top 15 call sites by total {label} elapsed time (ms) ===")
    site_time: dict = collections.defaultdict(float)
    for r in timing_recs:
        k = (r["caller_file"].rsplit("/", 1)[-1], r["caller_line"], r["caller_func"])
        site_time[k] += r["elapsed_us"] / 1000.0
    for (fn, ln, func), ms in sorted(site_time.items(), key=lambda kv: -kv[1])[:15]:
        n = site_counts[(fn, ln, func)]
        avg_us = ms * 1000 / n if n else 0.0
        print(f"  {ms:>8.1f} ms  ({n:>5} calls, avg {avg_us:>7.1f} us)  {fn}:{ln}  in {func}()")

    # ---- per-op reanalysis: identity = (op_pyid, operation_name) ----
    print("\n=== top 15 ops by re-analysis count (raw calls only) ===")
    if legacy_mode:
        raw_kind = LEGACY_RAW
    else:
        raw_kind = KIND_RAW
    op_counts: collections.Counter = collections.Counter(
        _op_key(r) for r in records if r["kind"] == raw_kind
    )
    for (pyid, opname), n in op_counts.most_common(15):
        pyid_disp = pyid if pyid else "<legacy>"
        print(f"  {n:>6}  {opname}   (pyid={pyid_disp})")

    # ---- summary line for the finding ----
    print("\n=== summary line for the finding ===")
    if legacy_mode:
        raw_n = by_kind.get(LEGACY_RAW, 0)
        memo_n = by_kind.get(LEGACY_MEMO, 0)
        memo_hits = sum(
            1 for r in records if r["kind"] == LEGACY_MEMO and r.get("cache") == "hit"
        )
        print(
            f"  [LEGACY/inclusive] raw={raw_n} memo={memo_n} (hits={memo_hits}) "
            f"inclusive_total={inclusive_total_ms:.1f}ms unique_sites={len(site_counts)}"
        )
    else:
        raw_n = by_kind.get(KIND_RAW, 0)
        hits_n = by_kind.get(KIND_HIT, 0)
        misses_n = by_kind.get(KIND_MISS_INC, 0)
        print(
            f"  raw={raw_n} memo_hits={hits_n} memo_misses={misses_n} "
            f"exclusive_total={exclusive_total_ms:.1f}ms "
            f"inclusive_total={inclusive_total_ms:.1f}ms "
            f"unique_sites={len(site_counts)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
