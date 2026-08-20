"""Summarize a get_read_writes / op_read_writes instrumentation JSONL.

Reads the JSONL emitted by `instrument_read_writes.py` and prints:

- total call count (raw vs memoized)
- unique caller-file:line sites
- cache hit rate on `op_read_writes_memo` records
- top-N call sites by count and by total time
- top-N ops by call count (which ops keep getting re-analyzed)
- histogram of per-call `elapsed_us`

Usage:
    python analyze.py <log.jsonl>
"""

from __future__ import annotations

import collections
import json
import statistics
import sys


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


def main(path: str) -> int:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "instrument_installed":
                print(f"  (installed at pid={r['pid']}, argv={r['argv']})")
                continue
            records.append(r)

    if not records:
        print("no records to analyze")
        return 0

    by_kind = collections.Counter(r["kind"] for r in records)
    print("\n=== totals ===")
    for k, v in by_kind.most_common():
        print(f"  {k}: {v}")

    print("\n=== per-call time (us) — all kinds combined ===")
    times = [r["elapsed_us"] for r in records]
    print(f"  count: {len(times)}")
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
    hist = collections.Counter(_bucket(r["elapsed_us"]) for r in records)
    order = ["<100us", "100-500us", "500us-1ms", "1-5ms", "5-20ms", ">=20ms"]
    for b in order:
        print(f"  {b:>10}: {hist.get(b, 0)}")

    print("\n=== cache stats (op_read_writes_memo) ===")
    memo = [r for r in records if r["kind"] == "op_read_writes_memo"]
    if memo:
        hits = sum(1 for r in memo if r["cache"] == "hit")
        misses = sum(1 for r in memo if r["cache"] == "miss")
        print(f"  hits:   {hits}")
        print(f"  misses: {misses}")
        pct = 100.0 * hits / len(memo) if memo else 0
        print(f"  hit rate: {pct:.1f}%")
        hit_times = [r["elapsed_us"] for r in memo if r["cache"] == "hit"]
        miss_times = [r["elapsed_us"] for r in memo if r["cache"] == "miss"]
        if hit_times:
            print(
                f"  hit  p50/mean: {statistics.median(hit_times):.1f}/{statistics.mean(hit_times):.1f} us"
            )
        if miss_times:
            print(
                f"  miss p50/mean: {statistics.median(miss_times):.1f}/{statistics.mean(miss_times):.1f} us"
            )

    print("\n=== top 15 call sites by count ===")
    site_counts = collections.Counter(
        (r["caller_file"].rsplit("/", 1)[-1], r["caller_line"], r["caller_func"])
        for r in records
    )
    for (fn, ln, func), n in site_counts.most_common(15):
        print(f"  {n:>6}  {fn}:{ln}  in {func}()")

    print("\n=== top 15 call sites by total elapsed time (ms) ===")
    site_time = collections.defaultdict(float)
    for r in records:
        k = (r["caller_file"].rsplit("/", 1)[-1], r["caller_line"], r["caller_func"])
        site_time[k] += r["elapsed_us"] / 1000.0
    for (fn, ln, func), ms in sorted(site_time.items(), key=lambda kv: -kv[1])[:15]:
        n = site_counts[(fn, ln, func)]
        print(f"  {ms:>8.1f} ms  ({n:>5} calls, avg {ms*1000/n:>7.1f} us)  {fn}:{ln}  in {func}()")

    print("\n=== top 15 ops by re-analysis count (raw calls only) ===")
    op_counts = collections.Counter(
        r["op_name"]
        for r in records
        if r["kind"] == "get_read_writes_raw"
    )
    for op_name, n in op_counts.most_common(15):
        print(f"  {n:>6}  {op_name}")

    print("\n=== summary line for the finding ===")
    raw = by_kind.get("get_read_writes_raw", 0)
    memo_n = by_kind.get("op_read_writes_memo", 0)
    memo_hits = sum(1 for r in records if r["kind"] == "op_read_writes_memo" and r["cache"] == "hit")
    total_ms = sum(r["elapsed_us"] for r in records) / 1000.0
    print(
        f"  raw={raw} memo={memo_n} (hits={memo_hits}) total_time={total_ms:.1f}ms unique_sites={len(site_counts)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
