"""Paired-measurement harness with alternating-order runs.

Purpose
-------
When measuring "is my candidate change faster than the baseline?" the
noise floor from cold caches, background CPU pressure, and OS jitter
dwarfs the effect for microsecond-scale kernel/pass rework. This harness
reduces those confounds by:

- **Alternating order.** Run the baseline and candidate interleaved
  (B, C, C, B, B, C, ...) so any monotonic drift (thermal, page-cache
  warmup, GC state) affects both variants equally on average.
- **Paired differences.** Sort completed trials into pairs by trial
  index, compute per-pair ``candidate - baseline``. Report median, IQR,
  min, and max of these pairwise deltas (percent and absolute).
- **Bootstrap 95% CI for the median paired delta.** Resample pairs
  with replacement (default 10_000 iterations) and report the 2.5th /
  97.5th percentile of the resampled median.

Config
------
A JSON file specifying the two variants::

    {
      "iters": 20,
      "warmup_iters": 2,
      "seed": 12345,
      "baseline": {
        "kind": "python_callable",
        "module": "measurements.2026-08-20.scripts.run_test_flash",
        "callable": "main",
        "args": [],
        "kwargs": {}
      },
      "candidate": {
        "kind": "python_callable",
        "module": "some.module.with_candidate",
        "callable": "main",
        "args": [],
        "kwargs": {}
      }
    }

Supported ``kind`` values:

- ``python_callable`` — import ``module`` and call ``callable(*args, **kwargs)``.
  Measured span: wall time of that call.
- ``subprocess`` — run ``["/usr/bin/env", *cmd]`` as a subprocess.
  Measured span: subprocess wall time from spawn to exit.

Both variants must have the same ``kind`` for a fair comparison.

Output
------
A single JSON blob written to ``--out`` (default: ``paired_result.json``)
containing per-trial timings, the ordering used, and the summary stats.
A short textual summary is printed to stdout.

Determinism note
----------------
``random.Random(seed)`` fixes the alternating order (as a specific
permutation) and the bootstrap indices. Passing the same seed + iters
reproduces the same trial order and CI, which makes review comments
about "your run got lucky" checkable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from typing import Any


def _run_python_callable(spec: dict[str, Any]) -> float:
    module_name = spec["module"]
    callable_name = spec["callable"]
    args = spec.get("args", [])
    kwargs = spec.get("kwargs", {})
    mod = importlib.import_module(module_name)
    fn = getattr(mod, callable_name)
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def _run_subprocess(spec: dict[str, Any]) -> float:
    cmd = spec["cmd"]
    env = os.environ.copy()
    env.update(spec.get("env", {}))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, check=False, capture_output=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(
            f"[paired_measure] subprocess exited {r.returncode}: {cmd}\n"
            f"stderr tail:\n{r.stderr[-2048:].decode('utf-8', errors='replace')}\n"
        )
    return elapsed


DISPATCH = {
    "python_callable": _run_python_callable,
    "subprocess": _run_subprocess,
}


def build_order(iters: int, rng: random.Random) -> list[str]:
    """Produce an interleaved order of length ``2 * iters`` with equal counts.

    Start from strict alternation B, C, B, C, ..., then apply a small
    randomized swap pass so drift is not perfectly correlated with
    trial parity. Each variant appears exactly ``iters`` times.
    """
    order = ["B", "C"] * iters
    # Perform iters // 4 random adjacent swaps; keeps counts equal by
    # construction (swap preserves multiset).
    n_swaps = max(1, iters // 4)
    for _ in range(n_swaps):
        i = rng.randrange(len(order) - 1)
        order[i], order[i + 1] = order[i + 1], order[i]
    return order


def bootstrap_ci_median(diffs: list[float], iters: int, rng: random.Random) -> tuple[float, float]:
    """Return (lo, hi) 95% CI for the median of ``diffs`` via percentile bootstrap."""
    if not diffs:
        return (float("nan"), float("nan"))
    n = len(diffs)
    medians = []
    for _ in range(iters):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo = medians[int(0.025 * iters)]
    hi = medians[int(0.975 * iters)]
    return lo, hi


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    srt = sorted(values)
    n = len(srt)
    q1 = srt[max(0, int(0.25 * n) - 1)]
    q3 = srt[min(n - 1, int(0.75 * n))]
    return {
        "n": n,
        "min": srt[0],
        "median": statistics.median(srt),
        "mean": statistics.mean(srt),
        "max": srt[-1],
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired baseline-vs-candidate timing harness.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--out", default="paired_result.json")
    parser.add_argument("--bootstrap-iters", type=int, default=10_000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    iters = int(cfg.get("iters", 10))
    warmup = int(cfg.get("warmup_iters", 1))
    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)

    baseline = cfg["baseline"]
    candidate = cfg["candidate"]
    if baseline["kind"] != candidate["kind"]:
        print("baseline and candidate must share the same 'kind'", file=sys.stderr)
        return 2
    runner = DISPATCH.get(baseline["kind"])
    if runner is None:
        print(f"unknown kind: {baseline['kind']}", file=sys.stderr)
        return 2

    # Warmup runs (not timed toward results).
    for _ in range(warmup):
        runner(baseline)
        runner(candidate)

    order = build_order(iters, rng)
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    trial_log: list[dict[str, Any]] = []
    for idx, tag in enumerate(order):
        spec = baseline if tag == "B" else candidate
        elapsed = runner(spec)
        trial_log.append({"idx": idx, "variant": tag, "elapsed_s": elapsed})
        if tag == "B":
            baseline_times.append(elapsed)
        else:
            candidate_times.append(elapsed)

    # Pair by trial index within variant (i.e. the i-th B pairs with the i-th C).
    pairs = list(zip(baseline_times, candidate_times))
    diffs_abs = [c - b for (b, c) in pairs]
    diffs_pct = [((c - b) / b * 100.0) if b else float("nan") for (b, c) in pairs]

    ci_abs = bootstrap_ci_median(diffs_abs, args.bootstrap_iters, rng)
    ci_pct = bootstrap_ci_median(diffs_pct, args.bootstrap_iters, rng)

    result = {
        "config": cfg,
        "order": order,
        "trial_log": trial_log,
        "baseline_summary_s": summarize(baseline_times),
        "candidate_summary_s": summarize(candidate_times),
        "paired_diff_abs_s": summarize(diffs_abs),
        "paired_diff_pct": summarize(diffs_pct),
        "bootstrap_ci_median_abs_s": {"lo": ci_abs[0], "hi": ci_abs[1]},
        "bootstrap_ci_median_pct": {"lo": ci_pct[0], "hi": ci_pct[1]},
        "seed": seed,
        "iters": iters,
        "warmup_iters": warmup,
        "bootstrap_iters": args.bootstrap_iters,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"=== paired measurement (n={iters} pairs, seed={seed}) ===")
    print(f"baseline  median={result['baseline_summary_s']['median']:.4f}s  "
          f"IQR={result['baseline_summary_s']['iqr']:.4f}s  "
          f"min={result['baseline_summary_s']['min']:.4f}s  "
          f"max={result['baseline_summary_s']['max']:.4f}s")
    print(f"candidate median={result['candidate_summary_s']['median']:.4f}s  "
          f"IQR={result['candidate_summary_s']['iqr']:.4f}s  "
          f"min={result['candidate_summary_s']['min']:.4f}s  "
          f"max={result['candidate_summary_s']['max']:.4f}s")
    print(f"paired delta (c-b) median={result['paired_diff_abs_s']['median']:.4f}s "
          f"[{ci_abs[0]:.4f}, {ci_abs[1]:.4f}] 95% CI")
    print(f"paired delta pct        median={result['paired_diff_pct']['median']:.2f}%    "
          f"[{ci_pct[0]:.2f}%, {ci_pct[1]:.2f}%] 95% CI")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
