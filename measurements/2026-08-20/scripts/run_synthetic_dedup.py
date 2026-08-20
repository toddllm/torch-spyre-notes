"""Synthetic dedup-stress workload.

Purpose
-------
Force `dedup_and_promote_constants` to see a large `D` (many duplicate
constants) so its O(D × N) scan is observable in the JSONL.

Design
------
Each `torch.nn.functional.pad` call with a constant fill value causes
`aten.constant_pad_nd` to lower into a separate constant-materialization
op. When N pad calls all use the same fill value, they produce N
structurally identical `SpyreConstantFallback` operations — exactly
the pattern `dedup_and_promote_constants` deduplicates.

We stack K such pads with the SAME fill value (so `_constant_key` groups
them all into one group of K duplicates, giving D = K-1 non-canonical
duplicates), plus a small amount of downstream work to keep the graph
non-trivial and give the scan real ops to iterate over.

The workload is intentionally simple. The goal is not "realistic" — it's
"maximizes the specific hot path so we can see whether the cost is
where we think it is." Compare the resulting numbers to the test_flash
run to see whether the same scan is dominant in a realistic compile
or a corner case.
"""

from __future__ import annotations

import gc
import os
import sys

if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    print(
        "FATAL: TORCHINDUCTOR_CACHE_DIR must be set (cold-compile hygiene).",
        file=sys.stderr,
    )
    sys.exit(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "patches"))
import instrument_read_writes  # noqa: F401 — installs on import

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# Number of duplicate constants we want the pass to see. K pads with
# the same fill value produce K SpyreConstantFallback ops, of which
# K-1 are non-canonical duplicates.
K = int(os.environ.get("SYNTHETIC_K", "16"))


def stacked_pads(x: torch.Tensor) -> torch.Tensor:
    """Apply K pads with the same fill value, then reduce.

    Each F.pad with a scalar `value` lowers via aten.constant_pad_nd,
    which materializes the fill scalar as its own SpyreConstantFallback
    op. K identical scalar values → K duplicates in one group.
    """
    outs = []
    for _ in range(K):
        y = F.pad(x, (1, 1), value=0.5)
        outs.append(y)
    # Reduce to keep the graph output shape independent of K.
    return torch.stack(outs, dim=0).sum(dim=0)


def main() -> int:
    # Skip torch.manual_seed entirely — it iterates custom devices via
    # `_seed_custom_device`, which invokes torch_spyre's device-init path.
    # That path calls into the runtime and can transiently fail
    # (RAS::VFIO::DeviceOpenFail on shared/detached pods). Deterministic
    # graph structure does not require RNG seeding here.
    x = torch.randn(1, 64, 128, dtype=torch.float16)

    print(f"synthetic dedup workload: K={K} duplicate constants", flush=True)
    print("moving tensor to spyre device", flush=True)
    x_s = x.to("spyre")

    gc.collect()

    print("beginning cold compile of stacked_pads", flush=True)
    fn = torch.compile(stacked_pads)
    _ = fn(x_s)
    print("compile complete", flush=True)
    print(
        f"instrumentation log at: {os.environ.get('TORCH_SPYRE_INSTRUMENT_LOG', '/tmp/torch_spyre_read_writes.jsonl')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
