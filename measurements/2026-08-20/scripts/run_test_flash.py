"""Minimal test_flash driver for get_read_writes instrumentation.

Runs the same OpSpec-tiling flash-attention workload from torch-spyre
`tests/inductor/test_opspec_tiling.py::TestOpSpecTiling::test_flash`
(also used by the compiler-timing PR #3806 study), but WITHOUT any
of that study's timing infrastructure. The only instrumentation
loaded is the sibling `instrument_read_writes.py`, which
monkey-patches `ComputedBuffer.get_read_writes` and
`torch_spyre._inductor.pass_utils.op_read_writes` and writes one
JSONL row per call.

No modification of torch-spyre source. No modification of the
compiler-timing repo. Cold-compile hygiene:
`TORCHINDUCTOR_CACHE_DIR` is required and the cache directory is
wiped before the compile.

Usage inside the pod:

    export TORCH_SPYRE_INSTRUMENT_LOG=/tmp/test_flash_read_writes.jsonl
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_test_flash
    rm -rf "$TORCHINDUCTOR_CACHE_DIR"
    cd $HOME/torch-spyre-work/torch-spyre
    source .venv/bin/activate
    python /path/to/run_test_flash.py

The workload runs at the study's baseline configuration
(B=1, H=8, D=128, Lq=512, Lk=1024,
 b_block_size=1, h_block_size=4, q_block_size=256, kv_block_size=512),
which produces 8 inner_bodies at compile_fx entry — the same shape
the PR #3806 timing study measures.

The `build_flash_closure` body is a verbatim copy of the closure in
torch-spyre's test file at the pinned SHA
`fea0c4be901e1383b1f700dbad8887128b0fcb27`.
"""

from __future__ import annotations

import gc
import math
import os
import sys

# Fail closed if cold-compile hygiene is not set.
if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    print(
        "FATAL: TORCHINDUCTOR_CACHE_DIR must be set (cold-compile hygiene).",
        file=sys.stderr,
    )
    sys.exit(2)

# Install instrumentation BEFORE importing torch_spyre so the
# monkey-patch of op_read_writes attaches to the imported symbol.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "patches"))
import instrument_read_writes  # noqa: F401 — side effect: installs patches on import

import torch  # noqa: E402


def build_flash_closure(
    B: int,
    H: int,
    D: int,
    Lq: int,
    Lk: int,
    b_block_size: int,
    h_block_size: int,
    q_block_size: int,
    kv_block_size: int,
):
    """Verbatim from torch-spyre tests/inductor/test_opspec_tiling.py::test_flash."""

    def flash(queries, keys, values, mask):
        scale = 1.0 / math.sqrt(math.sqrt(D))
        output = torch.zeros_like(queries)
        real_max = torch.full(
            (B, H, Lq, 1), float("-inf"), dtype=queries.dtype, device=queries.device
        )
        real_sum = torch.zeros(
            B, H, Lq, 1, dtype=queries.dtype, device=queries.device
        )
        for b in range(0, B, b_block_size):
            for h in range(0, H, h_block_size):
                for lq in range(0, Lq, q_block_size):
                    m_i = real_max[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ]
                    l_i = real_sum[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ]
                    o_i = output[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ]
                    q_tile = queries[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ] * scale
                    for lk in range(0, Lk, kv_block_size):
                        k_tile = keys[
                            b : b + b_block_size,
                            h : h + h_block_size,
                            lk : lk + kv_block_size,
                        ] * scale
                        v_tile = values[
                            b : b + b_block_size,
                            h : h + h_block_size,
                            lk : lk + kv_block_size,
                        ]
                        m_tile = mask[
                            :, :, lq : lq + q_block_size, lk : lk + kv_block_size
                        ]
                        s = torch.matmul(q_tile, k_tile.transpose(-1, -2)) + m_tile
                        m_ij = torch.maximum(m_i, s.max(dim=-1, keepdim=True).values)
                        p = torch.exp(s - m_ij)
                        l_ij = l_i * torch.exp(m_i - m_ij) + p.sum(
                            dim=-1, keepdim=True
                        )
                        o_ij = o_i * torch.exp(m_i - m_ij) + torch.matmul(p, v_tile)
                        m_i = m_ij
                        l_i = l_ij
                        o_i = o_ij
                    real_max[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ] = m_i
                    real_sum[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ] = l_i
                    output[
                        b : b + b_block_size, h : h + h_block_size, lq : lq + q_block_size
                    ] = o_i / l_i
        return output

    return flash


def main() -> int:
    B, H, D, Lq, Lk = 1, 8, 128, 512, 1024
    b_bs, h_bs, q_bs, kv_bs = 1, 4, 256, 512

    torch.manual_seed(0xAFFE)
    flash = build_flash_closure(B, H, D, Lq, Lk, b_bs, h_bs, q_bs, kv_bs)

    queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
    keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
    values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
    causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
    mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
    mask_t.masked_fill_(~causal, float("-inf"))

    print("moving tensors to spyre device", flush=True)
    queries_s = queries_t.to("spyre")
    keys_s = keys_t.to("spyre")
    values_s = values_t.to("spyre")
    mask_s = mask_t.to(device="spyre")

    gc.collect()

    print("beginning cold compile of test_flash", flush=True)
    flash_spyre = torch.compile(flash)
    _ = flash_spyre(queries_s, keys_s, values_s, mask_s)
    print("compile complete", flush=True)
    print(
        f"instrumentation log at: {os.environ.get('TORCH_SPYRE_INSTRUMENT_LOG', '/tmp/torch_spyre_read_writes.jsonl')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
