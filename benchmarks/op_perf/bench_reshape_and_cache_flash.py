#!/usr/bin/env python3
"""MUSA-0151 regression bench for csrc/musa/cache_kernels.mu.

`reshape_and_cache_flash_nhd_kernel<T, BLOCK_X=512, TOKENS_PER_BLOCK>` is a
pure bandwidth-bound gather/scatter from (key, value) to the paged
kv-cache. Per-shape lockstep timing + achieved-vs-GDDR6 ratio. Captures
the kernel's current band; future kernel edits should not regress > 10 %.

Note: this kernel has no sphere-kb probe yet. The 'baseline' here is the
kernel's own first measurement; the V2 adaptive-BLOCK_X candidate (see
PERF NOTE in cache_kernels.mu) would re-establish the band when shipped.

Usage (inside authorized MUSA container with vllm-musa editable-installed):

  python3 benchmarks/op_perf/bench_reshape_and_cache_flash.py

Roofline ratio requires the GDDR6 anchor:

  export OP_PERF_PEAK_GDDR6_GBPS=1200   # read+write practical (sphere-kb
                                        # s5000.practical_gddr6_bandwidth_updated)
"""
from __future__ import annotations
import json
import os
import statistics
import time

import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401  -- registers MUSA backend so torch.ops.load_library can resolve MUSA symbols
# Load torch.ops directly from the .so files. Avoids the
# vllm-plugin circular-import error that fires when `import vllm`
# triggers `vllm.plugins.load_plugins_by_group()` against a
# partially-initialized vllm_musa module.
import os as _os
for _so in (
    "/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
    "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so",
):
    if _os.path.exists(_so):
        torch.ops.load_library(_so)
del _os

_DEVICE = "musa"


def _device_sync() -> None:
    torch.musa.synchronize()


def _peak_gbps() -> float:
    return float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9


def _run_bench(name, shape, build_inputs, run_op, io_bytes_fn,
               n_warmup=20, n_iters=100):
    inputs = build_inputs(shape)
    for _ in range(n_warmup):
        run_op(inputs)
    _device_sync()

    durs_us = []
    for _ in range(n_iters):
        _device_sync()
        t0 = time.perf_counter_ns()
        run_op(inputs)
        _device_sync()
        t1 = time.perf_counter_ns()
        durs_us.append((t1 - t0) / 1e3)

    median_us = statistics.median(durs_us)
    min_us = min(durs_us)
    p95_us = statistics.quantiles(durs_us, n=20)[18]
    io_bytes = io_bytes_fn(shape)
    bps = io_bytes / (median_us * 1e-6) if median_us > 0 else 0.0
    bw_ratio = bps / _peak_gbps() if _peak_gbps() > 0 else None

    print(
        f"{name:<50s} median_us={median_us:>8.2f} "
        f"min={min_us:>7.2f} p95={p95_us:>7.2f} "
        f"GB/s={bps/1e9:>7.1f} "
        f"roofline_pct={(bw_ratio or 0)*100:>6.2f}%"
    )
    return {
        "name": name,
        "shape": shape,
        "median_us": median_us,
        "io_bytes": io_bytes,
        "achieved_gbps": bps / 1e9,
        "roofline_pct": (bw_ratio or 0) * 100,
    }


def _build_inputs(shape):
    T = shape["num_tokens"]
    H = shape["num_heads"]
    D = shape["head_size"]
    B = shape["block_size"]
    NB = shape["num_blocks"]
    dtype = shape.get("dtype", torch.bfloat16)
    key = torch.randn(T, H, D, device=_DEVICE, dtype=dtype)
    value = torch.randn(T, H, D, device=_DEVICE, dtype=dtype)
    key_cache = torch.zeros(NB, B, H, D, device=_DEVICE, dtype=dtype)
    value_cache = torch.zeros(NB, B, H, D, device=_DEVICE, dtype=dtype)
    # Slot mapping: simple sequential assignment, with some -1 padding sometimes.
    slot_mapping = torch.arange(T, device=_DEVICE, dtype=torch.int64)
    return key, value, key_cache, value_cache, slot_mapping


def _run_op(inputs):
    key, value, key_cache, value_cache, slot_mapping = inputs
    torch.ops._C_musa_ops.musa_reshape_and_cache_flash_nhd(
        key, value, key_cache, value_cache, slot_mapping
    )


def _io_bytes(shape):
    T = shape["num_tokens"]
    H = shape["num_heads"]
    D = shape["head_size"]
    elem = 2  # bf16
    # Read key + value, write key_cache + value_cache. 4 × T × H × D × 2 B.
    return 4 * T * H * D * elem


# Representative shapes:
#   - cookbook M2.5 decode: num_kv_heads=8, head_size=128, block_size=16,
#     T = 1 (1-token decode) to T = 6 (verify chain pass)
#   - prefill / chunked: T = 4096 (cookbook 4k prefill)
#   - typical config: block_size 16 or 32; num_blocks set so the cache fits.
SHAPES = [
    # cookbook decode (small T)
    {"num_tokens": 1,    "num_heads": 8, "head_size": 128, "block_size": 16,
     "num_blocks": 4096},
    {"num_tokens": 6,    "num_heads": 8, "head_size": 128, "block_size": 16,
     "num_blocks": 4096},
    # mid
    {"num_tokens": 64,   "num_heads": 8, "head_size": 128, "block_size": 16,
     "num_blocks": 4096},
    {"num_tokens": 256,  "num_heads": 8, "head_size": 128, "block_size": 16,
     "num_blocks": 4096},
    # cookbook prefill (large T)
    {"num_tokens": 4096, "num_heads": 8, "head_size": 128, "block_size": 16,
     "num_blocks": 4096},
    # alt config: larger num_heads (sanity)
    {"num_tokens": 256,  "num_heads": 32, "head_size": 128, "block_size": 16,
     "num_blocks": 2048},
]


def main():
    print("reshape_and_cache_flash_nhd_kernel regression bench")
    print(f"peak GDDR6: {_peak_gbps()/1e9:.0f} GB/s")
    print()
    results = []
    for shape in SHAPES:
        name = (
            f"T{shape['num_tokens']:>4d}_H{shape['num_heads']:>2d}_"
            f"D{shape['head_size']:>3d}_B{shape['block_size']:>2d}"
        )
        results.append(_run_bench(
            name, shape, _build_inputs, _run_op, _io_bytes
        ))

    out_path = os.environ.get(
        "MUSA_0151_BENCH_OUT", "/tmp/musa_0151_reshape_and_cache_flash.json"
    )
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print()
    print(f"Results JSON: {out_path}")


if __name__ == "__main__":
    main()
