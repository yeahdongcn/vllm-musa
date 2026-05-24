#!/usr/bin/env python3
"""MUSA-0152 regression bench for csrc/musa/quantization/per_token_group_quant.cu.

Targets `per_token_group_quant_128_register_kernel<bf16, fp8>` (the cookbook
FP8 quant fast path, group_size=128). Captures the kernel's current band
so the deferred V1..V3 ladder (Vec16 + LSU bypass + async G2S) can be
measured against this baseline when picked up.

Usage (inside authorized MUSA container with vllm-musa editable-installed):

  python3 benchmarks/op_perf/bench_per_token_group_quant.py
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
GROUP_SIZE = 128


def _device_sync(): torch.musa.synchronize()


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
    io_bytes = io_bytes_fn(shape)
    bps = io_bytes / (median_us * 1e-6) if median_us > 0 else 0.0
    bw_ratio = bps / _peak_gbps() if _peak_gbps() > 0 else None
    print(f"{name:<40s} median_us={median_us:>8.2f} "
          f"GB/s={bps/1e9:>7.1f} roofline_pct={(bw_ratio or 0)*100:>6.2f}%")
    return {"name": name, "shape": shape, "median_us": median_us,
            "achieved_gbps": bps / 1e9, "roofline_pct": (bw_ratio or 0) * 100}


def _build_inputs(shape):
    M, N = shape["M"], shape["N"]  # M = num_tokens, N = hidden
    assert N % GROUP_SIZE == 0, f"N={N} must be multiple of {GROUP_SIZE}"
    inp = torch.randn(M, N, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    out_q = torch.empty(M, N, device=_DEVICE, dtype=torch.float8_e4m3fn)
    num_groups = N // GROUP_SIZE
    out_s = torch.zeros(M, num_groups, device=_DEVICE, dtype=torch.float32)
    return inp, out_q, out_s


def _run_op(inputs):
    inp, out_q, out_s = inputs
    # The binding is per_token_group_quant; resolve via the dispatch shim.
    op = getattr(torch.ops._C_musa_ops, "per_token_group_fp8_quant", None)
    if op is None:
        # Try alternate binding name
        op = getattr(torch.ops._C_musa_ops, "per_token_group_fp8_quant", None)
    if op is None:
        raise RuntimeError(
            "per_token_group_quant fp8 binding not found on torch.ops._C_musa_ops. "
            "Check csrc/musa/torch_bindings.cpp for the exposed symbol."
        )
    op(inp, out_q, out_s, GROUP_SIZE, 1e-10, -448.0, 448.0,
       False,  # scale_ue8m0
       False,  # dummy_is_scale_transposed
       False)  # dummy_is_tma_aligned


def _io_bytes(shape):
    M, N = shape["M"], shape["N"]
    # Read bf16 input (2 B), write fp8 output (1 B) + fp32 scale per group (4 B / 128 elems)
    return M * N * 2 + M * N * 1 + (M * (N // GROUP_SIZE)) * 4


SHAPES = [
    {"M": 1,    "N": 6144},
    {"M": 6,    "N": 6144},
    {"M": 32,   "N": 6144},
    {"M": 128,  "N": 6144},
    {"M": 512,  "N": 6144},
    {"M": 4096, "N": 6144},
]


def main():
    print("per_token_group_quant_128_register_kernel<bf16,fp8> bench")
    print(f"peak GDDR6: {_peak_gbps()/1e9:.0f} GB/s")
    print()
    results = []
    for shape in SHAPES:
        name = f"M{shape['M']:>4d}_N{shape['N']:>5d}"
        try:
            results.append(_run_bench(name, shape, _build_inputs, _run_op, _io_bytes))
        except Exception as e:
            print(f"{name}: skip ({e})")
    out_path = os.environ.get(
        "MUSA_0152_BENCH_OUT", "/tmp/musa_0152_per_token_group_quant.json"
    )
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out_path}")


if __name__ == "__main__":
    main()
