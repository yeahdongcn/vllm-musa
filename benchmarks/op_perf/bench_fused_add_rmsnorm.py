#!/usr/bin/env python3
"""MUSA-0150 regression bench for csrc/musa/fused_add_rmsnorm.mu (V14).

Per-shape lockstep timing + achieved-vs-GDDR6-roofline ratio. Captures the
kernel's current performance band; future kernel edits should not regress
this band by more than 10 %.

Reference (silicon-validated 2026-04-28..29 on MTT-S5000 worker33017):
  sphere-kb/probes/sgl_kernel_ports_20260428/01-fused_add_rmsnorm/RESULTS.md

  Frozen bands at large M (representative):
    M=128  N=4096  → ~+62 % over sgl-kernel  (sphere V8 vs sgl)
    M=512  N=12288 → ~1000 GB/s (≈75 % of GDDR6 ceiling)
    M=4096 N=12288 → ~1470 GB/s (≈ GDDR6 roofline)

  Small-M (cookbook decode), launch-overhead bound:
    M=1    N=6144  → ~6 µs/call (≈2.8 % SM utilization, NOT bandwidth bound)
    M=6    N=6144  → similar

Usage (inside authorized MUSA container, with vllm-musa editable-installed):

  python3 benchmarks/op_perf/bench_fused_add_rmsnorm.py

Roofline ratios require the GDDR6 bandwidth anchor exported via:

  export OP_PERF_PEAK_GDDR6_GBPS=1200   # read+write practical
  # or 1400 for read-dominated patterns; see sphere-kb claim
  # s5000.practical_gddr6_bandwidth_updated.
"""
from __future__ import annotations
import json
import os
import statistics
import sys
import time

import torch

# Prime MUSA patches before any torch.cuda symbols are captured.
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


# --- bench primitives (self-contained — does not depend on the workspace
# harness path, so the bench works regardless of where vllm-musa is built) ---


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
    achieved_bps = io_bytes / (median_us * 1e-6) if median_us > 0 else 0.0
    bw_ratio = achieved_bps / _peak_gbps() if _peak_gbps() > 0 else None

    print(
        f"{name:<40s} median_us={median_us:>8.2f} "
        f"min={min_us:>7.2f} p95={p95_us:>7.2f} "
        f"GB/s={achieved_bps/1e9:>7.1f} "
        f"roofline_pct={(bw_ratio or 0)*100:>6.2f}%"
    )
    return {
        "name": name,
        "shape": shape,
        "median_us": median_us,
        "min_us": min_us,
        "p95_us": p95_us,
        "io_bytes": io_bytes,
        "achieved_gbps": achieved_bps / 1e9,
        "roofline_pct": (bw_ratio or 0) * 100,
    }


# --- fused_add_rmsnorm specifics -------------------------------------------


def _build_inputs(shape):
    M, N = shape["M"], shape["N"]
    dtype = shape.get("dtype", torch.bfloat16)
    inp = torch.randn(M, N, device=_DEVICE, dtype=dtype) * 0.1
    res = torch.randn(M, N, device=_DEVICE, dtype=dtype) * 0.1
    w = torch.randn(N, device=_DEVICE, dtype=dtype) * 0.1 + 1.0
    return inp, res, w


def _run_op(inputs):
    inp, res, w = inputs
    # In-place fused add + rms norm: input -> normed; residual -> sum.
    torch.ops._C_musa_ops.musa_fused_add_rms_norm(inp, res, w, 1e-5)


def _io_bytes(shape):
    M, N = shape["M"], shape["N"]
    elem = 2  # bf16
    # input load + residual load + residual store + weight load + output store
    # weight load is amortized over M, but we count it conservatively once.
    return M * N * elem * 4 + N * elem


# Representative shapes for the M2.5 cookbook (hidden ≈ 6144) + the
# probe's published shapes (so this bench rerun matches the silicon
# RESULTS.md numbers within timing noise).
SHAPES = [
    # cookbook decode regime (launch-overhead bound)
    {"M": 1,    "N": 6144},
    {"M": 6,    "N": 6144},   # Eagle3 5-deep verify (1 base + 5 chain)
    # mid-batch
    {"M": 32,   "N": 6144},
    {"M": 128,  "N": 6144},
    # probe-aligned shapes (for cross-check vs RESULTS.md)
    {"M": 256,  "N": 4096},
    {"M": 512,  "N": 8192},
    # cookbook prefill regime (bandwidth bound, expected near roofline)
    {"M": 4096, "N": 6144},
]


def main():
    results = []
    print("fused_add_rmsnorm.mu (V14) regression bench")
    print(f"peak GDDR6: {_peak_gbps()/1e9:.0f} GB/s "
          f"(override via OP_PERF_PEAK_GDDR6_GBPS)")
    print()
    for shape in SHAPES:
        name = f"M{shape['M']:>4d}_N{shape['N']:>5d}"
        results.append(_run_bench(
            name, shape, _build_inputs, _run_op, _io_bytes
        ))

    out_path = os.environ.get(
        "MUSA_0150_BENCH_OUT", "/tmp/musa_0150_fused_add_rmsnorm.json"
    )
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print()
    print(f"Results JSON: {out_path}")
    print()
    print("Regression check: if any large-M (M >= 256) shape drops below "
          "the historical band (~1000 GB/s at M=512, ~1470 GB/s at M >= 1024) "
          "by more than 10 %, the kernel has regressed.")


if __name__ == "__main__":
    main()
