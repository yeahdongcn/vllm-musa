#!/usr/bin/env python3
"""MUSA-0154 regression bench for rotary embedding (RoPE).

Currently vllm-musa consumes `csrc/pos_encoding_kernels.cu` directly from
upstream vllm — there is NO local MUSA-port replacement. The
sphere-kb probe at
`sphere-kb/probes/sgl_kernel_ports_20260428/02-rotary_embedding/`
ships a silicon-validated V5 design (+79.6 % geomean over sgl-kernel,
up to 4.02× at T=2048 H=32) that should be ported into vllm-musa as
`csrc/musa/pos_encoding.mu` + REPLACE_VLLM_FILES entry.

This bench measures whichever RoPE impl is currently registered
(upstream until the port lands; ported V5 afterward) at M2.5-relevant
shapes, so we can quantify the lift when the port lands.

Usage (inside authorized MUSA container with vllm-musa editable-installed):

  python3 benchmarks/op_perf/bench_rotary_embedding.py
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


def _device_sync(): torch.musa.synchronize()


def _io_bytes(shape):
    T = shape["T"]
    H = shape["H"]
    Hkv = shape.get("Hkv", H)
    D = shape["D"]
    R = shape.get("R", D)
    elem = 2  # bf16
    # Q + K read+write + cos_sin_cache read
    return 2 * T * (H + Hkv) * D * elem + T * R * elem


def _peak_gbps():
    return float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9


def _run_bench(name, shape, build_inputs, run_op, n_warmup=20, n_iters=100):
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
    io = _io_bytes(shape)
    gbps = (io / 1e9) / (median_us * 1e-6) if median_us > 0 else 0
    bw_pct = gbps * 1e9 / _peak_gbps() * 100
    print(f"{name:<40s} median_us={median_us:>8.2f}  GB/s={gbps:>7.1f}  roofline_pct={bw_pct:>6.2f}%")
    return {"name": name, "shape": shape, "median_us": median_us,
            "achieved_gbps": gbps, "roofline_pct": bw_pct}


def _build_inputs(shape):
    T = shape["T"]
    H = shape["H"]  # num_heads (Q)
    Hkv = shape.get("Hkv", H)  # num_kv_heads
    D = shape["D"]  # head_dim
    R = shape.get("R", D)  # rot_dim, default = D
    dtype = shape.get("dtype", torch.bfloat16)
    positions = torch.arange(T, device=_DEVICE, dtype=torch.int64)
    query = torch.randn(T, H * D, device=_DEVICE, dtype=dtype)
    key = torch.randn(T, Hkv * D, device=_DEVICE, dtype=dtype)
    cos_sin_cache = torch.randn(T, R, device=_DEVICE, dtype=dtype)
    return positions, query, key, cos_sin_cache, R


def _run_op(inputs):
    positions, query, key, cos_sin_cache, head_size = inputs
    # vllm's RoPE binding (upstream); on MUSA today this calls into the
    # upstream pos_encoding_kernels.cu via torch.ops._C.rotary_embedding.
    # When the V5 port lands as `torch.ops._C_musa_ops.musa_rotary_embedding`
    # we'll switch the call here.
    # Prefer the MUSA-port (MUSA-0154 V5) when available; fall back to upstream.
    op = getattr(torch.ops._C_musa_ops, "musa_rotary_embedding", None)
    if op is None:
        op = getattr(torch.ops._C, "rotary_embedding", None)
    if op is None:
        raise RuntimeError(
            "neither torch.ops._C_musa_ops.musa_rotary_embedding nor "
            "torch.ops._C.rotary_embedding is available"
        )
    op(positions, query, key, head_size, cos_sin_cache, True)  # is_neox=True


# Shapes match the probe's published RESULTS.md so cross-check is direct.
SHAPES = [
    {"T": 1,    "H": 32, "D": 128},
    {"T": 1,    "H": 64, "D": 128},
    {"T": 8,    "H": 32, "D": 128},
    {"T": 32,   "H": 32, "D": 128},
    {"T": 32,   "H": 64, "D": 128},
    {"T": 128,  "H": 32, "D": 128},
    {"T": 512,  "H": 32, "D": 128},
    {"T": 512,  "H": 64, "D": 128},
    {"T": 2048, "H": 32, "D": 128},
]


def main():
    print("rotary_embedding (RoPE) bench (M2.5-relevant shapes)")
    print()
    print("Today: measures whichever RoPE impl is registered. Currently")
    print("vllm-musa uses upstream vllm's csrc/pos_encoding_kernels.cu")
    print("(CUDA-style scalar loads, no LSU bypass, no async G2S).")
    print()
    print("Target: sphere-kb probe V5 design (see PERF NOTE in")
    print("csrc/musa/pos_encoding.mu when ported). Probe achievement on")
    print("the same hardware vs sgl-kernel MUSA port:")
    print("  geomean +79.6 %, max +301.8 % at T=2048 H=32 (4.02× speedup)")
    print()
    results = []
    for shape in SHAPES:
        name = f"T{shape['T']:>4d}_H{shape['H']:>3d}_D{shape['D']:>3d}"
        try:
            results.append(_run_bench(name, shape, _build_inputs, _run_op))
        except Exception as e:
            print(f"{name}: skip ({e})")

    out_path = os.environ.get(
        "MUSA_0154_BENCH_OUT", "/tmp/musa_0154_rotary_embedding.json"
    )
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out_path}")


if __name__ == "__main__":
    main()
