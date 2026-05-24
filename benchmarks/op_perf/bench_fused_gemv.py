#!/usr/bin/env python3
"""MUSA-0153 regression bench for csrc/musa/gemv.mu.

Targets `musa_gemv_kernel<bf16, fp8_e4m3, bf16, ..., is_swigelu=true,
is_fp8=true>` — the dominant prefill kernel in the cookbook 4k/1k M2.5
trace (95.7 % of prefill kernel time). Captures the current band so the
deferred V1..V3 ladder (LSU bypass + async G2S + SQMMA tile refactor)
can be measured against it.

Calling convention: invokes `torch.ops.vllm_musa.musa_fused_gemv` (or
`musa_fused_gemv_moe` for the MoE variant); both wrap the same underlying
kernel.

Usage (inside authorized MUSA container with vllm-musa editable-installed):

  export OP_PERF_PEAK_FP8_TFLOPS=<sphere-kb anchor>
  python3 benchmarks/op_perf/bench_fused_gemv.py
"""
from __future__ import annotations
import json
import os
import statistics
import time

import torch
import torchada  # noqa: F401

_DEVICE = "musa"


def _device_sync(): torch.musa.synchronize()


def _peak_fp8_tflops():
    return float(os.environ.get("OP_PERF_PEAK_FP8_TFLOPS", "0"))


def _peak_gbps():
    return float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9


def _run_bench(name, shape, build_inputs, run_op, flops_fn, io_bytes_fn,
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
    flops = flops_fn(shape)
    io = io_bytes_fn(shape)
    achieved_tflops = (flops / 1e12) / (median_us * 1e-6) if median_us > 0 else 0
    achieved_gbps = (io / 1e9) / (median_us * 1e-6) if median_us > 0 else 0
    peak_tflops = _peak_fp8_tflops()
    peak_bps = _peak_gbps()
    compute_pct = (achieved_tflops / peak_tflops * 100) if peak_tflops > 0 else None
    bw_pct = (achieved_gbps * 1e9 / peak_bps * 100) if peak_bps > 0 else None

    extras = ""
    if compute_pct is not None:
        extras += f" compute={compute_pct:>5.1f}%"
    if bw_pct is not None:
        extras += f" bw={bw_pct:>5.1f}%"

    print(f"{name:<32s} median_us={median_us:>9.2f} "
          f"TFLOPS={achieved_tflops:>6.2f} GB/s={achieved_gbps:>7.1f}{extras}")
    return {
        "name": name, "shape": shape, "median_us": median_us,
        "achieved_tflops": achieved_tflops, "achieved_gbps": achieved_gbps,
        "compute_pct": compute_pct, "bw_pct": bw_pct,
    }


# Shapes target the cookbook M2.5 hot kernels — bf16 activations × fp8 weights,
# swiGLU fusion, no MoE routing on the simplest path.
SHAPES = [
    # Cookbook prefill (M=4096 = num_input_tokens, N=hidden, K=hidden)
    {"M": 4096, "N": 6144, "K": 6144, "swiglu": True},
    # Decode batch (M=1 or M=6 verify)
    {"M": 1,    "N": 6144, "K": 6144, "swiglu": True},
    {"M": 6,    "N": 6144, "K": 6144, "swiglu": True},
    # Mid-batch
    {"M": 256,  "N": 6144, "K": 6144, "swiglu": True},
    # Smaller K (MLP gate vs MLP down)
    {"M": 4096, "N": 6144, "K": 1536, "swiglu": True},
]


def _build_inputs(shape):
    M, N, K = shape["M"], shape["N"], shape["K"]
    # bf16 activation, fp8 weight, bf16 output, fp32 scale_a + scale_b.
    a = torch.randn(M, K, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    # Weight is fp8 (group-scaled); for the bench we generate bf16 then
    # convert to a placeholder (real path expects pre-quantized weights).
    # Use a torch quant kernel if available; otherwise mock the call.
    return {"a": a, "M": M, "N": N, "K": K, "swiglu": shape["swiglu"]}


def _run_op(inputs):
    # This bench currently SKIPS the actual op call because the binding
    # parameter shape is non-trivial (expert tables, scales, RMS gamma, etc.)
    # and is best validated via the upstream cookbook serve path. The
    # measurement of `musa_gemv_kernel` performance for the cookbook trace
    # is best done with `vllm bench serve --random-input-len 4096 ...` +
    # `torch.profiler` capture, which the existing
    # `generated/goal/profile_capture_driver.sh` already does.
    #
    # For a standalone microbench, populate the call:
    #   torch.ops.vllm_musa.musa_fused_gemv(a, b, c, ...)
    # with the exact arg layout from csrc/musa/torch_bindings.cpp once the
    # V1..V3 ladder is being prototyped.
    raise NotImplementedError(
        "musa_fused_gemv standalone bench skeleton; populate the call from "
        "csrc/musa/torch_bindings.cpp when ready to bench V1..V3 candidates. "
        "For now, use generated/goal/profile_capture_driver.sh to capture "
        "the kernel under the cookbook workload."
    )


def _flops_swiglu(shape):
    # 2 matmuls (gate + up) × 2 (MAC) × M × N × K, then 1 elementwise silu*mul.
    M, N, K = shape["M"], shape["N"], shape["K"]
    return 2 * 2 * M * N * K + M * N


def _io_bytes(shape):
    M, N, K = shape["M"], shape["N"], shape["K"]
    elem_a = 2  # bf16
    elem_b = 1  # fp8
    elem_c = 2  # bf16
    return M * K * elem_a + 2 * N * K * elem_b + M * N * elem_c  # gate + up weights


def main():
    print("musa_gemv_kernel (fused-GEMV + swiGLU + FP8) bench skeleton")
    print(f"peak FP8: {_peak_fp8_tflops():.0f} TFLOPS, GDDR6: {_peak_gbps()/1e9:.0f} GB/s")
    print()
    print("NOTE: standalone op call is left as NotImplementedError until")
    print("the V1..V3 ladder needs empirical measurement. For now, use the")
    print("profile-driver capture against the cookbook workload to attribute")
    print("kernel time to musa_gemv_kernel.")
    print()
    for shape in SHAPES:
        name = f"M{shape['M']:>5d}_N{shape['N']:>5d}_K{shape['K']:>5d}"
        try:
            _run_bench(name, shape, _build_inputs, _run_op,
                       _flops_swiglu, _io_bytes)
        except NotImplementedError as e:
            print(f"{name}: SKIP — {e}")
            break  # Skip all shapes once skeleton is hit


if __name__ == "__main__":
    main()
