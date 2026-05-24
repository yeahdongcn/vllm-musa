#!/usr/bin/env python3
"""MUSA-0155 bench skeleton for silu_mul + fp8 quant (activation_kernels.cu).

Times whichever silu+mul+fp8-quant impl is registered. Use this to measure
the deferred V1..V3 ladder (Vec16 + LSU bypass + async G2S) when ported.

Today the call path goes through the local-replacement
csrc/quantization/activation_kernels.cu (kernel
silu_mul_fp8_quant_deep_gemm_kernel) when VLLM_MUSA_SPHERE_SILU_FP8=1.
"""
import time, statistics, os, json, torch
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

def _sync(): torch.musa.synchronize()


def bench(name, shape, n_warmup=20, n_iters=100):
    M, N = shape["M"], shape["N"]
    x = torch.randn(M, 2 * N, device="musa", dtype=torch.bfloat16) * 0.1
    out_q = torch.empty(M, N, device="musa", dtype=torch.float8_e4m3fn)
    out_s = torch.zeros(M, N // 128, device="musa", dtype=torch.float32)
    op = getattr(torch.ops._C_musa_ops, "silu_and_mul_per_token_group_fp8_quant", None)
    if op is None:
        print(f"{name}: SKIP (binding not found)")
        return None
    for _ in range(n_warmup):
        op(x, out_q, out_s, 128, 1e-10, -448.0, 448.0)
    _sync()
    durs = []
    for _ in range(n_iters):
        _sync()
        t0 = time.perf_counter_ns()
        op(x, out_q, out_s, 128, 1e-10, -448.0, 448.0)
        _sync()
        durs.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(durs)
    # io: read bf16 x (2*N), write fp8 (N), write fp32 scale (N/128)
    io_bytes = M * 2 * N * 2 + M * N * 1 + M * (N // 128) * 4
    peak_gbps = float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9
    gbps = (io_bytes / 1e9) / (med * 1e-6) if med > 0 else 0
    bw_pct = gbps * 1e9 / peak_gbps * 100
    print(f"{name:<30s} median_us={med:>8.2f}  GB/s={gbps:>7.1f}  roofline_pct={bw_pct:>6.2f}%")
    return {"name": name, "shape": shape, "median_us": med,
            "achieved_gbps": gbps, "roofline_pct": bw_pct}


SHAPES = [
    {"M": 1,    "N": 6144},
    {"M": 32,   "N": 6144},
    {"M": 128,  "N": 6144},
    {"M": 4096, "N": 6144},
]


if __name__ == "__main__":
    results = []
    for s in SHAPES:
        r = bench(f"M{s['M']:>4d}_N{s['N']:>5d}", s)
        if r: results.append(r)
    json.dump(results, open(os.environ.get(
        "MUSA_0155_BENCH_OUT", "/tmp/musa_0155_silu_mul_fp8.json"
    ), "w"), indent=2)
