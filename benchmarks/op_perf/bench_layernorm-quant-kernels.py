#!/usr/bin/env python3
"""MUSA-0156 bench: rms_norm_static_fp8_quant (csrc/layernorm_quant_kernels.cu).

Measures upstream-csrc kernel at M2.5-relevant shapes. GDDR6 anchor 1200 GB/s.
Usage:  python3 benchmarks/op_perf/bench_layernorm-quant-kernels.py
"""
from __future__ import annotations
import json, os, statistics, time, torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401
import os as _os
for _so in ("/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
            "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so"):
    if _os.path.exists(_so):
        torch.ops.load_library(_so)
del _os

_DEVICE = "musa"


def _sync(): torch.musa.synchronize()
def _peak(): return float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9


def _io_bytes(s):
    M, N = s["M"], s["N"]
    # read input bf16 + read weight bf16 + write fp8 + read scale
    return M * N * 2 + N * 2 + M * N * 1 + 4


def _build(s):
    M, N = s["M"], s["N"]
    inp = torch.randn(M, N, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(N, device=_DEVICE, dtype=torch.bfloat16) * 0.1 + 1.0
    result = torch.empty(M, N, device=_DEVICE, dtype=torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device=_DEVICE, dtype=torch.float32)
    return result, inp, weight, scale


def _op_upstream(t):
    result, inp, weight, scale = t
    torch.ops._C.rms_norm_static_fp8_quant(result, inp, weight, scale, 1e-5)


def _op_musa(t):
    result, inp, weight, scale = t
    torch.ops._C_musa_ops.musa_rms_norm_static_fp8_quant(result, inp, weight, scale, 1e-5)


_op = _op_upstream  # legacy alias


def _bench(name, s, op, nw=20, ni=100):
    t = _build(s)
    for _ in range(nw): op(t)
    _sync()
    durs = []
    for _ in range(ni):
        _sync(); t0 = time.perf_counter_ns(); op(t); _sync()
        durs.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(durs)
    io = _io_bytes(s)
    bps = io / (med * 1e-6) if med > 0 else 0
    pct = bps / _peak() * 100
    print(f"{name:<40s} med_us={med:>8.2f}  GB/s={bps/1e9:>7.1f}  roofline={pct:>6.2f}%")
    return {"name": name, "shape": s, "median_us": med, "achieved_gbps": bps/1e9, "roofline_pct": pct}


SHAPES = [{"M": 1, "N": 6144}, {"M": 32, "N": 6144}, {"M": 128, "N": 6144}, {"M": 4096, "N": 6144}]


def main():
    have_upstream = getattr(torch.ops._C, "rms_norm_static_fp8_quant", None) is not None
    have_musa = (hasattr(torch.ops, "_C_musa_ops") and
                 getattr(torch.ops._C_musa_ops, "musa_rms_norm_static_fp8_quant", None) is not None)
    print(f"MUSA-0156 rms_norm_static_fp8_quant (peak GDDR6 = {_peak()/1e9:.0f} GB/s)")
    print(f"Available impls: upstream={have_upstream}  musa_port={have_musa}\n")
    results = []
    for s in SHAPES:
        name = f"M{s['M']:>4d}_N{s['N']:>5d}"
        if have_upstream:
            try: results.append(_bench(f"upstream_{name}", s, _op_upstream))
            except Exception as e: print(f"upstream_{name}: FAIL ({type(e).__name__}: {str(e)[:80]})")
        if have_musa:
            try: results.append(_bench(f"musa-port_{name}", s, _op_musa))
            except Exception as e: print(f"musa-port_{name}: FAIL ({type(e).__name__}: {str(e)[:80]})")
    out = os.environ.get("MUSA_0156_BENCH_OUT", "/tmp/musa_0156_layernorm_quant.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
