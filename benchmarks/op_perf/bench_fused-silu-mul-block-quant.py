#!/usr/bin/env python3
"""MUSA-0158 bench: silu_and_mul_per_block_quant
(csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu).

Per-block FP8 quant of silu(x[:,:N]) * x[:,N:] with N=hidden, block_size=128.

Usage:  python3 benchmarks/op_perf/bench_fused-silu-mul-block-quant.py
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
    M, N, G = s["M"], s["N"], s["G"]
    # read M*(2N)*2 bf16 + write M*N*1 fp8 + write M*(N/G)*4 scales
    return M * 2 * N * 2 + M * N * 1 + M * (N // G) * 4


def _build(s):
    M, N, G = s["M"], s["N"], s["G"]
    inp = torch.randn(M, 2 * N, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    out = torch.empty(M, N, device=_DEVICE, dtype=torch.float8_e4m3fn)
    scales = torch.empty(M, N // G, device=_DEVICE, dtype=torch.float32)
    return out, inp, scales, G


def _op_upstream(t):
    out, inp, scales, G = t
    torch.ops._C.silu_and_mul_per_block_quant(out, inp, scales, G, None, False)


def _op_musa(t):
    out, inp, scales, G = t
    torch.ops._C_musa_ops.musa_silu_and_mul_per_block_quant(out, inp, scales, G)


_op = _op_upstream  # legacy alias


def _torch_native_op(t):
    """Pure-torch reference: silu(gate) * up, then per-128-block FP8 quant.
    If THIS is faster than the upstream kernel, the upstream kernel is
    broken on MUSA."""
    _, inp, _, G = t
    gate, up = inp.chunk(2, dim=-1)
    activated = (gate.float() * torch.sigmoid(gate.float()) * up.float())  # silu * up
    M, N = activated.shape
    blocks = activated.view(M, N // G, G)
    abs_max = blocks.abs().amax(-1, keepdim=True)
    scale_local = abs_max.clamp(min=1e-12) / 448.0
    quantized = (blocks / scale_local).clamp(-448, 448).to(torch.float8_e4m3fn)
    out_local = quantized.view(M, N)
    return out_local, scale_local.squeeze(-1)


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


SHAPES = [
    {"M": 1,    "N": 6144, "G": 128},
    {"M": 32,   "N": 6144, "G": 128},
    {"M": 128,  "N": 6144, "G": 128},
    {"M": 4096, "N": 6144, "G": 128},
]


def main():
    have_upstream = getattr(torch.ops._C, "silu_and_mul_per_block_quant", None) is not None
    have_musa = (hasattr(torch.ops, "_C_musa_ops") and
                 getattr(torch.ops._C_musa_ops, "musa_silu_and_mul_per_block_quant", None) is not None)
    print(f"MUSA-0158 silu_and_mul_per_block_quant baseline (peak GDDR6 = {_peak()/1e9:.0f} GB/s)\n")
    print(f"Available impls: upstream={have_upstream}  musa_port={have_musa}\n")
    results = []
    for s in SHAPES:
        name = f"M{s['M']:>4d}_N{s['N']:>5d}_G{s['G']}"
        if have_upstream:
            try: results.append(_bench(f"upstream_{name}", s, _op_upstream))
            except Exception as e: print(f"upstream_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
        try: results.append(_bench(f"torch-native_{name}", s, _torch_native_op))
        except Exception as e: print(f"torch-native_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
        if have_musa:
            try: results.append(_bench(f"musa-port_{name}", s, _op_musa))
            except Exception as e: print(f"musa-port_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
    out = os.environ.get("MUSA_0158_BENCH_OUT", "/tmp/musa_0158_silu_block_quant.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
