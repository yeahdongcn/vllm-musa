#!/usr/bin/env python3
"""MUSA-0162 bench: topk_softmax + moe_align_block_size (csrc/moe/).

MoE routing critical path: softmax over experts -> top-k experts ->
align tokens into expert-blocks for the grouped GEMM.

Note: MUSA needs the _moe_C binding too — load it if it exists.

Usage:  python3 benchmarks/op_perf/bench_moe-routing-kernels.py
"""
from __future__ import annotations
import json, os, statistics, time, torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401
import os as _os
for _so in ("/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
            "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so",
            "/ws/vllm/_moe_C.cpython-310-x86_64-linux-gnu.so"):
    if _os.path.exists(_so):
        torch.ops.load_library(_so)
del _os

_DEVICE = "musa"


def _sync(): torch.musa.synchronize()
def _peak(): return float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9


def _io_bytes(s):
    T, E, K = s["T"], s["E"], s["K"]
    # read gating (T*E*4) + write topk_weights/indices/token_expert_indices
    return T * E * 4 + T * K * 4 * 3


def _build_softmax(s):
    T, E, K = s["T"], s["E"], s["K"]
    gating = torch.randn(T, E, device=_DEVICE, dtype=torch.float32)
    topk_weights = torch.empty(T, K, device=_DEVICE, dtype=torch.float32)
    topk_indices = torch.empty(T, K, device=_DEVICE, dtype=torch.int32)
    token_expert_indices = torch.empty(T, K, device=_DEVICE, dtype=torch.int32)
    return topk_weights, topk_indices, token_expert_indices, gating


def _op_softmax_upstream(t):
    topk_w, topk_i, token_e, gating = t
    torch.ops._moe_C.topk_softmax(topk_w, topk_i, token_e, gating, True, None)


def _op_softmax_musa(t):
    topk_w, topk_i, token_e, gating = t
    torch.ops._C_musa_ops.musa_topk_softmax(topk_w, topk_i, token_e, gating, True)


_op_softmax = _op_softmax_upstream  # legacy alias


def _bench(name, s, build, op, nw=20, ni=100):
    t = build(s)
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
    print(f"{name:<32s} med_us={med:>8.2f}  GB/s={bps/1e9:>7.1f}  roofline={pct:>6.2f}%")
    return {"name": name, "shape": s, "median_us": med, "achieved_gbps": bps/1e9, "roofline_pct": pct}


SHAPES = [
    {"T": 32,   "E": 128, "K": 8},
    {"T": 512,  "E": 128, "K": 8},
    {"T": 4096, "E": 128, "K": 8},
]


def main():
    have_upstream = (hasattr(torch.ops, "_moe_C") and
                     getattr(torch.ops._moe_C, "topk_softmax", None) is not None)
    have_musa = (hasattr(torch.ops, "_C_musa_ops") and
                 getattr(torch.ops._C_musa_ops, "musa_topk_softmax", None) is not None)
    print(f"MUSA-0162 topk_softmax (peak GDDR6 = {_peak()/1e9:.0f} GB/s)")
    print(f"Available impls: upstream={have_upstream}  musa_port={have_musa}\n")
    results = []
    for s in SHAPES:
        name = f"T{s['T']:>4d}_E{s['E']}_K{s['K']}"
        if have_upstream:
            try: results.append(_bench(f"upstream_{name}", s, _build_softmax, _op_softmax_upstream))
            except Exception as e: print(f"upstream_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
        if have_musa:
            try: results.append(_bench(f"musa-port_{name}", s, _build_softmax, _op_softmax_musa))
            except Exception as e: print(f"musa-port_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
    out = os.environ.get("MUSA_0162_BENCH_OUT", "/tmp/musa_0162_moe_routing.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
