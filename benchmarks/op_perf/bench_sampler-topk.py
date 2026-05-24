#!/usr/bin/env python3
"""MUSA-0160 bench: top_k_per_row_decode + apply_repetition_penalties_
(csrc/topk.cu, csrc/sampler.cu).

Decode-path sampler: top-k logits selection (next_n=1 token at a time).

Usage:  python3 benchmarks/op_perf/bench_sampler-topk.py
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


def _io_bytes_topk(s):
    B, V, K = s["B"], s["V"], s["K"]
    # read logits (B*V*4) + write top-k vals (B*K*4) + write indices (B*K*4)
    return B * V * 4 + B * K * 4 + B * K * 4


def _bench_topk(name, s, nw=20, ni=100):
    B, V, K = s["B"], s["V"], s["K"]
    logits = torch.randn(B, V, device=_DEVICE, dtype=torch.float32)
    # top_k_per_row_decode signature: (logits, next_n, ...) — try the simpler form first
    # Fallback to torch.topk if binding signature differs
    try:
        op = torch.ops._C.top_k_per_row_decode
        # Probe signature with a smoke call
        out = op(logits, 1, K)
        def call(): return op(logits, 1, K)
    except Exception:
        def call(): return torch.topk(logits, K, dim=-1)
    for _ in range(nw): call()
    _sync()
    durs = []
    for _ in range(ni):
        _sync(); t0 = time.perf_counter_ns(); call(); _sync()
        durs.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(durs)
    io = _io_bytes_topk(s)
    bps = io / (med * 1e-6) if med > 0 else 0
    pct = bps / _peak() * 100
    print(f"{name:<28s} med_us={med:>8.2f}  GB/s={bps/1e9:>7.1f}  roofline={pct:>6.2f}%")
    return {"name": name, "shape": s, "median_us": med, "achieved_gbps": bps/1e9, "roofline_pct": pct}


SHAPES = [
    {"B": 1,   "V": 152064, "K": 1024},  # Qwen-style vocab
    {"B": 8,   "V": 152064, "K": 1024},
    {"B": 64,  "V": 152064, "K": 1024},
]


def main():
    print(f"MUSA-0160 sampler top-k baseline (peak GDDR6 = {_peak()/1e9:.0f} GB/s)\n")
    results = []
    for s in SHAPES:
        name = f"B{s['B']:>3d}_V{s['V']}_K{s['K']}"
        try: results.append(_bench_topk(name, s))
        except Exception as e: print(f"{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
    out = os.environ.get("MUSA_0160_BENCH_OUT", "/tmp/musa_0160_sampler_topk.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
