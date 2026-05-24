#!/usr/bin/env python3
"""MUSA-0163 bench: paged_attention_v1 (csrc/attention/paged_attention_v1.cu).

Dense paged-attention fallback. mate's FA is the primary attention path on
MUSA, so this kernel rarely fires in production — but it's still on the
fallback path. Compute-bound, not bandwidth-bound.

Usage:  python3 benchmarks/op_perf/bench_paged-attention-v1-v2.py
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


def _flops(s):
    T, H, D, S = s["T"], s["H"], s["D"], s["seqlen"]
    # per token: 2 * S * D (QK^T) + 2 * S * D (PV) = 4 * S * D
    return T * H * 4 * S * D


def _peak_tflops(): return 200e12  # MTT S5000 BF16 peak nominal


def _build(s):
    T, H, Hkv, D, S, B = s["T"], s["H"], s["Hkv"], s["D"], s["seqlen"], s["block_size"]
    # Paged cache: enough blocks for T sequences of length S
    blocks_per_seq = (S + B - 1) // B
    num_blocks = max(T * blocks_per_seq + 32, 1024)
    query = torch.randn(T, H, D, device=_DEVICE, dtype=torch.bfloat16)
    key_cache = torch.randn(num_blocks, Hkv, D // 8, B, 8, device=_DEVICE, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, Hkv, D, B, device=_DEVICE, dtype=torch.bfloat16)
    out = torch.empty(T, H, D, device=_DEVICE, dtype=torch.bfloat16)
    block_tables = torch.arange(T * blocks_per_seq, device=_DEVICE, dtype=torch.int32).reshape(T, blocks_per_seq)
    seq_lens = torch.full((T,), S, device=_DEVICE, dtype=torch.int32)
    k_scale = torch.tensor(1.0, device=_DEVICE, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=_DEVICE, dtype=torch.float32)
    return (out, query, key_cache, value_cache, Hkv, 1.0 / (D ** 0.5),
            block_tables, seq_lens, B, S, None, "auto", k_scale, v_scale, 0, 0, 0, 0, 0)


def _op(t):
    torch.ops._C.paged_attention_v1(*t)


def _bench(name, s, nw=10, ni=50):
    t = _build(s)
    for _ in range(nw): _op(t)
    _sync()
    durs = []
    for _ in range(ni):
        _sync(); t0 = time.perf_counter_ns(); _op(t); _sync()
        durs.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(durs)
    f = _flops(s)
    tflops = f / (med * 1e-6) / 1e12 if med > 0 else 0
    pct = tflops * 1e12 / _peak_tflops() * 100
    print(f"{name:<32s} med_us={med:>8.2f}  TFLOPS={tflops:>6.2f}  compute_pct={pct:>6.2f}%")
    return {"name": name, "shape": s, "median_us": med, "tflops": tflops, "compute_pct": pct}


SHAPES = [
    {"T": 1,   "H": 32, "Hkv": 8, "D": 128, "seqlen": 1024, "block_size": 16},
    {"T": 1,   "H": 32, "Hkv": 8, "D": 128, "seqlen": 4096, "block_size": 16},
    {"T": 32,  "H": 32, "Hkv": 8, "D": 128, "seqlen": 4096, "block_size": 16},
]


def main():
    if getattr(torch.ops._C, "paged_attention_v1", None) is None:
        print("paged_attention_v1 binding not available — SKIP")
        return
    print(f"MUSA-0163 paged_attention_v1 baseline (peak BF16 = {_peak_tflops()/1e12:.0f} TFLOPS — nominal)\n")
    results = []
    for s in SHAPES:
        name = f"T{s['T']:>3d}_H{s['H']}_S{s['seqlen']}"
        try: results.append(_bench(name, s))
        except Exception as e: print(f"{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
    out = os.environ.get("MUSA_0163_BENCH_OUT", "/tmp/musa_0163_paged_attention.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
